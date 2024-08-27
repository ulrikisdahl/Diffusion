"""
    Note: Dataset paths must be set manually in configs/data_config.yaml in order to run this script
"""

from load_data.load_inpainting_data import get_inpaint_loader, show_tensor_image
import torch
from tqdm import tqdm
from statistics import mean
from diffusers import AutoencoderKL
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModel
import torch.nn.functional as F
from models.cross_attention_unet import InpaintDiffusion

#params
BATCH_SIZE=16
DEVICE="cuda"
device = DEVICE
EPOCHS = 45 #10
max_len = 16

model = InpaintDiffusion(text_seq_len=max_len)

#tokenizer = BERTTokenizer(max_length=20)
clip_model_name = "openai/clip-vit-base-patch32"
# processor = CLIPProcessor.from_pretrained(clip_model_name)
# clip_model = CLIPModel.from_pretrained(clip_model_name)
text_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
clip_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16').to(device).eval()
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#Freeze weights 
for param in vae.parameters():
    param.requires_grad = False

for param in clip_model.parameters():
    param.requires_grad = False

data_loader = get_inpaint_loader(batch_size=BATCH_SIZE)


model.to(device)
vae.to(device)
clip_model.to(device)
# checkpoint = torch.load("weights/inpainting/clip_latent_inpaint_50e.pth")
# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])



def linear_schedule(n_steps, start=0.0001, end=0.02):
    """
        Returns all the beta values indexed by their corresponding timestep
    """
    return torch.linspace(start, end, n_steps)

#Linear scheduling
n_steps = 300 # T
betas = linear_schedule(n_steps).to(DEVICE) #the variance of the noise added to the image at each time step
alphas = 1 - betas
alpha_t_cumprod = torch.cumprod(alphas, axis=0).to(DEVICE) #cumulative product of the alpha values for all time steps

def q_step(x_zero, timestep):
    """
        Forward diffusion step - uses sampling during closed form expression to return the sample for timestep t 
        Args:
            x_zero: input image of shape [batch_size, channels, height, width]
            timestep: sample time step of shape [batch_size, 1]
        Return:
            noisy image at time step t
    """
    noise = torch.randn_like(x_zero).to(DEVICE) #(b, c, h, w)
    sqrt_alpha_t_cumprod = torch.sqrt(alpha_t_cumprod[timestep]).unsqueeze(-1).unsqueeze(-1) #square root of cumulative product of alphas up to time step t
    sampled_noise = torch.sqrt(1 - alpha_t_cumprod[timestep]).unsqueeze(-1). unsqueeze(-1) * noise #scaled noise from standard normal distribution (z-distribution)
    x_t = x_zero * sqrt_alpha_t_cumprod + sampled_noise  
    return x_t, noise #return the sampled_noise which actually represents the noise added to the image at the timestep, instead of returning noise

@torch.no_grad() #avoid tracking gradient calculations in order to save memoryx
def inpaint_p_step(diffusion_model, x_t, source_image, token_conditions, t):
    """
        Preforms a reverse diffusion step. Predicts the noise that was added to the image at x_t-1 to get x_t, and subtracts it from x_t to get x_t-1
        Args:
            diffusion_model: model
            x_t: noisy image a time step t of shape [1, channels, height, width]
            token_conditions: tokenized text conditions
            t: timestep
    """
    x_t_cat = torch.cat((x_t, source_image), dim=1) 
    predicted_noise = diffusion_model(x_t_cat, token_conditions, t)
    predicted_noise_scaled = betas[t] * predicted_noise / (torch.sqrt(1 - alpha_t_cumprod[t])) #scaled predicted noise  
    subtracted_noise = x_t - predicted_noise_scaled #subtract the predicted noise from the noise of the image at timestep t to 

    one_over_sqrt_alpha = 1/torch.sqrt(alphas[t])
    x_prev = one_over_sqrt_alpha * subtracted_noise #the estimated image at time step t-1
    
    if t == 0:
        return x_prev
    
    sample_noise = torch.randn_like(x_prev) #sample from a standard normal distribution
    sample_noise_scaled = sample_noise * torch.sqrt(betas[t]) #scale the noise sample z by an approximate sigma_t term
    x_prev = x_prev + sample_noise_scaled #adding a small amount of noise back into the image has a regularizing effect
    return x_prev

@torch.no_grad()
def reverse_inpaint_sampling(diffusion_model, source_image, token_conditions, sample_steps):
    """
        Samples an image from the model by reverse sampling
    """

    saved_images = []
    x_prev = torch.randn(1, 4, 32, 32).to(DEVICE) #initial noise at time step T

    for t in range(n_steps)[::-1]:
        t = torch.tensor([t]).unsqueeze(-1).to(DEVICE)
        x_prev = inpaint_p_step(diffusion_model, x_prev, source_image, token_conditions, t)

        if t in sample_steps:
            saved_images.append(x_prev)

    print("Sampled!")
    return x_prev, saved_images

def inpaint_loss_function(model, source_image, target_image, text_conditions, timestep_t):
    """
        x_0: original image without noise
        x_t: x_0 image with noise after t forward diffusion steps
    """
    x_t, noise = q_step(target_image, timestep_t)
    x_t_cat = torch.cat((x_t, source_image), dim=1)
    noise_pred = model(x_t_cat, text_conditions, timestep_t)
    return F.l1_loss(noise, noise_pred)



print("Starting training")
for epoch in range(EPOCHS):
    losses = []
    for step, batch in enumerate(tqdm(data_loader), 0):
        optimizer.zero_grad()

        source_images = batch["source_image"].to(device)
        target_images = batch["target_image"].to(device)
        t = torch.randint(0, n_steps, (BATCH_SIZE, 1), device=DEVICE).long() #random timesteps
        #text_conditions = tokenizer.encode(batch["text"]).to(torch.float).to(device)
        with torch.no_grad():
            # inputs = processor(text=batch["text"], return_tensors="pt", padding=True, truncation=True).to(device) 
            # text_embeddings = clip_model.get_text_features(**inputs)
            tokens = text_tokenizer(batch["text"], max_length=max_len, return_tensors="pt", padding="max_length", truncation=True).to(device)
            text_embeddings = clip_model(**tokens).last_hidden_state #(bs, max_len, 512)

        #encode images to the latent space
        with torch.no_grad():
            latent_source = vae.encode(source_images, return_dict=False)[0].sample()
            latent_target = vae.encode(target_images, return_dict=False)[0].sample()

        loss = inpaint_loss_function(model, latent_source, latent_target, text_embeddings, t)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} - Loss: {mean(losses)}")

torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "weights/inpainting/clip_latent_inpaint_50e.pth")
print("Model saved!")





####### INFERENCE ########
save_steps = [0, 50, 100, 150, 200, 250]

sample_idx = 0
itr_dataloader = iter(data_loader)
itr_batch = next(itr_dataloader)


with torch.no_grad():
    # inputs = processor(text=batch["text"], return_tensors="pt", padding=True, truncation=True).to(device) 
    # text_embeddings = clip_model.get_text_features(**inputs)
    tokens = text_tokenizer(itr_batch["text"], max_length=max_len, return_tensors="pt", padding="max_length", truncation=True).to(device) 
    text_embeddings = clip_model(**tokens).last_hidden_state #(bs, max_len, 512)

for idx in range(10):
    inference_source_image = itr_batch["source_image"][idx].unsqueeze(0).to(device)
    #token = tokenizer.encode(itr_batch["text"][idx]).to(torch.float).to(device)


    with torch.no_grad():
        latent_inference_source_image = vae.encode(inference_source_image, return_dict=False)[0].sample()

    final_image, intermediary_images = reverse_inpaint_sampling(model, latent_inference_source_image, text_embeddings[idx].unsqueeze(0), sample_steps=save_steps)

    print(f"Condition: {itr_batch['text'][idx]}")


    viz = itr_batch["source_image"][idx].permute(1, 2, 0)
    viz = (viz - viz.min()) / (viz.max() - viz.min())
    plt.imshow(viz.cpu())
    plt.show()

    fig, axr = plt.subplots(1, len(save_steps) + 1, figsize=(20, 10))
    for idx2, image in enumerate(intermediary_images):

        with torch.no_grad():
            intermediary_decoded = vae.decode(intermediary_images[idx2]).sample
        img = show_tensor_image(intermediary_decoded, False)
        axr[idx2].imshow(img)

    with torch.no_grad():
        final_image_decoded = vae.decode(final_image).sample
    axr[-1].imshow(show_tensor_image(final_image_decoded, False))

    plt.show()  

