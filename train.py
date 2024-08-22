import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from load_data.load_data import get_data_loader, show_tensor_image, BATCH_SIZE
import matplotlib.pyplot as plt
from tqdm import tqdm
from statistics import mean
from models.attention_unet import AttUNet
from models.gated_attention_unet import AttentionUNet
from models.simple_unet import BasicUNet

#q(xt | xt-1) = N(xt, sqrt(1-b)*xt-1, b)

device = "cpu"

def linear_schedule(n_steps, start=0.0001, end=0.02):
    """
        Returns all the beta values indexed by their corresponding timestep
    """
    return torch.linspace(start, end, n_steps)

def cosine_schedule(n_steps):
    """
        Takes image of time step t as input and adds noise using cosine scheduler
        Args:
            n_steps: T from the paper
    """
    s = 0.008
    schedule = torch.linspace(0, n_steps, n_steps+1)
    f_t = lambda t: torch.cos((t / n_steps + s) / (1 + s) * (torch.pi / 2))**2
    alpha_t_cumprod = f_t(schedule) / f_t(torch.tensor([0]))
    betas = 1 - alpha_t_cumprod[1:] / alpha_t_cumprod[:-1] #shift the alpha values by 1 to calculate the beta values
    betas = torch.clip(betas, 0.0001, 0.999) #clip the betas to be no larger than 0.999
    return alpha_t_cumprod, betas


#Linear scheduling
# n_steps = 300 # T
# betas = linear_schedule(n_steps).to(device) #the variance of the noise added to the image at each time step
# alphas = 1 - betas
# alpha_t_cumprod = torch.cumprod(alphas, axis=0).to(device) #cumulative product of the alpha values for all time steps

#Cosine scheduling
n_steps = 300
alpha_t_cumprod, betas = cosine_schedule(n_steps)
alpha_t_cumprod = alpha_t_cumprod.to(device)
betas = betas.to(device)
alphas = 1 - betas


def q_step(x_zero, timestep):
    """
        Forward diffusion step - uses sampling during closed form expression to return the sample for timestep t 
        Args:
            x_zero: input image of shape [batch_size, channels, height, width]
            timestep: sample time step of shape [batch_size, 1]
        Return:
            noisy image at time step t
    """
    noise = torch.randn_like(x_zero).to(device) #(b, c, h, w)
    sqrt_alpha_t_cumprod = torch.sqrt(alpha_t_cumprod[timestep]).unsqueeze(-1).unsqueeze(-1) #square root of cumulative product of alphas up to time step t
    sampled_noise = torch.sqrt(1 - alpha_t_cumprod[timestep]).unsqueeze(-1). unsqueeze(-1) * noise #scaled noise from standard normal distribution (z-distribution)
    x_t = x_zero * sqrt_alpha_t_cumprod + sampled_noise  
    return x_t, noise

@torch.no_grad() #avoid tracking gradient calculations in order to save memoryx
def p_step(diffusion_model, x_t, t):
    """
        Preforms a reverse diffusion step. Only used for sampling during inference
    """
    predicted_noise = diffusion_model(x_t, t)
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
def reverse_sampling(diffusion_model, sample_steps):
    """
        Samples an image from the model by reverse sampling
    """

    saved_images = []
    x_prev = torch.randn(1, 3, 64, 64).to(device) #initial noise at time step T

    for t in range(n_steps)[::-1]:
        t = torch.tensor([t]).unsqueeze(-1).to(device)
        x_prev = p_step(diffusion_model, x_prev, t)

        if t in sample_steps:
            saved_images.append(x_prev)

    print("Sampled!")
    return x_prev, saved_images


def loss_function(model, x_0, timestep_t):
    """
        x_0: original image without noise
        x_t: x_0 image with noise after t forward diffusion steps
    """
    x_t, noise = q_step(x_0, timestep_t)
    loss_pred = model(x_t, t)
    return F.l1_loss(noise, loss_pred)



if __name__ == "__main__":

    model = BasicUNet()
    model.to(device)
    #model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.to(device)
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    data_loader = get_data_loader("")
    EPOCHS = 200

    print("Starting training")
    print(f"Initial weight: {torch.sum(model.middle_conv1.weight.data)}")
    for epoch in range(EPOCHS):
        losses = []
        for step, batch in enumerate(tqdm(data_loader), 0):
            optimizer.zero_grad()

            data = batch[0].to(device)

            t = torch.randint(0, n_steps, (BATCH_SIZE, 1), device=device).long()
            loss = loss_function(model, data, t)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} - Loss: {mean(losses)}")


    # torch.save({
    #     "model_state_dict": model.state_dict(),
    #     "optimizer_state_dict": optimizer.state_dict(),
    # }, "weights/ckpt200eCos.pth")

    save_steps = [0, 50, 100, 150, 200, 250]
    final_image, intermediary_images = reverse_sampling(model, sample_steps=save_steps)

    fig, axr = plt.subplots(1, len(save_steps) + 1, figsize=(20, 10))
    for idx, image in enumerate(intermediary_images):
        img = show_tensor_image(intermediary_images[idx], False)
        axr[idx].imshow(img)
    axr[-1].imshow(show_tensor_image(final_image, False))

    plt.show()  

    print("done")
