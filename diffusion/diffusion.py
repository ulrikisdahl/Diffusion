import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from load_data.load_data import get_data_loader, show_tensor_image, BATCH_SIZE
import matplotlib.pyplot as plt
from tqdm import tqdm
from statistics import mean

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


class SinusoidalEmbedding(nn.Module):
    """
        Creates a time embedding vector for the current time step. Not really a embedding, but rather an encoding used as input for embedding layer
        Args:
            embedding_dim: dimension of the embedding vector
        Returns:
            batch of embedding vectors of shape [batch_size, embedding_dim]
    """
    def __init__(self, embedding_dim):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimensions must be divisible by 2")
        
        self.embedding_dim = embedding_dim

    def forward(self, time_t):
        """
            time_t: current time step for sample. shape of [batch_size, 1]
        """
        device = time_t.device
        embeddings = torch.zeros((time_t.shape[0], self.embedding_dim), device=device)

        position = torch.arange(0, self.embedding_dim, dtype=torch.float32, device=device)
        div_term = torch.pow(10000, (2 * position // 2) / self.embedding_dim)

        #sine encoding for even indices
        embeddings[:, 0::2] = torch.sin(time_t / div_term[0::2])
        
        #cosine encoding for odd indices
        embeddings[:, 1::2] = torch.cos(time_t / div_term[1::2])
        return embeddings


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, embedding_dim, upsample):
        super().__init__()
        self.upsample = upsample

        #adapts the embedding dimension to match the number of channels for the conv blocks
        self.time_embedding_adapter = nn.Linear(in_features=embedding_dim, out_features=output_channels) 

        #up or down sampling
        if not upsample:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels=2*input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1) #double the expected input channels because of concatenation
            self.rescale = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels // 2, kernel_size=2, stride=2) #doubles the resolution
            self.bn3 = nn.BatchNorm2d(output_channels // 2)

        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x, t_embedding):
        x = torch.relu(self.bn1(self.conv1(x)))
        emb = self.time_embedding_adapter(t_embedding) #adapt embedding dimensions to the number of channels of the conv output
        emb = torch.relu(emb)
        emb = torch.unsqueeze(torch.unsqueeze(emb, -1), -1) #reshape
        x = x + emb 
        x = torch.relu(self.bn2(self.conv2(x)))  
        if self.upsample:
            x = torch.relu(self.bn3(self.rescale(x)))
        return x


class BasicUNet(nn.Module):
    """
        Classical U-Net with only positional embeddings
    """
    def __init__(self):
        super().__init__()

        self.downsample_channels = [3, 64, 128, 256, 512]
        self.upsample_channels = self.downsample_channels[::-1]
        self.embedding_dim = 30
        self.image_channels = 3
        self.output_channels = 3

        self.embedding_layer = nn.Sequential(
            SinusoidalEmbedding(self.embedding_dim),
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
            nn.ReLU()
        )

        self.downsample_blocks = nn.ModuleList([])
        for idx in range(len(self.downsample_channels) - 1): 
            self.downsample_blocks.append(ConvBlock(
                input_channels=self.downsample_channels[idx], 
                output_channels=self.downsample_channels[idx + 1], 
                embedding_dim=self.embedding_dim, 
                upsample=False))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle_conv1 = nn.Conv2d(in_channels=self.downsample_channels[-1], out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.middle_conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.middle_upsample = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.middel_layer_embedder = nn.Linear(in_features=self.embedding_dim, out_features=1024)
        self.bn_middle_1 = nn.BatchNorm2d(1024)
        self.bn_middle_2 = nn.BatchNorm2d(1024)
            
        self.upsample_blocks = nn.ModuleList([])
        for idx in range(len(self.upsample_channels) - 2):
            self.upsample_blocks.append(ConvBlock(
                input_channels=self.upsample_channels[idx], 
                output_channels=self.upsample_channels[idx], 
                embedding_dim=self.embedding_dim, 
                upsample=True))
        self.upsample_blocks.append(
            ConvBlock(input_channels=2*64, output_channels=64, embedding_dim=self.embedding_dim, upsample=False) #last block without upsampling
        )
            
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=self.output_channels, kernel_size=1)

    def forward(self, x, time_t):
        time_embedding = self.embedding_layer(time_t)

        skip_connections = []
        for downsample_block in self.downsample_blocks:
            x = downsample_block(x, time_embedding)
            skip_connections.append(x) 
            x = self.maxpool(x) 

        x = torch.relu(self.bn_middle_1(self.middle_conv1(x))) 
        middle_embedding = self.middel_layer_embedder(time_embedding)
        middle_embedding = torch.unsqueeze(torch.unsqueeze(middle_embedding, -1), -1)
        x = x + middle_embedding
        x = torch.relu(self.bn_middle_2(self.middle_conv2(x)))
        x = torch.relu(self.middle_upsample(x))

        for idx, upsample_block in enumerate(self.upsample_blocks):
            x = torch.cat((x, skip_connections[-idx - 1]), dim=1)
            x = upsample_block(x, time_embedding)

        x = self.out_conv(x) #IDEA: Add tanh activation
        return x





if __name__ == "__main__":

    #load model
    #checkpoint = torch.load("weights/ckpt250e.pth")
    model = BasicUNet()
    model.to(device)
    #model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.to(device)
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(model)

    data_loader = get_data_loader("")
    EPOCHS = 200
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

            # if epoch % 5 == 0 and step == 0:
            #     reverse_sampling(model)


    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, "weights/ckpt200eCos.pth")
    print("Model saved!")

    save_steps = [0, 50, 100, 150, 200, 250]
    final_image, intermediary_images = reverse_sampling(model, sample_steps=save_steps)

    fig, axr = plt.subplots(1, len(save_steps) + 1, figsize=(20, 10))
    for idx, image in enumerate(intermediary_images):
        img = show_tensor_image(intermediary_images[idx], False)
        axr[idx].imshow(img)
    axr[-1].imshow(show_tensor_image(final_image, False))

    plt.show()  



    #simulate forward diffusion
    # data_loader = get_data_loader("")
    # image = next(iter(data_loader))[0]

    # step_size = int(n_steps/10)
    # for idx in range(0, n_steps, step_size):
    #     t = torch.Tensor([idx]).type(torch.int64)
    #     plt.subplot(1, 10+1, int(idx/step_size) + 1)
    #     img, noise = q_step(image, t)
    #     show_tensor_image(img)

    # plt.show()

    # print(image.shape)



    print("done")
