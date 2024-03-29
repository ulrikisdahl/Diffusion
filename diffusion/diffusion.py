import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from load_data.load_data import get_data_loader, show_tensor_image, BATCH_SIZE
import matplotlib.pyplot as plt
from tqdm import tqdm

#q(xt | xt-1) = N(xt, sqrt(1-b)*xt-1, b)


def cosine_schedule(x_t):
    """
        Takes image of time step t as input and adds noise using cosine scheduler
    """

    return

def linear_schedule(n_steps, start=0.0001, end=0.02):
    """
        Returns all the beta values indexed by their corresponding timestep
    """
    return torch.linspace(start, end, n_steps)


#Can be fixed for time step T
n_steps = 300
t = 200
betas = linear_schedule(n_steps) 
alphas = 1 - betas
alpha_t_cumprod = torch.cumprod(alphas, axis=0) #cumulative product of the alpha values for all time steps

def q_step(x_zero, timestep):
    """
        Forward diffusion step - uses sampling during closed form expression to return the sample for timestep t 
        Args:
            x_zero: input image at time step 0
            timestep: time step to sample from
        Return:
            noisy image at time step t
    """
    noise = torch.rand_like(x_zero)
    sqrt_alpha_t_cumprod = torch.sqrt(alpha_t_cumprod[timestep]) #square root of cumulative product of alphas up to time step t
    sampled_noise = torch.sqrt(1 - alpha_t_cumprod[timestep]) * noise #scaled noise from standard normal distribution (z-distribution)
    x_t = x_zero * sqrt_alpha_t_cumprod + sampled_noise  
    return x_t, noise

@torch.no_grad() #avoid tracking gradient calculations in order to save memoryx
def p_step(x_t, t):
    """
        Preforms a reverse diffusion step. Only used for sampling during inference
    """
    one_over_sqrt_alpha = 1/alphas[t]
    predicted_noise = betas[t] * model(x_t, t) * 1 / (torch.sqrt(betas[t])) #scaled predicted noise  
    subtracted_noise = x_t - predicted_noise #subtract the predicted noise from the noise of the image at timestep t
    
    x_prev = one_over_sqrt_alpha * subtracted_noise #the estimated image at time step t-1
    return x_prev




def p_step(x_t):
    """
        Reverse diffusion step
    """
    return 


def loss(model, x_0, timestep_t):
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
        embeddings = torch.zeros((time_t.shape[0], self.embedding_dim))

        position = torch.arange(0, self.embedding_dim, dtype=torch.float32)
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

        #adapts the embedding dimension to suit the number of channels for the conv blocks
        self.time_embedding_adapter = nn.Linear(in_features=embedding_dim, out_features=output_channels)

        #up or down sampling
        if not upsample:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels=2*input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1) #double the expected input channels because of concatenation
            self.rescale = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels // 2, kernel_size=2, stride=2) #doubles the resolution
            
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t_embedding):
        x = torch.relu(self.conv1(x))
        emb = self.time_embedding_adapter(t_embedding) #adapt embedding dimensions to the number of channels of the conv output
        emb = torch.unsqueeze(torch.unsqueeze(emb, -1), -1) #reshape
        x = x + emb
        x = torch.relu(self.conv2(x))
        if self.upsample:
            x = self.rescale(x)
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

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle_conv1 = nn.Conv2d(in_channels=self.downsample_channels[-1], out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.middle_conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.middle_upsample = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)

        self.downsample_blocks = []
        for idx in range(len(self.downsample_channels) - 1): 
            self.downsample_blocks.append(ConvBlock(
                input_channels=self.downsample_channels[idx], 
                output_channels=self.downsample_channels[idx + 1], 
                embedding_dim=self.embedding_dim, 
                upsample=False))
            
        self.upsample_blocks = []
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
            skip_connections.append(x) #last x has 512 channels with h,w
            x = self.maxpool(x) #512 channels with h/2,w/2

        x = self.middle_conv1(x)
        x = self.middle_conv2(x)
        x = self.middle_upsample(x)

        for idx, upsample_block in enumerate(self.upsample_blocks):
            x = torch.cat((x, skip_connections[-idx - 1]), dim=1)
            x = upsample_block(x, time_embedding)

        x = self.out_conv(x)
        return x





device = "cuda"
model = BasicUNet()
model.to(device)

data_loader = get_data_loader("").to(device)

EPOCHS = 10
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(EPOCHS):
    for step, batch in enumerate(tqdm(data_loader), 0):
        optimizer.zero_grad()

        t = torch.randint(0, n_steps, (BATCH_SIZE, 1), device=device).long()
        loss = loss(model, batch[0], t)

        loss.backward()
        loss.step()











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
