import torch
import torch.nn as nn


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
