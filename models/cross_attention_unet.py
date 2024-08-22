import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, heads, input_channels, height, width, text_emb_dim):
        super().__init__()
        self.num_heads = heads
        self.head_dim = input_channels // heads
        self.attention_dim = 10 #Q, K, V are all projected to this dimension
        self.input_channels = input_channels
        self.output_channels = input_channels #in_channels = out_channels: becomes the common attention-dimension 

        self.query = nn.Linear(in_features=input_channels, out_features=self.output_channels)
        self.key = nn.Linear(in_features=text_emb_dim, out_features=self.output_channels) #transform text embedding from text_emb_dim to attention dim
        self.value = nn.Linear(in_features=text_emb_dim, out_features=self.output_channels)

        self.output_layer = nn.Linear(in_features=self.output_channels, out_features=self.output_channels)
        
    def forward(self, img, text_emb):
        """
            Args:
                img: feature map of shape [batch_size, channels, height, width]
                text_emb: text embedding of shape [batch_size, seq_length, text_emb_dim]
        """

        height = img.size(2) #move to device?
        width = img.size(3)
        img = img.flatten(start_dim=-2, end_dim=-1).transpose(1, 2) #(bs, c, h, w) -> (bs, c, h*w) -> (bs, h*w, c), operate on the channels as attention-dimension

        Q = self.query(img)
        K = self.key(text_emb)
        V = self.value(text_emb)

        Q_heads = Q.reshape(img.size(0), img.size(1), self.num_heads, self.head_dim).transpose(1, 2) #(b, h*w, c) -> (b, h*w, head, head_d) -> (b, head, h*w, head_d)
        K_heads = K.reshape(img.size(0), K.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        V_heads = V.reshape(img.size(0), V.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        dim_sqrt = torch.sqrt(torch.tensor([self.head_dim], device=img.device))
        attention_scores = torch.einsum("bhqd,bhkd -> bhqk", Q_heads, K_heads) / dim_sqrt  #(seq_q, seq_k)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attented_values = torch.einsum("bhqk,bhvd -> bhqd", attention_weights, V_heads) #(bs, heads, h*w, head_d)
        attented_values = attented_values.transpose(1, 2).reshape(img.size(0), img.size(1), self.output_channels) #(bs, h*w, c)

        output = self.output_layer(attented_values)
        return output.transpose(1, 2).reshape(img.size(0), self.output_channels, height, width) #(bs, c, h, w)


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


class ConvBlock(nn.Module): #IDEA: ADD SKIP CONNECTIONS?
    def __init__(self, input_channels, output_channels, embedding_dim, upsample):
        super().__init__()
        self.upsample = upsample
        self.silu = nn.SiLU()

        #adapts the embedding dimension to match the number of channels for the conv blocks
        self.time_embedding_adapter = nn.Linear(in_features=embedding_dim, out_features=output_channels)

        #up or down sampling
        if not upsample:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels=2*input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1) #double the expected input channels because of concatenation

        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x, t_embedding):
        """
            Args:
                x: image
                t_embedding: time embedding
        """
        x = self.silu(self.bn1(self.conv1(x)))
        emb = self.time_embedding_adapter(t_embedding) #adapt embedding dimensions to the number of channels of the conv output
        emb = self.silu(emb)
        emb = torch.unsqueeze(torch.unsqueeze(emb, -1), -1) #reshape
        x = x + emb 
        x = self.silu(self.bn2(self.conv2(x)))  
        return x


class InpaintDiffusion(nn.Module):
    """
        Diffusion model for inpainting images
    """
    def __init__(self, text_seq_len):
        super().__init__()
        self.channel_latent_dim = 4
        self.embedding_dim = 128 #time embedding dim
        self.bert_length = text_seq_len #max_len
        self.bert_dim = 512 #embedding dimension for each token
        #self.text_embedder = TextTransformer(self.bert_length, self.bert_dim)
        self.image_channels = 3

        self.time_embedder = nn.Sequential(
            SinusoidalEmbedding(self.embedding_dim),
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
            nn.ReLU()
        )

        #downsampling blocks
        self.downsample_idx = [0, 2, 4, 6]
        self.downsample_blocks = nn.ModuleList([]) #64 -> 32 (64) -> 16(128) -> 8(256) -> 4(512)
        self.downsample_blocks.append(ConvBlock( #r=64
            input_channels=self.channel_latent_dim*2, #concatenate source and target image
            output_channels=64,
            embedding_dim=self.embedding_dim,
            upsample=False
        ))
        self.downsample_blocks.append(ConvBlock( #r=32
            input_channels=64,
            output_channels=128,
            embedding_dim=self.embedding_dim,
            upsample=False
        ))
        self.downsample_blocks.append(CrossAttentionBlock(
            heads=4,
            input_channels=128,
            height=0,
            width=0,
            text_emb_dim=self.bert_dim
        ))
        self.downsample_blocks.append(ConvBlock(
            input_channels=128,
            output_channels=256,
            embedding_dim=self.embedding_dim,
            upsample=False
        ))
        self.downsample_blocks.append(CrossAttentionBlock(
            heads=4,
            input_channels=256,
            height=0,
            width=0,
            text_emb_dim=self.bert_dim
        ))
        self.downsample_blocks.append(ConvBlock(
            input_channels=256,
            output_channels=512,
            embedding_dim=self.embedding_dim,
            upsample=False
        ))
        self.downsample_blocks.append(CrossAttentionBlock(
            heads=4,
            input_channels=512,
            height=0,
            width=0,
            text_emb_dim=self.bert_dim
        ))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #middle block
        self.middle_conv1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.middle_time_embedder = nn.Linear(in_features=self.embedding_dim, out_features=1024)
        self.middle_conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.middle_attention_block = CrossAttentionBlock(heads=4, input_channels=1024, height=0, width=0, text_emb_dim=self.bert_dim)
        self.middle_bn1 = nn.BatchNorm2d(1024)
        self.middle_bn2 = nn.BatchNorm2d(1024)
        self.middle_bn3 = nn.BatchNorm2d(512)
        self.middle_upsample = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2) #r=8

        #upsampling blocks
        self.upsampling_blocks = nn.ModuleList([])
        self.upsampling_blocks.append(ConvBlock( 
            input_channels=512,
            output_channels=512,
            embedding_dim=self.embedding_dim,
            upsample=True
        ))
        self.upsampling_blocks.append(CrossAttentionBlock(
            heads=4,
            input_channels=512, 
            height=0,
            width=0,
            text_emb_dim=self.bert_dim
        ))
        self.upsampling_blocks.append(ConvBlock( #r=16
            input_channels=256,
            output_channels=256,
            embedding_dim=self.embedding_dim,
            upsample=True
        ))
        self.upsampling_blocks.append(CrossAttentionBlock(
            heads=4,
            input_channels=256, 
            height=0,
            width=0,
            text_emb_dim=self.bert_dim
        ))
        self.upsampling_blocks.append(ConvBlock( #r=32
            input_channels=128,
            output_channels=128,
            embedding_dim=self.embedding_dim,
            upsample=True
        ))
        self.upsampling_blocks.append(CrossAttentionBlock(
            heads=4,
            input_channels=128, 
            height=0,
            width=0,
            text_emb_dim=self.bert_dim
        ))
        self.upsampling_blocks.append(ConvBlock( #r=64
            input_channels=64,
            output_channels=64,
            embedding_dim=self.embedding_dim,
            upsample=True
        ))

        #Define the layers that actually do the upsampling at the end of the upsampling block
        self.upsampling_channels = [512, 256, 128]
        self.upsamplers = nn.ModuleList([])
        for channel in self.upsampling_channels:
            self.upsamplers.append(
                nn.ConvTranspose2d(in_channels=channel, out_channels=channel//2, kernel_size=2, stride=2)
            )
        self.upsampleNorms = nn.ModuleList([
            nn.BatchNorm2d(256), nn.BatchNorm2d(128), nn.BatchNorm2d(64) 
        ])

        #final output layer to project the channel space to image_channels dimension
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=self.channel_latent_dim, kernel_size=1)

    def forward(self, x, text, time_t):
        """
            Args:
                x: noisy image at x_t concatenated with guiding image. Of shape [batch_size, image_channels*2, height, width]
                text: clip text embedding #text: batch of text conditional strings of shape [batch_size, length]
                time_t: timestep of shape [batch_size, 1]
        """
        
        time_embedding = self.time_embedder(time_t)
        text_embedding = text #self.text_embedder(text)

        
        #downsampling
        skip_connections = []
        for idx, downsample_block in enumerate(self.downsample_blocks):
            if isinstance(downsample_block, ConvBlock):
                x = downsample_block(x, time_embedding)
                if idx == 0: #clunky
                    skip_connections.append(x)
            else:
                x = downsample_block(x, text_embedding) 
                skip_connections.append(x)

            if idx in self.downsample_idx: #these are the blocks where we downsample afterwards
                x = self.maxpool(x)

        #middle block
        x = torch.relu(self.middle_bn1(self.middle_conv1(x)))
        middle_time_embedding = self.middle_time_embedder(time_embedding)
        middle_time_embedding = torch.unsqueeze(torch.unsqueeze(middle_time_embedding, -1), -1)
        x = x + middle_time_embedding
        x = torch.relu(self.middle_bn2(self.middle_conv2(x))) #change to SiLU?
        x = self.middle_attention_block(x, text_embedding) 
        x = torch.relu(self.middle_bn3(self.middle_upsample(x))) 

        #upsampling
        for idx, upsample_block in enumerate(self.upsampling_blocks):
            if isinstance(upsample_block, ConvBlock):
                x = torch.cat((x, skip_connections[-1 - idx//2]), dim=1)
                x = upsample_block(x, time_embedding) #conv block
            else:
                x = upsample_block(x, text_embedding) #attention block
            
            if idx % 2 != 0: #upsample after every attention block, and they all have odd index in self.upsample_blocks
                x = torch.relu(self.upsampleNorms[idx//2](self.upsamplers[idx//2](x)))

        x = self.out_conv(x)
        return x    



#Ideas
# - Attention in the first input layer (ish) for the two images that are concatenated