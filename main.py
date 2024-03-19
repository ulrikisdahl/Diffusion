"""
Different implementations:
    - Gated Attention
    - Self-Attention
    - Multi head Self-Attention
"""


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


mean = (0.5, 0.5, 0.5)
std = (.25, .25, .25)
transform = torchvision.transforms.Compose([
    #reshape to 128x128
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])
dataset = torchvision.datasets.CIFAR10("data/cifar10", download=True, train=True, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

batch = next(iter(dataloader))
print(len(batch))
print(batch[0].shape)


def conv_block(in_channels, num_filters, filter_size, use_batchnorm=False):
    """
        Args:
            x: input tensor
            in_channels: number of channels in the input tensor
            num_filters: number of filters to use
            filter_size: size of the filter
            use_batchnorm: whether to use batch normalization
        Returns:
            tensor after applying convolution and batch normalization
    """
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=filter_size, stride=1, padding=(1,1)), #padding to prevent conv layers from reducing resolution
        nn.ReLU(),
        nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=filter_size, stride=1, padding=(1,1)),
        nn.ReLU(),
    ]

    if use_batchnorm:
        layers.insert(1, nn.BatchNorm2d(num_filters))
        layers.insert(4, nn.BatchNorm2d(num_filters))

    # Construct the Sequential block
    block = nn.Sequential(*layers)

    return block



class AttentionGate(nn.Module):
    """
        Filters out the irrelevant features of the concatenation before the concat operation
    """
    def __init__(self, x_channels, g_channels):
        super(AttentionGate, self).__init__()

        #downsample x - it is more important to preserve the features of the vector g since it is coming from a deeper layer
        self.x_downsampler = nn.Conv2d(in_channels=x_channels, out_channels=x_channels, kernel_size=2, stride=2) #x is halfed at each downsample level (and it is one level above )

        #makes g have same amount of filters as x
        self.g_conv = nn.Conv2d(in_channels=g_channels, out_channels=x_channels, kernel_size=1, stride=1)

        #collapse the channels to 1 dimension 
        self.conv_1x1 = nn.Conv2d(in_channels=x_channels, out_channels=1, kernel_size=1, stride=1) 


    def forward(self, x, g):
        """
            Args:
                x: skip connection from encoder (batch_size, channels, height, width)
                g: input from lower dim layer below (batch_size, channels, height, width)
            Returns:
                attention embedding 
        """
        x_downsampled = self.x_downsampler(x)
        g_conv = self.g_conv(g)

        x_g_sum = F.relu(x_downsampled + g_conv)
        conv_1d = self.conv_1x1(x_g_sum)

        #sigmoid layer to produce the attention coefficients
        attentions_coeffs = F.sigmoid(conv_1d)

        #upscale to the resolution for the layer above g
        scaled_coeffs = F.interpolate(attentions_coeffs, mode="bilinear", size=x.shape[2:], align_corners=False)

        #gating mechanism
        gated_features = x * scaled_coeffs

        return gated_features

        
class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()

        self.encBlock1 = conv_block(3, 64, 3)
        self.encBlock2 = conv_block(64, 128, 3)
        self.encBlock3 = conv_block(128, 256, 3)

        self.latentDimBlock = conv_block(256, 256, 3, False)

        self.latentConv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1)

        self.dec3Gate = AttentionGate(256, 256)
        self.dec2Gate = AttentionGate(128, 256)
        self.dec1Gate = AttentionGate(64, 128)

        self.decBlock3 = conv_block(512, 256, 3) #The in_channels size is double because they are concatenated along the channels dimension
        self.decBlock2 = conv_block(256, 128, 3)
        self.decBlock1 = conv_block(128, 3, 3)

        self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

    def forward(self, x):
        #(batch_size, 3, 256, 256)

        #downsampling
        enc1 = self.encBlock1(x)
        enc1_pooled = nn.MaxPool2d(kernel_size=2, stride=2)(enc1)
        enc2 = self.encBlock2(enc1_pooled)
        enc2_pooled = nn.MaxPool2d(kernel_size=2, stride=2)(enc2)
        enc3 = self.encBlock3(enc2_pooled) 
        enc3_pooled = nn.MaxPool2d(kernel_size=2, stride=2)(enc3)

        #low dimensional latent space
        latent_dim = self.latentDimBlock(enc3_pooled) #(32, 256, 256)

        #upsampling with gated attention
        dec3_gated = self.dec3Gate(enc3, latent_dim)
        latent_dim_upsampled = self.upconv3(latent_dim)
        concatenation = torch.concat((latent_dim_upsampled, dec3_gated), dim=1)
        dec3 = self.decBlock3(concatenation)

        enc2_gated = self.dec2Gate(enc2, dec3)
        dec3_upsampled = self.upconv2(dec3)
        concatenation = torch.concat((dec3_upsampled, enc2_gated), dim=1)
        dec2 = self.decBlock2(concatenation) #(128, 128, 128)


        #incoming: (64, 256, 256)
        enc1_gated = self.dec1Gate(enc1, dec2)
        dec2_upsampled = self.upconv1(dec2)
        concatenation = torch.concat((dec2_upsampled, enc1_gated), dim=1)
        dec1 = self.decBlock1(concatenation)

        return dec1
        

model = AttentionUNet()

pred = model(batch[0])
print(pred.shape)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable params: {pytorch_total_params}")

print("done")   