# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from torch import nn
from timm.layers import LayerNorm2d

#this is a small block used in the mask head of the model. 
#It consists of a convolutional layer followed by an activation function and another convolutional layer, 
#with a normalization layer at the end. 
#The purpose of this block is to process the features extracted by the encoder and produce refined features that 
#can be used to generate accurate segmentation masks for each query. 
#The conv1_layer parameter allows for flexibility in choosing the type of convolutional layer used in the first convolution operation, 
#which can be either a standard convolution or a transposed convolution (used for upsampling).

#image
#   ↓
#ViT patch features
#   ↓
#query-mask interaction
#   ↓
#low-resolution mask features
#   ↓
#SCALEBLOCK UNSAMPLING
#   ↓
#higher-resolution features
#   ↓
#final segmentation masks

class ScaleBlock(nn.Module):
    def __init__(self, embed_dim, conv1_layer=nn.ConvTranspose2d):
        super().__init__()

        self.conv1 = conv1_layer(
            embed_dim,
            embed_dim,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x
