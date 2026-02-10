from typing import List
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv, kernel_size=3, stride=1, padding=1, mid_channels=None):
        super(ConvBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), 
            nn.BatchNorm2d(num_features=mid_channels),
            nn.LeakyReLU(negative_slope=.02, inplace=True),
        ]

        for _ in range(max(n_conv-1,0)):
            layers.append(nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.LeakyReLU(negative_slope=.02, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, filter_per_level, unet_depth, n_conv):
        super(EncoderBlock, self).__init__()
        self.module_dict = nn.ModuleDict()

        for n in range(unet_depth):
            if n == 0:
                self.module_dict["conv_stack_{}".format(n)] = ConvBlock(in_channels=1, out_channels=filter_per_level[n], n_conv=n_conv)
            else:
                self.module_dict["conv_stack_{}".format(n)] = ConvBlock(in_channels=filter_per_level[n-1], out_channels=filter_per_level[n], n_conv=n_conv)
            # self.module_dict["down_sample_{}".format(n)] = nn.Conv2d(in_channels=filter_per_level[n], out_channels=filter_per_level[n], kernel_size=2, stride=2)
            self.module_dict["down_sample_{}".format(n)] = nn.MaxPool2d(2)
        
        self.module_dict["bottleneck"] = ConvBlock(in_channels=filter_per_level[unet_depth-1], out_channels=filter_per_level[unet_depth-1], n_conv=n_conv, mid_channels=filter_per_level[unet_depth])
    
    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith('down'):
                down_sampling_features.append(x)
            x = op(x)
        return x, down_sampling_features

class DecoderBlock(nn.Module):
    def __init__(self, filter_per_level, unet_depth, n_conv):
        super(DecoderBlock, self).__init__()
        self.module_dict = nn.ModuleDict()
        for n in reversed(range(unet_depth)):
            # self.module_dict["up_sample_{}".format(n)] = nn.Sequential(
                    # nn.Conv2d(filter_per_level[n], filter_per_level[n]*4, kernel_size=3, padding=1),
                    # nn.PixelShuffle(2),
                    # )
            self.module_dict["up_sample_{}".format(n)] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if n:
                self.module_dict["conv_stack_{}".format(n)] = ConvBlock(filter_per_level[n]*2, filter_per_level[n-1],n_conv=n_conv, mid_channels=filter_per_level[n])
            else:
                self.module_dict["conv_stack_{}".format(n)] = ConvBlock(filter_per_level[n]*2, filter_per_level[n],n_conv=n_conv, mid_channels=filter_per_level[n])

    def forward(self, x, down_sampling_features: List[torch.Tensor]):
        for k, op in self.module_dict.items():
            x=op(x)
            if k.startswith("up"):
                x = torch.cat((down_sampling_features.pop(), x), dim=1)
        return x

class Unet(nn.Module):
    def __init__(self,filter_base=64,unet_depth=4, add_last=True):
        super(Unet, self).__init__()
        self.add_last = add_last
        filter_per_level = [filter_base * 2**i for i in range(unet_depth+1)]
        n_conv = 2 # conv number in ConvBlock

        self.encoder = EncoderBlock(filter_per_level=filter_per_level, unet_depth=unet_depth, n_conv=n_conv)
        self.decoder = DecoderBlock(filter_per_level=filter_per_level, unet_depth=unet_depth, n_conv=n_conv)
        self.final = nn.Conv2d(in_channels=filter_base, out_channels=1, kernel_size=1)
       
    def forward(self, x):
        residual_input = x
        x, down_sampling_features = self.encoder(x)
        x = self.decoder(x, down_sampling_features)
        y = self.final(x)
        if self.add_last:
            y += residual_input
        return y
