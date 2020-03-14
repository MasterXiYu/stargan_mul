import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

import Mul_Attribute_net


class ResidualBlock1(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv2(x) + self.conv1(x)


class ResidualBlock2(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock2, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return self.main(x)


class up(nn.Module):
    def __init__(self, dim_in, dim_out,kernel_size,stride,padding):
        super(up, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(dim_out),
            nn.LeakyReLU(0.01))

    def forward(self, x):
        return self.main(x)

class down(nn.Module):
    def __init__(self,dim_in,dim_out,kernel_size,stride,padding):
        super(down,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(dim_out),
            nn.ReLU())

    def forward(self, x):
        return self.main(x)



class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64):
        super(Generator, self).__init__()
        self.down1 = down(3,conv_dim,kernel_size=7,stride=1,padding=3)
        self.down2 = down(conv_dim,conv_dim*2,kernel_size=4,stride=2,padding=1)
        self.down3 = down(conv_dim*2, conv_dim * 4, kernel_size=4, stride=2, padding=1)
        self.res1_1 = ResidualBlock1(conv_dim * 8,conv_dim*4)
        self.res1_2 = ResidualBlock1(conv_dim * 8, conv_dim * 4)
        self.res1_3 = ResidualBlock1(conv_dim * 8, conv_dim * 4)
        self.res1_4 = ResidualBlock1(conv_dim * 8, conv_dim * 4)
        self.res1_5 = ResidualBlock1(conv_dim * 8, conv_dim * 4)
        self.res2_1 = ResidualBlock2(conv_dim * 4, conv_dim * 4)
        self.up1 = up(conv_dim * 4,conv_dim * 2,kernel_size=4,stride=2,padding=1)
        self.up2 = up(conv_dim * 2, conv_dim , kernel_size=4, stride=2, padding=1)
        self.up3 = up(conv_dim , 3, kernel_size=7, stride=1, padding=3)

    def forward(self, x,z1,z2,z3,z4,z5):
    #def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        print(x.size())
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        print(x.size())
        print(z1.size())
        x = self.res1_1(torch.cat([x,z1],dim=1))
        x = self.res1_2(torch.cat([x,z2],dim=1))
        x = self.res1_3(torch.cat([x,z3],dim=1))
        x = self.res1_4(torch.cat([x,z4],dim=1))
        x = self.res1_5(torch.cat([x,z5],dim=1))
        x = self.res2_1(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=2, stride=1, padding=0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        real_or_fake = self.main(x)
        return real_or_fake





