'''
@project:stargan
@author:yixu
@file:Mul_Attribute_net.py
@ide:PyCharm
@create_time:2020/3/12 11:31
'''

# -*- encoding: utf-8 -*-
from torchsummary import summary
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image
import torch.nn as nn
import os
from read_data import read_img

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class down(nn.Module):
    def __init__(self,dim_in,dim_out,kernel_size,stride,padding):
        super(down,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.01))

    def forward(self, x):
        return self.main(x)

class up(nn.Module):
    def __init__(self, dim_in, dim_out,kernel_size,stride,padding):
        super(up, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.01))

    def forward(self, x):
        return self.main(x)


class Multi_Attribute_net(nn.Module):
    def __init__(self,image_size = 256 ,conv_dim = 64,repeat_num = 6):
        super(Multi_Attribute_net,self).__init__()
        self.down1 = down(3,conv_dim,4,2,1)
        self.down2 = down(conv_dim, conv_dim*2,4,2,1)
        self.down3 = down(conv_dim*2, conv_dim*4,4,2,1)
        self.down4 = down(conv_dim*4, conv_dim*8,4,2,1)
        self.down5 = down(conv_dim*8, conv_dim*16,4,2,1)
        self.down6 = down(conv_dim*16, conv_dim*32,4,1,0)
        self.up1 = up(conv_dim * 32, conv_dim*16,4,1,0)
        self.up2 = up(conv_dim * 32, conv_dim * 8,4,2,1)
        self.up3 = up(conv_dim * 16, conv_dim * 4,4,2,1)
        self.up4 = up(conv_dim * 8, conv_dim * 2,4,2,1)
        self.up5 = up(conv_dim * 4, conv_dim * 1,4,2,1)
        self.up6 = nn.ConvTranspose2d(conv_dim*2, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x = self.up1(x6)
        z1 = torch.cat([x,x5],dim=1)
        x = self.up2(z1)
        z2 = torch.cat([x,x4],dim=1)
        x = self.up3(z2)
        z3 = torch.cat([x,x3],dim=1)
        x = self.up4(z3)
        z4 = torch.cat([x,x2],dim=1)
        x = self.up5(z4)
        z5 = torch.cat([x,x1],dim=1)
        x = self.up6(z5)
        return x,z1,z2,z3,z4,z5

def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        nn.init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    net = Multi_Attribute_net()
    file_dir = "/home1/yixu/yixu_project/Datasets/celeba_128"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    data = read_img.get_file(file_dir)
    dataloader = DataLoader(data, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    if cuda:
        net = net.cuda()
    net_optimizer = torch.optim.Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    epoch_num = 40
    for epoch in range(epoch_num):
        print("+++++++++++++++++++++training start+++++++++++++++++++++++")
        for batch_idx, data in enumerate(dataloader):
            img = data.to(device)
            batch_size = img.size(0)
            preds,z1,z2,z3,z4,z5 = net(img)
            loss = criterion(img, preds)
            net.zero_grad()
            loss.backward()
            net_optimizer.step()
            if batch_idx % 1000 == 0 and batch_size is not 0:
                print('epoch:{}, epoch_num:{}, batch_idx:{},loss:{}'.format(epoch, epoch_num,batch_idx,loss))
                path_fake = '/home1/yixu/yixu_project/CVAE-GAN/output_image/images_epoch{:02d}_batch{:03d}.jpg'.format(epoch,batch_idx)
                path_real = '/home1/yixu/yixu_project/CVAE-GAN/output_image/images_epoch{:02d}_batch{:03d}_real.jpg'.format(epoch,batch_idx)
                save_image(preds, path_fake, nrow=8, normalize=True)
                save_image(img, path_real, nrow=8, normalize=True)
        torch.save(net.state_dict(),
                   '/home1/yixu/yixu_project/CVAE-GAN/saved_model/Multi_Att_epoch{:02d}.pkl'.format(epoch))
        print("++++++++++++++++++++++++save++++++++++++++++++++++++++++++")