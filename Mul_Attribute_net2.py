#-----coding:utf-8------
'''
@project:stargan
@author:yixu
@file:Mul_Attribute_net.py
@ide:PyCharm
@create_time:2020/3/12 11:31
'''

# -*- encoding: utf-8 -*-
'''
@Author  :   {Yixu}
@Contact :   {xiyu111@mail.ustc.edu.cn}
@Software:   PyCharm
@File    :   Mul_Attribute_net.py
@Time    :   12/3/2020 1:27 PM
'''
from torchsummary import summary
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image
import torch.nn as nn
import os
import Mul_Attribute_net
from demo_on_myself import read_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class down(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(down,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01))

    def forward(self, x):
        return self.main(x)


class up(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(up, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01))

    def forward(self, x):
        return self.main(x)


class Multi_Attribute_net2(nn.Module):
    def __init__(self,image_size = 256 ,conv_dim = 64,repeat_num = 6):
        super(Multi_Attribute_net2,self).__init__()
        self.down1 = down(3,conv_dim)
        self.down2 = down(conv_dim, conv_dim*2)
        self.down3 = down(conv_dim*2, conv_dim*4)
        self.down4 = down(conv_dim*4, conv_dim*8)
        self.down5 = down(conv_dim*8, conv_dim*16)
        self.down6 = down(conv_dim*16, conv_dim*32)
        self.up1 = up(conv_dim * 32, conv_dim*16)
        self.up2 = up(conv_dim * 32, conv_dim * 8)
        self.up3 = up(conv_dim * 16, conv_dim * 4)
        self.up4 = up(conv_dim * 8, conv_dim * 2)
        self.up5 = up(conv_dim * 4, conv_dim * 1)
        self.up6 = nn.ConvTranspose2d(conv_dim, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x = self.up1(x6)
        x = self.up2(torch.cat([x,x5],dim=1))
        x = self.up3(torch.cat([x,x4],dim=1))
        x = self.up4(torch.cat([x,x3],dim=1))
        x = self.up5(torch.cat([x,x2],dim=1))
        x = self.up6(x)
        x = nn.Tanh()
        return x
    
def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        nn.init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    net = Mul_Attribute_net.Multi_Attribute_net()
    x,z1,z2,z3,z4,z5 = net(img)
    print(net)
    summary(net,(3,128,128))
    # file_dir = "/home1/yixu/yixu_project/Datasets/celeba_128"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # cuda = True if torch.cuda.is_available() else False
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # data = read_img.get_file(file_dir)
    # dataloader = DataLoader(data, batch_size=32, shuffle=True)
    #
    # criterion = nn.L1Loss()
    # if cuda:
    #     net = net.cuda()
    #
    # net_optimizer = torch.optim.Adam(net.parameters(),lr = 0.0002,betas = (0.5,0.999))
    #
    # epoch_num = 40
    # for epoch in range(epoch_num):
    #     for batch_idx, data in enumerate(dataloader):
    #         img = data.to(device)
    #         batch_size = img.size(0)
    #         preds = net(img)
    #         loss = criterion(img.reshape(-1), preds.reshape(-1))
    #         net.zero_grad()
    #         loss.backword()
    #         net_optimizer.step()
    #         print('epoch:{}, epoch_num:{}, loss:{}'.format(epoch,epoch_num,loss))
    #






