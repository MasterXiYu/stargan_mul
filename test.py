#-----coding:utf-8------
'''
@project:stargan
@author:yixu
@file:test.py
@ide:PyCharm
@create_time:2020/3/12 14:12
'''
import numpy as np
import torch
import torch.nn as nn

import Stargan_net
import Mul_Attribute_net
from Arcface import test

if __name__ == '__main__':
    Mul_Att_net = Mul_Attribute_net.Multi_Attribute_net()
    stargan = Stargan_net.Generator()
    Arcface = test.Arcface()
    file_dir = "/home1/yixu/yixu_project/Datasets/celeba_128"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    data = read_img.get_file(file_dir)
    dataloader = DataLoader(data, batch_size=32, shuffle=True)
    criterion_Att = nn.MSELoss()
    criterion_id = nn.CosineEmbeddingLoss()
    if cuda:
        stargan = stargan.cuda()
        Arcface = Arcface.cuda()
        Mul_Att_net = Mul_Att_net.cuda()
    stargan_optimizer = torch.optim.Adam(stargan.parameters(), lr=0.0002, betas=(0.5, 0.999))
    Mul_Att_net_optimizer = torch.optim.Adam(Mul_Att_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    epoch_num = 40
    for epoch in range(epoch_num):
        print("+++++++++++++++++++++training start+++++++++++++++++++++++")
        for batch_idx, data in enumerate(dataloader):
            img = data.to(device)
            batch_size = img.size(0)
            preds, z1, z2, z3, z4, z5 = Mul_Att_net(img)
            Y_st = stargan(preds, z1, z2, z3, z4, z5)
            img_id = Arcface(img)
            Y_st_id = Arcface(Y_st)

            loss_Atr = criterion_Att(Y_st, img)
            loss_id = criterion_id(img_id,Y_st_id)

            stargan.zero_grad()
            Mul_Att_net.zero_grad()

            loss_Atr.backward()
            loss_id.backward()

            stargan_optimizer.step()
            Mul_Att_net_optimizer.step()

            if batch_idx % 1000 == 0 and batch_size is not 0:
                print('epoch:{}, epoch_num:{}, batch_idx:{},loss:{}'.format(epoch, epoch_num, batch_idx, loss))
                path_fake = '/home1/yixu/yixu_project/CVAE-GAN/output_image/images_epoch{:02d}_batch{:03d}.jpg'.format(
                    epoch, batch_idx)
                path_real = '/home1/yixu/yixu_project/CVAE-GAN/output_image/images_epoch{:02d}_batch{:03d}_real.jpg'.format(
                    epoch, batch_idx)
                save_image(preds, path_fake, nrow=8, normalize=True)
                save_image(img, path_real, nrow=8, normalize=True)
        torch.save(net.state_dict(),
                   '/home1/yixu/yixu_project/CVAE-GAN/saved_model/Multi_Att_epoch{:02d}.pkl'.format(epoch))
        print("++++++++++++++++++++++++save++++++++++++++++++++++++++++++")







