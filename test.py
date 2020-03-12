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

import torch
a = torch.ones([1,2])
b = torch.ones([1,2])
c = torch.cat([a,b],1)
print(c)
