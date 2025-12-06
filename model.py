# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:05:41 2022

@author: Chen Chunyuan
"""
# import torch
from torch import nn
from OpticalLayers import DiffLayer,  Diffraction

class Onn(nn.Module):
    #####自由空间之后直接进行三次相位调制得到结果，中间不再需要自由空间传播？是否与真实过程不同？
    # 定义传播层
    def __init__(self, M, L, lambda0, z):
        super(Onn, self).__init__()

        # 自由空间传播
        self.layer0 = Diffraction(M, L, lambda0, z[0])

        # 相位调制层
        self.DiffLayer1 = DiffLayer(M, L, lambda0, z[1])
        self.DiffLayer2 = DiffLayer(M, L, lambda0, z[2])
        self.DiffLayer3 = DiffLayer(M, L, lambda0, z[3])

    # 进行变换
    def forward(self, u1):        
        u = self.layer0(u1)
        u = self.DiffLayer1(u)
        u = self.DiffLayer2(u)
        u = self.DiffLayer3(u)

        return u

#####频率为400GHz？

######################## Optical parameters /mm ##########################
#c = 3e8*1e3             # speed of light
#f = 400e9               # 400GHz
#lambda0 = c/f           # wavelength
#L = 80                  # （衍射光学元件）DOE size

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#M = 256                 # sample numbers
#z = [30,30,30,30]
#onn = Onn(M, L, lambda0, z).to(device)
