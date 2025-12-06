# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:41:18 2023

@author: sleepingcat
github: https://github.com/sleepingcat42
e-mail: sleepingcat@aliyun.com
"""

import torch
from torch import nn
from torch.fft import fft2, fftshift, ifft2, ifftshift


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PI = torch.pi

# M: sampling rate
# L: Length of the network
# lambda0: wavelength

##相位调制层
class DiffLayer(nn.Module):
    def __init__(self, M, L, lambda0, z):
        super(DiffLayer, self).__init__()
        # self.params=nn.Parameter(torch.rand(1, M, M))*2*torch.pi
        # params为可调节参数，代表调制相位角，为(1, M, M)形状的张量
        # 第一维代表batch数，为1即用单个样本来更新参数；余下二维为变换平平面。设置初始值为0
        self.params=nn.Parameter(torch.zeros(1, M, M))
        # 传播核 H，依赖于网格尺寸 M、波长 lambda0 和传播距离 z。
        self.H = self.get_kernel(M, L, lambda0, z)

    # 对输入信号u1调制相位，并通过傅里叶变换和逆傅里叶变换进行传播
    def forward(self, u1):
        # u1乘params相位因子
        u1 = u1*torch.exp(1j*self.params)
        # 二维傅里叶变换
        U1 = fft2(fftshift(u1))
        # 乘以传播核 H（对应频率域传播特性）
        U2 = U1*self.H
        # 逆傅里叶变换 ifft2 得到输出
        return ifftshift(ifft2(U2))

    #####传播核H的物理意义（调制层/自由空间中的空间传播相位变化？）和计算原理（二维频率域与lambda0，传播因子A的计算，传播核H的计算）
    # 计算了传播核 H，它是一个复数矩阵，描述了信号在自由空间传播时的相位变化
    # 根据给定的参数 M（网格大小，长宽分割数目）、L（网格宽度，左右实际宽度）、lambda0（波长）和 z（传播距离）来计算传播核 H。
    def get_kernel(self, M, L, lambda0, z):

        # dx 表示每个采样点之间的间隔，也就是网格的分辨率。L 是整个网格的宽度，M 是网格的尺寸
        dx = L/M
        # 波数
        k = 2 * PI / lambda0
        # 频率网格 FX 和 FY 用于描述在频域中的频率分量。linspace 函数创建了一个从 -1/(2*dx) 到 1/(2*dx)-1/L 的线性间隔向量，包含 M 个元素。这些值代表了频率域中的频率分量
        # fx 是一维的频率向量，FX 和 FY 是将这个频率向量扩展成二维网格
        # 这样每个点 (FX[i,j], FY[i,j]) 表示一个特定的频率组合（[i，j]表示点）
        fx = torch.linspace(-1/(2*dx), 1/(2*dx)-1/L, M)   
        FX, FY = torch.meshgrid(fx, fx, indexing='xy')

        # 传播因子 A，它描述频率分量在传播过程中的变化。
        A=1 - ((lambda0 *FX)**2 + (lambda0 *FY)**2)
        # A 被转换成了复数类型，以便后续能够进行复数指数运算。
        A = A+0j
        # 传播核 H 是通过对传播因子 A 应用复数指数运算来计算的。
        # 这里 torch.sqrt(A) 计算了传播因子的平方根，1j * k * z * torch.sqrt(A) 计算了传播路径上的相位延迟
        # torch.exp() 将这个相位延迟转换为复数形式的传播核。
        H = torch.exp(1j * k * z * torch.sqrt(A))
        H = fftshift(H)
        H = H.to('cuda:0')
        return H.unsqueeze(0)    

    # 允许手动初始化相位参数 params
    def phase_init(self, phase):
        self.params = phase

##自由空间传播层
# 通过傅里叶变换和逆傅里叶变换来进行信号传播。区别在于它没有可学习的参数，而是直接计算传播效果。
class Diffraction(nn.Module):
    def __init__(self, M, L, lambda0, z):
        super(Diffraction, self).__init__()
        # 计算了传播核 H 和网格坐标 X 和 Y。
        self.H = self.get_kernel(M, L, lambda0, z)
        # 网格坐标 X 和 Y，这些坐标用于后续的计算
        self.X, self.Y = self.get_gridXY(M, L)
        # 网格的宽度
        #####此处L为DOE size？对应什么物理过程？
        self.L = L

    # forward 方法,只乘以传播核
    def forward(self, u1):

        U1 = fft2(fftshift(u1))
        U2 = U1*self.H
        return ifftshift(ifft2(U2))

    # 计算传播核 H，类似 DiffLayer。
    def get_kernel(self, M, L, lambda0, z):
        
        dx = L/M
        k = 2 * PI / lambda0
        fx = torch.linspace(-1/(2*dx), 1/(2*dx)-1/L, M)   
        FX, FY = torch.meshgrid(fx, fx, indexing='ij')
        
        A= 1 - ((lambda0 *FX)**2 + (lambda0 *FY)**2)
        A = A+0j
        H = torch.exp(1j * k * z * torch.sqrt(A))
        H = fftshift(H)
        H = H.to('cuda:0')
        return H.unsqueeze(0)    

    # 生成网格坐标 X 和 Y，用于后续计算。
    def get_gridXY(self, M, L):
        dx = L/M
        x = torch.linspace(-L/2, L/2-dx, M).to(device)   
        X, Y = torch.meshgrid(x,x, indexing='ij')
        return X, Y
