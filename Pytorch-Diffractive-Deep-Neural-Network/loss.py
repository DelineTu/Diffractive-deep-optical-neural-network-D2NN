# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:41:18 2023

@author: sleepingcat
github: https://github.com/sleepingcat42
e-mail: sleepingcat@aliyun.com
"""
import torch

# 计算归一化点积相关系数（Normalized Pointwise Cross Correlation, NPCC）损失的函数，用于衡量两张量间的相似度
def npcc_loss(X, Y):
    # dim=(2,3) 指的是沿着张量的第 2 维和第 3 维进行操作
    # “mean”是指对此张量的元素取均值，并保持了维度不变。结果的形状应为 (batch_size, channels, 1, 1)。也就是对长宽分别求均值
    # 注意区分“维度”与“某一维度的长度”
    X_mean = torch.mean(X, dim=(2,3), keepdim=True)
    Y_mean = torch.mean(Y,dim=(2,3), keepdim=True)
    # a与b都是张量
    # X 和 Y 在去除各自的均值之后的点积，同样保留了维度
    a = torch.sum((X - X_mean)*(Y - Y_mean), dim=(2,3), keepdim=True)
    # X 和 Y 在去除各自的均值之后的标准差的乘积
    b = torch.sqrt(torch.sum((X - X_mean)**2, dim=(2,3), keepdim=True) *
                   torch.sum((Y - Y_mean)**2, dim=(2,3), keepdim=True) + 1e-8)
    # 归一化的点积相关系数，并取负平均值作为损失值。
    # 不再保持维度，对二维张量求均值，得到常数
    #npcc = torch.mean(-a/b)

    # 归一化的点积相关系数
    npcc = torch.mean(a / b, dim=(2, 3), keepdim=True)

    # 将npcc转换为损失形式
    loss = 1 - npcc

    # 对整个batch的loss求平均
    loss = torch.mean(loss)

    return loss