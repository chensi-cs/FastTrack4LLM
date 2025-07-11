import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMSNorm实现
    参数：
        dim: 输入张量的最后一维大小
        eps: 数值稳定性的小常数
    """
    def __init__(self,dim,eps=1e-6):
        """
        RMSNorm实现
        参数：
            dim: 输入张量的最后一维大小
            eps: 数值稳定性的小常数
        """
        super().__init__()
        self.eps=eps
        self.dim=dim
        # 可训练的缩放参数
        self.gamma = nn.parameter(torch.ones(dim))
    
    def _rms(self,x):
        """
        计算输入张量的平方根均值
        return: 输入张量的平方根均值
        """
        # 假设 x 是一个三维张量，维度为 (batch_size, sequence_length, d_model)
        # .mean(-1, keepdim=True) 是指在最后一个维度（特征维度）上计算平均值，并保持维度不变
        return torch.sqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x):
        return self.gamma*(x/self._rms(x)).type_as(x)
    
