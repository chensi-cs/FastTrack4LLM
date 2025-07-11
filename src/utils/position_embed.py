import torch 
import torch.nn as nn 
import numpy as np



def Sinusoidal_position_embed(config):
    """
    生成位置编码张量。
    参数:
        d_model: 嵌入维度，默认为512。
        max_len: 最大序列长度，默认为1024。
    返回:
        位置编码张量，形状为 [1, max_len, d_model]
    """
    d_model = config.d_model
    max_len = config.max_len
    pe = torch.zeros(max_len,d_model)
    # shape : (max_len, 1)
    positon = torch.arange(0,max_len).unsqueeze(1).float()
    # shape : (d_model//2,)
    div_term = torch.exp(torch.arange(0,d_model,2).float() * (-np.log(10000.0)/ d_model))
    # shape : (max_len, d_model//2)
    pe[:,0::2]=torch.sin(positon * div_term)
    pe[:,1::2]=torch.cos(positon * div_term)
    # shape : (1, max_len, d_model)
    pe = pe.unsqueeze(0)
    return pe

def precompute_rope_matrix(config):
    """
    生成旋转位置编码矩阵
    参数：
        max_len: 最大序列长度，默认为1024。
        d_model: 嵌入维度，默认为512。
    返回：
        旋转位置编码矩阵
    """
    d_model = config.d_model
    max_len = config.max_len
    # 频率,决定不同维度的旋转速度 shape: (d_model//2,)
    theta = 10000.0
    freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
    # 位置,[0, 1, 2, ..., end-1]  shape: (sequence_length, 1)
    t = torch.arange(max_len,dtype=torch.float32).unsqueeze(1)
    # 计算位置和频率的外积，得到每个位置在各维度的角度值 freqs[m,i] = t[m] * freqs[i]，表示位置 m 在维度 i的旋转角度
    # 旋转位置编码矩阵, 计算位置和频率的乘积 [seq_len, d_model//2]
    freqs =  torch.outer(t,freqs).float()
    
    # 计算余弦和正弦值
    # 这里将结果在最后一维复制，以便后续与向量的偶数和奇数维度匹配
    # shape: (seq_len, d_model)
    freqs_sin = torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1)
    freqs_cos = torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1)
    return freqs_sin,freqs_cos



def apply_rotary_pos_emb(q,k,cos,sin,position_ids=None):
    """
    应用旋转位置编码
    参数：
        q: 查询张量，形状为 [batch_size, seq_len, num_heads, head_dim]
        k: 键张量，形状为 [batch_size, seq_len, num_heads, head_dim]
        cos: 预计算的余弦值张量，形状为 [seq_len, d_model]
        sin: 预计算的正弦值张量，形状为 [seq_len, d_model]
        position_ids: 可选的位置索引张量，形状为 [seq_len]
    """
    def rotate_half(x):
        # 将向量的后半部分取负后与前半部分拼接
        # 对应复数旋转中的虚部操作
        x1,x2 =  x[...,:x.shape[-1]//2],x[...,x.shape[-1]//2:]
        return torch.cat([-x2,x1],dim=-1)
    
    # 如果提供了position_ids，则使用它们来索引预计算的cos和sin值
    # 否则默认使用连续位置索引
    if position_ids is not None:
        # shape: (seq_len, d_model)
        cos = cos[position_ids]
        # shape: (seq_len, 1, d_model)
        cos = cos.unsqueeze(1)
        # shape: (seq_len, d_model)
        sin = sin[position_ids]
        # shape: (seq_len, 1, d_model)
        sin = sin.unsqueeze(1)
    
    # 应用旋转：q_rot = q·cos + rotate_half(q)·sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed,k_embed
    



