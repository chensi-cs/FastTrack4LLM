import torch 
import torch.nn as nn 
import numpy as np



def Sinusoidal_position_embed(max_len,d_model):
    """
    生成位置编码张量。
    参数:
        d_model: 嵌入维度，默认为512。
        max_len: 最大序列长度，默认为1024。
    返回:
        位置编码张量，形状为 [1, max_len, d_model]
    """
    d_model = d_model
    max_len = max_len
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

def precompute_rope_matrix(max_len,d_model):
    """
    生成旋转位置编码矩阵
    参数：
        max_len: 最大序列长度，默认为1024。
        d_model: 嵌入维度，默认为512。
    返回：
        旋转位置编码矩阵
    """
    d_model = d_model
    max_len = max_len
    # 频率,决定不同维度的旋转速度 shape: (d_model//2,)
    theta = 10000.0
    freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
    print("freqs shape: ",freqs.shape)
    print("freqs: ",freqs)
    # 位置,[0, 1, 2, ..., max_len-1]  shape: (max_len, 1)
    t = torch.arange(max_len,dtype=torch.float32)
    print("t shape: ",t.shape)
    print("t: ",t)
    # 计算位置和频率的外积，得到每个位置在各维度的角度值 freqs[m,i] = t[m] * freqs[i]，表示位置 m 在维度 i的旋转角度
    # 旋转位置编码矩阵, 计算位置和频率的乘积 [max_len, d_model//2]
    freqs =  torch.outer(t,freqs).float()
    print("freqs shape: ",freqs.shape)
    print("freqs: ",freqs)
    # 计算余弦和正弦值，freqs_sin 和 freqs_cos 是预计算的正弦和余弦矩阵，用于对查询（query）和键（key）向量进行旋转操作。
    # freqs_sin 和 freqs_cos 具体含义：存储每个位置在不同维度上的旋转角度的正弦和余弦值
    # 这里将结果在最后一维复制，以便后续与向量的偶数和奇数维度匹配
    # shape: (max_len, d_model)
    # 为什么需要复制？
    # RoPE 将向量的每两个维度视为一个复数对（实部和虚部）。例如，对于维度 i 和 i+1：

    # freqs_sin[m, i] 和 freqs_sin[m, i+1] 存储相同的正弦值。
    # freqs_cos[m, i] 和 freqs_cos[m, i+1] 存储相同的余弦值。
    freqs_sin = torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1)
    freqs_cos = torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1)
    print("freqs_sin shape: ",freqs_sin.shape)
    print("freqs_cos shape: ",freqs_cos.shape)
    print("freqs_sin: ",freqs_sin)
    print("freqs_cos: ",freqs_cos)

    return freqs_sin,freqs_cos

def apply_rotary_pos_emb(q,k,cos,sin):
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
        print("x1 shape: ",x1.shape)
        print("x2 shape: ",x2.shape)
        print("x1: ",x1)
        print("x2: ",x2)
        res=torch.cat([-x2,x1],dim=-1)
        print("res shape: ",res.shape)
        print("res: ",res)
        return torch.cat([-x2,x1],dim=-1)
    

    print("cos shape: ",cos.shape)
    print("sin shape: ",sin.shape)
    print("cos: ",cos)
    print("sin: ",sin)
    # 调整cos和sin的形状以匹配q和k的广播需求
    cos = cos = cos.unsqueeze(1)  # shape: (seq_len, 1, d_model)
    sin = sin = sin.unsqueeze(1)  # shape: (seq_len, 1, d_model)


    print("after unsqueeze")
    print("cos shape: ",cos.shape)
    print("sin shape: ",sin.shape)
    print("cos: ",cos)
    print("sin: ",sin)

    # 应用旋转：q_rot = q·cos + rotate_half(q)·sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    print("q_embed shape: ",q_embed.shape)
    print("k_embed shape: ",k_embed.shape)
    print("q_embed: ",q_embed)
    print("k_embed: ",k_embed)
    return q_embed,k_embed
    



