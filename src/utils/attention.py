import torch
import torch.nn as nn
import math
from utils.position_embed import apply_rotary_pos_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, config,freqs_cos = None  ,freqs_sin = None):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.head_dim = self.d_model // self.num_heads

        self.wq = nn.Linear(self.d_model, self.d_model)
        self.wk = nn.Linear(self.d_model, self.d_model)
        self.wv = nn.Linear(self.d_model, self.d_model)
        self.wo = nn.Linear(self.d_model, self.d_model)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.out_dropout = nn.Dropout(self.dropout)

        self.scale = math.sqrt(self.head_dim)
        self.position_type = config.position_type
        self.iscausal = config.iscausal
        self.kv_cache = config.kv_cache
        if self.position_type == "rope":
            self.freqs_cos = freqs_cos
            self.freqs_sin = freqs_sin
        

    def forward(self,x,cache_k=None,cache_v=None):
        batch_size, seq_len , _ = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        # 将张量从 [batch, seq_len, num_heads, head_dim] 重排为 [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        # 应用 rotary positional embedding
        if self.position_type == "rope":
            self.freqs_cos = self.freqs_cos[:seq_len]
            self.freqs_sin = self.freqs_sin[:seq_len]
            q, k = apply_rotary_pos_emb(q,k,self.freqs_cos,self.freqs_sin)

        # 应用 kv cache
        if  self.kv_cache:
            # 如果启用了kv_cache，则将当前k和v与缓存的k和v拼接
            if cache_k is not None and cache_v is not None:
                k = torch.cat([self.cache_k,k],dim=2)
                v = torch.cat([self.cache_v,v],dim=2)
        
        # 计算注意力分数
        # [batch, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q,k.transpose(-2,-1))/self.scale
        
        if self.iscausal:
            batch_size, num_heads, seq_len, _= attn_scores.shape
            mask = torch.triu(torch.ones(seq_len,seq_len),diagonal=1)
            attn_scores =  attn_scores.masked_fill(mask==1,float('-inf'))
        attn_weights = torch.softmax(attn_scores,dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        # 计算注意力输出
        # [batch, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights,v)
        # [batch, num_heads, seq_len, head_dim] > [batch, seq_len, num_heads, head_dim] > [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        output = self.wo(attn_output)
        output = self.out_dropout(output)
        return output,k,v






        
