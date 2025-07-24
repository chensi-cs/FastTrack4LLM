import torch 
import torch.nn as nn

from utils.position_embed import *
from utils.norm_data import RMSNorm
from utils.attention import  MultiHeadAttention
from utils.feed_forward import FeedForward

class Llama1Block(nn.Module):
    def __init__(self,layer_id,config,freqs_cos, freqs_sin):
        super().__init__()
        self.layer_id = layer_id
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.max_len = config.max_len
        self.freqs_cos, self.freqs_sin = freqs_cos, freqs_sin
        self.attention =  MultiHeadAttention(config,freqs_cos, freqs_sin)
        self.ffn = FeedForward(config)
        self.norm = RMSNorm(dim=self.d_model)
        self.dropout=nn.Dropout(config.dropout)
        

        
    def forward(self,x,past_k,past_v):
        # Pre-normalization + RMSNorm
        x=self.norm(x)
        # 注意力机制
        attn_out,past_k,past_v=self.attention(x,past_k,past_v)
        # 残差连接
        x=x+attn_out
        # Pre-normalization + RMSNorm
        x=self.norm(x)
        # 前馈神经网络
        ffn_out=self.ffn(x)
        # 残差连接
        x=x+ffn_out
        return x,past_k,past_v
    
class Llama1Model(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.max_len = config.max_len
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.token_embedding = nn.Embedding(self.vocab_size,self.d_model)
        self.freqs_cos, self.freqs_sin = precompute_rope_matrix(self.max_len,self.d_model //self.num_heads)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([Llama1Block(layer_id,config,self.freqs_cos, self.freqs_sin) for layer_id in range(self.num_layers)])
        self.norm = RMSNorm(dim=self.d_model)
        self.output_layer=nn.Linear(self.d_model,self.vocab_size)
    
    def forward(self,x):
        past_k,past_v=None,None
        x=self.token_embedding(x)
        for layer in self.layers:
            x,past_k,past_v=layer(x,past_k,past_v)
        output=self.output_layer(x)
        return output



