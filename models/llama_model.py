import torch 
import torch.nn as nn
from transformers import PreTrainedModel,GenerationMixin
from models.position_embed import *
from models.norm_data import RMSNorm
from models.attention import  MultiHeadAttention,GroupedQueryAttention
from models.feed_forward import FeedForward,MoEFeedForward
from typing import Optional, Tuple, List, Union
from transformers.modeling_outputs import CausalLMOutputWithPast

class Llama1Block(nn.Module):
    def __init__(self,layer_id,config,freqs_cos,freqs_sin):
        super().__init__()
        self.layer_id = layer_id
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.device = config.device
        freqs_cos = freqs_cos.to(self.device)
        freqs_sin = freqs_sin.to(self.device)
        self.attention =  MultiHeadAttention(config,freqs_cos, freqs_sin)
        self.ffn = FeedForward(config)
        self.norm = RMSNorm(dim=self.d_model)
        self.dropout=nn.Dropout(config.dropout)
        

        
    def forward(self,x,past_k,past_v,attention_mask):
        # Pre-normalization + RMSNorm
        x=self.norm(x)
        # 注意力机制
        attn_out,past_k,past_v=self.attention(x,past_k,past_v,attention_mask)
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
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.token_embedding = nn.Embedding(self.vocab_size,self.d_model)
        self.max_position_len = config.max_position_len
        self.freqs_cos, self.freqs_sin = precompute_rope_matrix(self.max_position_len,self.d_model //self.num_heads)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([Llama1Block(layer_id,config,self.freqs_cos, self.freqs_sin) for layer_id in range(self.num_layers)])
        self.norm = RMSNorm(dim=self.d_model)
        self.output_layer=nn.Linear(self.d_model,self.vocab_size)
    
    def forward(self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask : Optional[torch.Tensor] = None
            ):
        past_k,past_v=None,None
        x=self.token_embedding(input_ids)
        for layer in self.layers:
            x,past_k,past_v=layer(x = x,
                                  past_k = past_k,
                                  past_v = past_v,
                                  attention_mask = attention_mask)
        # 计算output之前的归一化
        x=self.norm(x)
        output=self.output_layer(x)
        return output

class Llama1ForCausalLM(PreTrainedModel,GenerationMixin):
    def __init__(self,config):
        self.config = config
        super().__init__(self.config)
        self.model=Llama1Model(config)
    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs):
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        out = CausalLMOutputWithPast (
            logits = logits,            # 预测的下一个 token 的概率分布（必需字段）
            past_key_values = None,     # 用于缓存注意力机制中的键值对（KV cache），加速后续生成步骤（可选字段）
            hidden_states = None,       # 隐藏状态（中间层输出），用于进一步分析或调试（可选字段）
            attentions = None           # 注意力权重（可选字段，通常用于可视化或分析模型注意力分布
        )
        return out

class Llama3Block(nn.Module):
    def __init__(self,layer_id,config,freqs_cos,freqs_sin):
        super().__init__()
        self.layer_id = layer_id
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.device = config.device
        freqs_cos = freqs_cos.to(self.device)
        freqs_sin = freqs_sin.to(self.device)
        self.attention = GroupedQueryAttention(config,freqs_cos,freqs_sin)
        self.ffn = MoEFeedForward(config) if config.use_moe else FeedForward(config)
        self.norm = RMSNorm(dim=self.d_model)
        self.dropout=nn.Dropout(config.dropout)
    
    def forward(self,x,past_k,past_v,attention_mask):
        # Pre-normalization + RMSNorm
        x = self.norm(x)
        # 注意力机制
        attn_out,past_k,past_v = self.attention(x,past_k,past_v,attention_mask)
        # 残差连接
        x = x + attn_out
        # Pre-normalization + RMSNorm
        x = self.norm(x)
        # 前馈神经网络
        ffn_out = self.ffn(x)
        # 残差连接
        x = x + ffn_out
        return x,past_k,past_v

class Llama3Model(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vob_size = config.vob_size
        self.num_layers = config.num_layers
        self.token_embedding = nn.Embedding(self.vob_size,self.d_model)
        self.max_position_len = config.max_position_len
        self.num_heads = config.num_heads
        self.freqs_cos, self.freqs_sin = precompute_rope_matrix(self.max_position_len, self.d_model // self.nums_heads)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([Llama3Block(layer_id,config,self.freqs_cos,self.freqs_sin) for layer_id in range(self.num_layers)])
        self.norm = RMSNorm(dim=self.d_model)
        self.output_layer = nn.Linear(self.d_model,self.vob_size)

    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None):
        x = self.token_embedding(input_ids)
        past_k,past_v=None,None
        for layer in self.layers:
            x,past_k,past_v = layer(x,past_k,past_v,attention_mask)
        # 计算output之前的归一化
        x = self.norm(x)
        output = self.output_layer(x)
        return output
    
class Llama3ForCausalLM(PreTrainedModel,GenerationMixin):
    def __init__(self,config):
        super().__init__(self.config)
        self.model = Llama3Model(config)
    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs):
        logits = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        out = CausalLMOutputWithPast(
            logits = logits,
            past_key_values = None,
            hidden_states=None,
            attentions=None
        )
        return out