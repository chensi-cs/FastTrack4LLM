import torch 
import torch.nn as nn
from ../utils/position_embed import Position_embed
from ../utils/norm_data import Norm_data

class Llama3Block(nn.Module):
    def __init__(self,layer_id,config):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        
        
class Llama3Model(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.token_embedding = nn.Embedding(self.vocab_size,self.embed_dim)
        self.position_embedding = Position_embed(self.config)
        self.n_head = config.n_head
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([Llama3Block(layer_id,config) for layer_id in range(self.num_layers)])
        self.norm = Norm_data(self.config)

