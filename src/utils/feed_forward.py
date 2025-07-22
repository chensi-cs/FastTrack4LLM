import torch
import torch.nn as nn
from utils.activations import relu, gelu,silu

class FeedForward(nn.Module):
    def __inin__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.hidden_dim = config.hidden_dim
        if config.activation == "relu":
            self.activation = relu
        elif config.activation == "gelu":
            self.activation = gelu
        elif config.activation == "silu":
            self.activation = silu
        else:
            raise ValueError("activation must be one of ['relu', 'gelu', 'silu']")
        self.up_proj = nn.Linear(self.d_model,self.hidden_dim)
        self.down_proj = nn.Linear(self.hidden_dim,self.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 上升维度
        # (batch_size, seq_len, d_model) > (batch_size, seq_len, hidden_dim)
        x = self.up_proj(x)
        # 应用激活函数
        x = self.activation(x)
        # 应用dropout
        x = self.dropout(x)
        # 下降维度
        # (batch_size, seq_len, hidden_dim) > (batch_size, seq_len, d_model)
        x = self.down_proj(x)
        # 应用dropout
        x = self.dropout(x)
        return x        

