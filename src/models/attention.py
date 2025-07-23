import torch
import torch.nn as nn
import math

class MultiheadAttention(nn.Module):
    """
    Args:
        embed_dim: the embedding dimension of inputs
        num_heads: the number of heads in the multiheadattention
        dropout: dropout probability on attn_output_weights
    """
    def __init__(
        self,
        embed_dim,
        num_heads,dropout=0.0,
        qdim=None,
        kdim=None,
        vdim=None,
        batch_first=False
        )->None:
        super().__init__()
        assert embed_dim % num_heads == 0 
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.dropout=dropout
        self.head_dim=self.embed_dim//self.num_heads
        self.batch_first=batch_first
        self.scale=math.sqrt(self.head_dim)
        # self.qdim=qdim if qdim is not None else embed_dim
        # self.kdim=kdim if kdim is not None else embed_dim
        # self.vdim=vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim=self.qdim==self.kdim==self.vdim
        
        self.wq=nn.Linear(self.embed_dim,self.embed_dim)
        self.wk=nn.Linear(self.embed_dim,self.embed_dim)
        self.wv=nn.Linear(self.embed_dim,self.embed_dim)

        self.fc=nn.Linear(self.num_heads*self.head_dim,self.embed_dim)

        self.attn_dropout=nn.Dropout(self.dropout)
        self.fc_dropout=nn.Dropout(self.dropout)

    def forward(self,x,decoder_mask=False):
        batch_size,seq_len,_=x.shape
        q=self.wq(x)
        k=self.wk(x)
        v=self.wv(x)

        q=q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k=k.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v=v.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        attn_scores=torch.matmul(q,k.transpose(-2,-1))/self.scale

        if decoder_mask is True:
            mask = torch.triu(torch.ones(seq_len,seq_len), diagonal=1)# 生成上三角部分为 1（不包括主对角线）,其余为0
            attn_scores=attn_scores.masked_fill(mask==1,float('-inf'))
        attn_weights=torch.softmax(attn_scores,dim=-1)
        attn_weights=self.attn_dropout(attn_weights)
        attn_output=torch.matmul(attn_weights,v)
        output=attn_output.transpose(1,2).contiguous().view(batch_size,seq_len,self.embed_dim)
        output=self.fc(output)
        output=self.fc_dropout(output)
        return output,attn_weights






        

        
