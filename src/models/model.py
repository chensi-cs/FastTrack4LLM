import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
from torch import Tensor

class MultiheadAttention(nn.Module):
    """
    Args:
        embed_dim: the embedding dimension of inputs
        num_heads: the number of heads in the multiheadattention
        dropout: dropout probability on attn_output_weights
        batch_first: if ``True``, then the input and output tensors are provided
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        qdim=None,
        kdim=None,
        vdim=None,
        batch_first=True    
        )->None:
        super().__init__()
        assert embed_dim % num_heads == 0 
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.dropout=dropout
        self.head_dim=self.embed_dim//self.num_heads
        self.batch_first=batch_first
        self.scale=math.sqrt(self.head_dim)
        self.qdim=qdim if qdim is not None else embed_dim
        self.kdim=kdim if kdim is not None else embed_dim
        self.vdim=vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim=self.qdim==self.kdim==self.vdim
        
        self.wq=nn.Linear(self.embed_dim,self.embed_dim)
        self.wk=nn.Linear(self.embed_dim,self.embed_dim)
        self.wv=nn.Linear(self.embed_dim,self.embed_dim)

        self.fc=nn.Linear(self.num_heads*self.head_dim,self.embed_dim)

        self.attn_dropout=nn.Dropout(self.dropout)
        self.fc_dropout=nn.Dropout(self.dropout)

    def forward(self,query,key,value,iscausal=False):
        """
        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            iscausal: whether to use causal attention (default=False)
        """
        batch_size,_,_=query.shape
        q=self.wq(query)
        k=self.wk(key)
        v=self.wv(value)

        q=q.view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
        k=k.view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)
        v=v.view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2)

        attn_scores=torch.matmul(q,k.transpose(-2,-1))/self.scale

        if iscausal is True:
            # 获取attn_scores的形状（batch_size, num_heads, seq_len, seq_len）
            batch_size, num_heads, seq_len, _ = attn_scores.shape
            # 生成上三角掩码，形状与attn_scores一致
            mask = torch.triu(torch.ones((seq_len, seq_len), device=attn_scores.device), diagonal=1)
            attn_scores = attn_scores.masked_fill(mask == 1, float('-inf'))
        attn_weights=torch.softmax(attn_scores,dim=-1)
        attn_weights=self.attn_dropout(attn_weights)
        attn_output=torch.matmul(attn_weights,v)
        output=attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.embed_dim)
        output=self.fc(output)
        output=self.fc_dropout(output)
        return output,attn_weights

class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8 ,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
        )->None:
        super().__init__()
        if activation == "relu":
            self.activation = F.relu  # 修复拼写错误：acivation -> activation
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError("activation must be relu or gelu")
        self.dropout=nn.Dropout(dropout)
        self.up_proj=nn.Linear(d_model,hidden_dim)
        self.down_proj=nn.Linear(hidden_dim,d_model)
    def forward(self,x):
        x=self.up_proj(x)
        x=self.activation(x)
        x=self.dropout(x)
        x=self.down_proj(x)
        x=self.dropout(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        max_len: int = 1024
        )->None:
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        # print("pe shape:",pe.shape)
        position=torch.arange(0,max_len).unsqueeze(1).float()
        # print("position shape:",position.shape)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-np.log(1000.0)/d_model))
        # print("div_term shape:",div_term.shape)
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe.unsqueeze(0) #[1, max_len, d_model]
        # print("pe shape:",pe.shape)
        pe=pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 注册为buffer，不会被梯度更新,'pe' 是这个张量的名称

    def forward(self,x):
        """
            x: Tensor shape [batch_size, seq_len, d_model]
        """
        return self.pe[:,:x.size(1),:]
    
    
class TransformerModel(nn.Module):
    """
    Args:
        d_model: the embedding dimensions of encoder/decoder inputs(default=512)
        nhead: the number of heads in the multiheadattention models(default=8)
        num_encoder_layers: the number of sub-encoder-layers in the encoder(default=6)
        num_decoder_layers: the number of sub-decoder-layers in the decoder(default=6)
        hidden_dim: the dimension of the feedforward network(default=2048)
        dropout: the dropout value(default=0.1)
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu. default=relu
    """
    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8,
        num_encoder_layers: int =6,
        num_decoder_layers: int =6,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        vocab_size: int = 10000,
        ) -> None:
        super().__init__()

        self.position_embedding = PositionalEncoding(d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # print("position_embedding shape:",self.position_embedding.pe.shape)

        # create encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            n_head=n_head,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation
        )

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        # create decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
            activation=activation
        )

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, d_model=d_model, vocab_size=vocab_size)

        # initialize the weights
        # self._reset_parameters()
        self.d_model = d_model
        self.n_head = n_head
    
    def _reset_parameters(self):
        """Initialize the model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src,
        tgt
        )->Tensor:
        """
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
        """
        # # print("src shape:",src.shape)
        # # print("tgt shape:",tgt.shape)

        src_token_emb = self.token_embedding(src)
        # print("src_token_emb shape:",src_token_emb.shape)
        src_position_emb = self.position_embedding(src)
        # print("src_position_emb shape:",src_position_emb.shape)
        src_embedding = src_token_emb + src_position_emb
        # print("src_embedding shape:",src_embedding.shape)
        

        tgt_token_emb = self.token_embedding(tgt)
        # print("tgt_token_emb shape:",tgt_token_emb.shape)
        tgt_position_emb = self.position_embedding(tgt)
        # print("tgt_position_emb shape:",tgt_position_emb.shape)
        tgt_embedding = tgt_token_emb + tgt_position_emb
        # print("tgt_embedding shape:",tgt_embedding.shape)

        memory=self.encoder(src_embedding)

        output=self.decoder(tgt_embedding,memory)

        return output
 
class TransformerEncoder(nn.Module):
    """
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers int the encoder (required).
    """
    def __init__(
        self,
        encoder_layer,
        num_layers
        ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        
    def forward(
        self,
        src,
        iscausal=False
        )->Tensor:
        """
        Args:
            x: the sequence to the encoder (required).
        """
        x=src
        for layer in self.layers:
            x=layer(x,iscausal=iscausal)
        return x
    
class TransformerEncoderLayer(nn.Module):
    """
    Args:
        d_model: the embedding dimensions of encoder/decoder inputs(default=512)
        nhead: the number of heads in the multiheadattention models(default=8)
        hidden_dim: the dimension of the feedforward network(default=2048)
        dropout: the dropout value(default=0.1)
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu. default=relu
    """
    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8 ,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
        ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_head, dropout)
        self.ffn=FeedForward(d_model,n_head,hidden_dim,dropout,activation)
        self.norm=nn.LayerNorm(d_model)
        
    def forward(
        self,
        src,
        iscausal=False
        ) -> Tensor:
        """
        Args:
            src: the sequence to the encoder layer (required)

            iscausal: whether to use causal attention (default=False)
        """
        # self attention
        attn_output,attn_weights = self.self_attn(src,src,src,iscausal=iscausal)
        # add and norm
        attn_output = src + attn_output
        ffn_input = self.norm(attn_output)
        # ffn
        ffn_output = self.ffn(ffn_input)
        # add and norm
        ffn_output = ffn_input + ffn_output
        output = self.norm(ffn_output)
        return output
          
class TransformerDecoder(nn.Module):
    """
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers int the encoder (required).
    """
    def __init__(
        self,
        decoder_layer,
        num_layers,
        d_model: int = 512,
        vocab_size: int = 10000
        ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)
    def forward(
            self,
            tgt,
            memory,
            iscausal=True
        ) -> Tensor:
        """
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            iscausal: whether to use causal attention (default=True)
        """
        x=tgt
        for layer in self.layers:
            x=layer(x,memory,iscausal=iscausal)
        x=self.output_layer(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_head: int = 8 ,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
        ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_head, dropout)
        self.ffn=FeedForward(d_model,n_head,hidden_dim,dropout,activation)
        self.norm=nn.LayerNorm(d_model)
    def forward(
        self,
        tgt,
        memory,
        iscausal=True
        ) -> Tensor:
        """
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            iscausal: whether to use causal attention (default=True)
        """
        # self attention
        self_attn_output,self_attn_weights = self.self_attn(tgt,tgt,tgt,iscausal=iscausal)
        # add and norm
        self_attn_output = tgt + self_attn_output
        cross_attn_input = self.norm(self_attn_output)
        # cross attention
        cross_attn_output,cross_attn_weights = self.self_attn(cross_attn_input,memory,memory)
        # add and norm
        cross_attn_output = cross_attn_input + cross_attn_output
        ffn_input = self.norm(cross_attn_output)
        # ffn
        ffn_output = self.ffn(ffn_input)
        # add and norm
        ffn_output = ffn_input + ffn_output
        output = self.norm(ffn_output)
        return output
    
