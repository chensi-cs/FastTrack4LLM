import torch
import torch.nn as nn
import math
from  models.position_embed import apply_rotary_pos_emb
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, config,freqs_cos = None ,freqs_sin = None):
        super().__init__()
        self.d_model = config.d_model
        # 注意力头数
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        # 确保注意力头数是d_model的倍数
        assert self.d_model % self.num_heads == 0
        # 每个注意力头的维度
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
        self.use_kv_cache = config.use_kv_cache
        self.istrain = config.istrain
        self.flash_att = config.flash_att and hasattr(torch.nn.functional,'scaled_dot_product_attention')

        if self.position_type == "rope":
            self.freqs_cos = freqs_cos
            self.freqs_sin = freqs_sin
        

    def forward(self,x,kv_cache=None,attention_mask=None):
        batch_size, seq_len , _ = x.shape
        # 计算查询、键、值
        # q,k,v :[batch, seq_len, d_model]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        # q,k,v :[batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size,seq_len,self.num_heads,self.head_dim)
        k = k.view(batch_size,seq_len,self.num_heads,self.head_dim)
        v = v.view(batch_size,seq_len,self.num_heads,self.head_dim)

        # 应用 rotary positional embedding
        if self.position_type == "rope":
            q, k = apply_rotary_pos_emb(q,k,self.freqs_cos,self.freqs_sin)


        # 应用 kv cache
        if  self.use_kv_cache:
            # 如果启用了kv_cache，则将当前k和v与缓存的k和v拼接
            if kv_cache is not None:
                k = torch.cat([kv_cache[0],k],dim=1)
                v = torch.cat([kv_cache[1],v],dim=1)
            kv_cache = (k,v)
        else:
            kv_cache = None

        # [batch, seq_len, num_heads, head_dim] > [batch, num_heads, seq_len, head_dim]
        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)

        # attention_mask [batch_size, seq_len]
        if attention_mask is not None:
            attenion_mask = attenion_mask.view(batch_size,1,1,-1)
            attenion_mask = attention_mask.expand(batch_size,self.num_heads,seq_len,-1)
            
        if  self.flash_att:
            # print("apply flash attention")
            dropout_p = self.dropout if self.istrain else 0.0
            attn_output = F.scaled_dot_product_attention(q,k,v,dropout_p=dropout_p,is_causal=self.iscausal,attn_mask=attenion_mask)

        else:
            # 计算注意力分数
            # [batch, num_heads, seq_len, seq_len]
            attn_scores = torch.matmul(q,k.transpose(-2,-1))/self.scale

            if self.iscausal:
                batch_size, num_heads, seq_len, _= attn_scores.shape
                mask = torch.triu(torch.ones(seq_len,seq_len),diagonal=1)
                mask = mask.to(attn_scores.device)
                attn_scores =  attn_scores.masked_fill(mask==1,float('-inf'))

            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(attention_mask==0,float('-inf'))

            attn_weights = torch.softmax(attn_scores,dim=-1)

            attn_weights = self.attn_dropout(attn_weights)

            # 计算注意力输出
            # [batch, num_heads, seq_len, head_dim]
            attn_output = torch.matmul(attn_weights,v)

        # [batch, num_heads, seq_len, head_dim] > [batch, seq_len, num_heads, head_dim] > [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)

        output = self.wo(attn_output)

        output = self.out_dropout(output)
        return output,kv_cache



class GroupedQueryAttention(nn.Module):
    def __init__(self,config, freqs_cos=None, freqs_sin=None):
        super().__init__()

        self.d_model = config.d_model
        # 注意力头数
        self.num_heads = config.num_heads
        # 注意力组数
        self.num_groups = config.num_groups
        # 确保注意力头数是注意力组数的倍数
        assert self.num_heads % self.num_groups == 0
        # 每个注意力组的注意力头数
        self.num_heads_per_group = self.num_heads // self.num_groups
        # 确保注意力头数是d_model的倍数
        assert self.d_model % self.num_heads == 0 
        # 每个注意力头的维度
        self.head_dim = self.d_model // self.num_heads

        # 查询权重，维度为[d_model, d_model] = [d_model, num_heads * head_dim]，即每个头的权重都不一样
        self.wq = nn.Linear(self.d_model, self.d_model)
        # 键权重， 维度为 [d_model, num_groups * head_dim], 即每个组的键权重不一样，但是组内的多个头的键权重共享
        self.wk = nn.Linear(self.d_model, self.num_groups * self.head_dim)
        # 值权重， 维度为 [d_model, num_groups * head_dim], 即每个组的值权重不一样，但是组内的多个头的值权重共享
        self.wv = nn.Linear(self.d_model, self.num_groups * self.head_dim)
        # 输出投影层，将注意力结果映射回原始维度
        self.wo = nn.Linear(self.d_model, self.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        self.scale = math.sqrt(self.head_dim)
        self.position_type = config.position_type
        self.iscausal = config.iscausal
        self.use_kv_cache = config.kv_cache
        self.istrain = config.istrain
        self.flash_att = config.flash_att and hasattr(torch.nn.functional,'scaled_dot_product_attention')

        if self.position_type == "rope":
            self.freqs_cos = freqs_cos
            self.freqs_sin = freqs_sin

    def forward(self,x,cache_k=None,cache_v=None,attention_mask=None):
        batch_size, seq_len, d_model = x.shape

        # 计算查询 [batch_size, seq_len, d_model] > [batch_size, seq_len, d_model] 
        q = self.wq(x)
        # 计算键 [batch_size, seq_len, d_model] > [batch_size, seq_len, num_groups * head_dim] 
        k = self.wk(x)
        # 计算值 [batch_size, seq_len, d_model] > [batch_size, seq_len, num_groups * head_dim] 
        v = self.wv(x)

        # [batch_size, seq_len, d_model] > [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)  
        # [batch_size, seq_len, num_groups * head_dim]  > [batch_size, seq_len, num_groups, head_dim]
        k = k.view(batch_size, seq_len, self.num_groups, self.head_dim) 
        #  [batch_size, seq_len, num_groups * head_dim]  > [batch_size, seq_len, num_groups, head_dim]
        v = v.view(batch_size, seq_len, self.num_groups, self.head_dim)

        if self.position_type == "rope":
            q, k = apply_rotary_pos_emb(q,k,self.freqs_cos,self.freqs_sin)

        # [batch_size, seq_len, num_heads, head_dim] > [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1,2)
        # [batch_size, seq_len, num_groups, head_dim] > [batch_size, num_groups, seq_len, head_dim]
        k = k.transpose(1,2)
        # [batch_size, seq_len, num_groups, head_dim] > [batch_size, num_groups, seq_len, head_dim]
        v = v.transpose(1,2)

        # 将每组的键和值复制到组内的每个头
        # [batch_size, num_groups, seq_len, head_dim] > [batch_size, num_groups, num_heads_per_group, seq_len, head_dim] > [batch_size, num_heads, seq_len, head_dim]
        # k.unsqueeze(2) : [batch_size, num_groups, seq_len, head_dim] > [batch_size, num_groups, 1, seq_len, head_dim] 
        # k.repeat(1,1,self.num_heads_per_group,1,1) : [batch_size, num_groups, 1, seq_len, head_dim] > [batch_size, num_groups, num_heads_per_group, seq_len, head_dim] 
        # k.view(batch_size, self.num_heads, seq_len, self.head_dim) : [batch_size, num_groups, num_heads_per_group, seq_len, head_dim] > [batch_size, num_groups * num_heads_per_group = num_heads, seq_len, head_dim] 
        k = k.unsqueeze(2).repeat(1,1,self.num_heads_per_group,1,1).view(batch_size, self.num_heads, seq_len, self.head_dim)

        # v同样 [batch_size, num_groups, seq_len, head_dim] > [batch_size, num_groups, num_heads_per_group, seq_len, head_dim] > [batch_size, num_heads, seq_len, head_dim]
        
        v = v.unsqueeze(2).repeat(1,1,self.num_heads_per_group,1,1).view(batch_size, self.num_heads, seq_len, self.head_dim)

        # attention_mask [batch_size, seq_len]
        if attention_mask is not None :
            # [batch_size, seq_len] > [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.view(batch_size,1,1,seq_len)
            attention_mask = attention_mask.expand(batch_size, self.num_heads, seq_len, seq_len)
        
        if  self.flash_att:
            dropout_p = self.dropout if self.istrain else 0.0
            attn_output = F.scaled_dot_product_attention(q,k,v,dropout_p=dropout_p,is_causal=self.iscausal,attn_mask=attention_mask)
        else:
            # 计算注意力分数
            # [batch_size, num_heads, seq_len, seq_len]
            attn_scores = torch.matmul(q,k.transpose(-2,-1)) / self.scale

            if self.iscausal:
                # 创建一个上三角矩阵，主对角线以上的元素为1
                mask = torch.tril(torch.ones(seq_len,seq_len),diagonal=1)
                mask = mask.to(attn_scores.device)
                # 用mask填充attn_scores,将attn_scores中主对角线以上的元素（为1）填充为-inf
                attn_scores = attn_scores.masked_fill(mask==1,float('-inf'))

            # 用attention_mask填充attn_scores,将attn_scores中attention_mask为0的元素填充为-inf
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(attention_mask==0,float('-inf'))
        
            # 计算注意力权重
            attn_weights = torch.softmax(attn_scores,dim=-1)
            # dropout
            attn_weights = self.attn_dropout(attn_weights)
            # 计算注意力输出
            # [batch_size, num_heads, seq_len, seq_len] * [batch_size, num_heads, seq_len, head_dim] = [batch_size, num_heads, seq_len, head_dim]
            attn_output = torch.matmul(attn_weights,v)
        
        # attn_output.transpose(1,2) : [batch_size, num_heads, seq_len, head_dim] >[batch_size, seq_len, num_heads, head_dim]
        # .view(batch_size, seq_len, self.d_model) : [batch_size, seq_len, num_heads, head_dim] > [batch_size, seq_len, num_heads*head_dim=d_model]
        attn_output = attn_output.transpose(1,2).view(batch_size, seq_len, self.d_model)

        # [batch_size, seq_len, num_heads*head_dim=d_model] > [batch_size, seq_len, d_model]
        output = self.wo(attn_output)
        # dropout
        output = self.out_dropout(output)
        
        return output,k,v













        
