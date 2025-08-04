import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.hidden_dim = config.hidden_dim
        if config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "gelu":
            self.activation = F.gelu
        elif config.activation == "silu":
            self.activation = F.silu
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

class MoEGate(nn.Module):
    def __init__(self,config):
        super.__init__()
        self.num_experts = config.num_experts
        self.num_independent_experts = config.num_independent_experts
        self.num_shared_experts = config.num_shared_experts
        self.experts_topk = config.experts_topk
        self.norm_topk_prob = config.norm_topk_prob
        self.d_model = config.d_model
        self.istrain = config.istrain

        # torch.empty 创建一个未初始化的张量（tensor），仅分配内存但不赋予初始值
        # 将张量转换为 PyTorch 的 Parameter 类型，使其成为模型的可学习参数—— 会被自动注册到模型的参数列表中，参与反向传播和梯度更新。
        # self.weightMoE门控网络中最核心的可学习参数,用于判断 “输入应该由哪个专家处理”
        self.weight = nn.Parameter(torch.empty((self.num_experts,self.d_model)))
        self.reset_parameters()

    def reset_parameters(self):
        import torch.nn.init as init
        # init.kaiming_uniform_：PyTorch 初始化模块（torch.nn.init）中的函数，用于对张量进行 Kaiming 均匀分布初始化
        # a=math.sqrt(5)：Kaiming 均匀分布初始化的参数,与激活函数的斜率相关
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))

    def forward(self,x):
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size * seq_len, d_model)
        #  None 表示不使用偏置
        # F.linear(x,self.weight,None)：PyTorch 的 F.linear 函数，用于执行矩阵乘法 + 偏置加法的线性变换操作，公式为：output = hidden_states × weight^T + bias
        # logits：MoE门控网络的输出，表示每个专家的得分
        logits = F.linear(x,self.weight,None)

        scores = logits.softmax(dim=-1)
        # torch.topk： 从张量中提取前 K 个最大值的函数
        # topk_scores：MoE门控网络的输出，表示每个专家的得分
        # topk_indices：MoE门控网络的输出，表示每个专家的索引
        # sorted=False：是否对结果按降序排序，False 表示返回的前 K 个值不强制排序（仅保证是最大的 K 个，顺序可能随机），提高效率
        topk_weight, topk_index = torch.topk(scores,k=self.experts_topk,dim=-1,sorted=False)

        # 归一化：每个权重除以总和，确保总和为 1
        if self.experts_topk > 1  and self.norm_topk_prob:
            # 计算 Top-K 权重的总和（dim=-1在最后一个维度求和，keepdim=True保持维度用于广播）
            topk_prob_sum = topk_weight.sum(dim=-1,keepdim=True) + 1e-20
            topk_weight = topk_weight / topk_prob_sum
        
        # 计算负载均衡损失（训练时使用），用于避免所有样本集中路由到少数几个专家，确保每个专家都能被充分利用，从而最大化模型容量的利用率。
        
        return topk_weight,topk_index
    




