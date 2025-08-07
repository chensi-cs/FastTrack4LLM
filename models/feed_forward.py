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
        super().__init__()
        self.num_experts = config.num_experts
        self.num_independent_experts = config.num_independent_experts
        self.num_shared_experts = config.num_shared_experts
        self.experts_topk = config.experts_topk
        self.norm_topk_prob = config.norm_topk_prob
        self.d_model = config.d_model
        self.istrain = config.istrain
        self.aux_loss_alpha = config.aux_loss_alpha

        # torch.empty 创建一个未初始化的张量（tensor），仅分配内存但不赋予初始值
        # 将张量转换为 PyTorch 的 Parameter 类型，使其成为模型的可学习参数—— 会被自动注册到模型的参数列表中，参与反向传播和梯度更新。
        # self.weightMoE门控网络中最核心的可学习参数,用于判断 “输入应该由哪个专家处理”
        # [num_experts,d_model]
        self.weight = nn.Parameter(torch.empty((self.num_experts,self.d_model)))
        self.reset_parameters()

    def reset_parameters(self):
        import torch.nn.init as init
        # init.kaiming_uniform_：PyTorch 初始化模块（torch.nn.init）中的函数，用于对张量进行 Kaiming 均匀分布初始化
        # a=math.sqrt(5)：Kaiming 均匀分布初始化的参数,与激活函数的斜率相关
        init.kaiming_uniform_(self.weight,a=math.sqrt(5))

    def forward(self,x):
        batch_size, seq_len, d_model = x.shape
        # [batch_size, seq_len, d_model] > [batch_size * seq_len, d_model]
        x = x.view(batch_size * seq_len, d_model)

        # F.linear(x,self.weight,None)：PyTorch 的 F.linear 函数，用于执行矩阵乘法 + 偏置加法的线性变换操作，公式为：output = hidden_states × weight^T + bias, None 表示不使用偏置
        # logits：MoE门控网络的输出，表示每个专家的得分
        # x: [batch_size * seq_len, d_model] , self.weight: [num_experts,d_model] , logits: [batch_size * seq_len, num_experts]
        logits = F.linear(x,self.weight,None)
        # [batch_size * seq_len, num_experts]
        scores = logits.softmax(dim=-1)

        # torch.topk： 从张量中提取前 K 个最大值的函数
        # topk_scores：MoE门控网络的输出，表示每个专家的得分 [batch_size * seq_len, experts_topk]
        # topk_index：MoE门控网络的输出，表示每个专家的索引 [batch_size * seq_len, experts_topk]
        # sorted=False：是否对结果按降序排序，False 表示返回的前 K 个值不强制排序（仅保证是最大的 K 个，顺序可能随机），提高效率
        topk_weight, topk_index = torch.topk(scores,k=self.experts_topk,dim=-1,sorted=False)

        # 归一化：每个权重除以总和，确保总和为 1
        if self.experts_topk > 1  and self.norm_topk_prob:
            # 计算 Top-K 权重的总和（dim=-1在最后一个维度求和，keepdim=True保持维度用于广播）
            # topk_prob_sum：Top-K 权重的总和 [batch_size * seq_len, 1] 即前experts_topk个权重的总和
            topk_prob_sum = topk_weight.sum(dim=-1,keepdim=True) + 1e-20
            # 归一化 Top-K 权重， topk_weigh： [batch_size * seq_len, experts_topk]
            topk_weight = topk_weight / topk_prob_sum
        
        aux_loss = 0

        # 计算负载均衡损失（训练时使用），用于避免所有样本集中路由到少数几个专家，确保每个专家都能被充分利用，从而最大化模型容量的利用率。
        if self.istrain:
            """
            1. 基于 “概率分布差异” 的 KL 散度：适合于专家数量较多的情况
            思想：将专家的负载视为概率分布（expert_load），通过 KL 散度衡量其与理想均匀分布的差异，分布越不均衡，KL 散度越大，损失越大
            # 计算每个专家的负载（即被路由到该专家的样本数量）
            # 在样本维度（dim=0）做 softmax，得到每个样本对专家的概率分布，再归一化使得所有样本对专家的总 “贡献” 之和为 1。
            # expert_load： [batch_size * seq_len, num_experts]
            expert_load =  logits.softmax(dim=0)
            # expert_load： [batch_size * seq_len, num_experts] > [num_experts]
            expert_load_mean = expert_load.mean(dim=0)

            # 定义理想负载
            # 理想情况下，每个专家的负载应该相等, expert_load： [batch_size * seq_len, num_experts]
            # ideal_load： [num_experts]
            ideal_load = torch.ones_like(expert_load_mean) / self.num_experts

            # expert_load_mean_log： [num_experts]
            expert_load_mean_log = expert_load_mean.log()

            # 计算负载均衡损失
            # F.kl_div 计算 KL 散度（input，target）,要求 input 必须是对数概率分布（即 log(P)）,target 必须是概率分布（即 Q，无需取对数）
            # aux_loss 返回一个值
            aux_loss = F.kl_div(expert_load_mean_log,ideal_load,reduction='sum')
            """
            """
            2.基于 “偏好（Pi）× 负载（fi）” 的乘积惩罚不均衡现象
            Pi（偏好）：门控网络对专家 i 的平均偏好程度（通常是 scores.mean(0)，即所有样本对专家 i 的平均匹配分数）。Pi 越高，说明门控网络越倾向于选择该专家
            fi（负载）：每个专家的实际选择频率（即topk中每个专家被选中的比例）。
            aux_loss = (Pi * fi).sum() 可使得高偏好且高负载的专家受到更大的惩罚
            """
            scores_for_aux = scores
            # [batch_size * seq_len, experts_topk] > [batch_size, seq_len * experts_topk]
            topk_idx_for_aux = topk_index.view(batch_size,-1)
            # [batch_size, seq_len * experts_topk] > [batch_size * seq_len * experts_topk]
            topk_idx_for_aux = topk_idx_for_aux.view(-1)
            #  [batch_size * seq_len * experts_topk] > topk_indx_one_hot [batch_size * seq_len * experts_topk, num_experts]
            topk_indx_one_hot = F.one_hot(topk_idx_for_aux, num_classes=self.num_experts)
            # [batch_size * seq_len * experts_topk, num_experts] > topk_indx_one_hot_sum [num_experts],即每个专家的实际选择频率的平均值
            topk_indx_one_hot_mean = topk_indx_one_hot.float().mean(0)
            # 负载,即每个专家的实际选择频率*专家总数,理想值为1  [num_experts]
            fi = topk_indx_one_hot_mean * self.num_experts
            # 偏好,即门控网络对每个专家的平均偏好  [num_experts]
            pi = scores_for_aux.mean(0)
            # 损失 = 偏好 × 相对负载 的总和（惩罚高偏好且高负载的专家）
            # pi * fi :  [num_experts]
            # aux_loss:  标量
            aux_loss = (pi * fi).sum() * self.aux_loss_alpha
            
        return topk_weight,topk_index, aux_loss
    

class MoEFeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.d_model = config.d_model
        self.hidden_dim = config.hidden_dim
        self.dropout = nn.Dropout(config.dropout)

        self.num_experts = config.num_experts
        self.num_independent_experts = config.num_independent_experts
        self.num_shared_experts = config.num_shared_experts
        self.experts_topk = config.experts_topk
        self.norm_topk_prob = config.norm_topk_prob
        self.istrain = config.istrain

        self.independent_experts = nn.ModuleList([FeedForward(config) for i in range(self.num_independent_experts)])
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([FeedForward(config) for i in range(self.num_shared_experts)])

        self.gate = MoEGate(config)

    def forward(self,x):
        #  处理共享专家（所有样本都会经过所有共享专家）
        shared_experts_output = 0.0
        if self.num_shared_experts > 0:
            for expert in self.shared_experts:
                # expert(x) 的维度和输入一样，都是 [batch_size, seq_len, d_model]
                shared_experts_output += expert(x)
        # 对共享专家的输出进行平均
        # shared_experts_output 的维度和输入一样，都是 [batch_size, seq_len, d_model]，他是所有共享专家输出的和的平均
        shared_experts_output /= self.num_shared_experts

        # 处理独立专家
        raw_x = x
        raw_x_shape = raw_x.shape
        batch_size, seq_len, d_model = x.shape
        topk_weight,topk_index,aux_loss = self.gate(x)
        # topk_weight , topk_index [batch_size * seq_len, experts_topk]

        # [batch_size, seq_len, d_model] > [batch_size * seq_len, d_model]
        x = x.view(-1,d_model)

        # [batch_size * seq_len, experts_topk] > [batch_size * seq_len * experts_topk]
        # flat_topk_idx  是展平后的专家索引张量，记录了每个 “处理位置” 对应的专家编号
        flat_topk_idx = topk_index.view(-1)

        # if self.istrain:
        # 复制输入，每个token被复制top_k次（与选中的专家数量匹配）
        # [batch_size * seq_len, d_model] > [batch_size * seq_len * experts_topk, d_model]
        x = x.repeat_interleave(self.experts_topk,dim=0)

        # [batch_size * seq_len * experts_topk, d_model]
        # 初始化输出，大小与复制后的输入相同
        y = torch.empty_like(x,dtype=torch.float16)

        for i,expert in enumerate(self.independent_experts):
            # mask : [batch_size * seq_len * experts_topk],值为True或False,表示该token是否由当前专家i处理
            mask = (flat_topk_idx == i )
            # x[mask] 是应该由当前专家i处理的输入，expert(x[mask]) 是当前专家i处理后的输
            y[mask] = expert(x[mask]).to(y.dtype)

        # 融合专家输出：按权重加权求和
        # y.view(batch_size * seq_len,self.experts_topk,d_model): [batch_size * seq_len * experts_topk, d_model] > [batch_size * seq_len, experts_topk, d_model]
        # topk_weight.unsqueeze(-1) : [batch_size * seq_len, experts_topk] > [batch_size * seq_len, experts_topk, 1]
        # y.view(batch_size * seq_len,self.experts_topk,d_model) * topk_weight.unsqueeze(-1) : [batch_size * seq_len, experts_topk, d_model]
        # .sum(dim=1): [batch_size * seq_len, d_model]
        # .sum(dim=1) 乘权重后在top_k维度求和，得到每个token的最终输出
        y = (y.view(batch_size * seq_len,self.experts_topk,d_model)) * topk_weight.unsqueeze(-1).sum(dim=1)
        # y 的维度和输入一样，都是 [batch_size, seq_len, d_model]
        y = y.view(raw_x_shape)
        if self.num_shared_experts > 0:
            y += shared_experts_output
        return y      

        
        