import torch
import torch.nn as nn

# 参考PEFT库中LoRA源码：peft/tuners/lora/layer.py
class LoRA(nn.Module):
    def __init__(self,in_feature,out_feature,rank,lora_alpha=16,lora_dropout=0.1):
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.dropout = nn.Dropout(lora_dropout)

        self.A = nn.Linear(in_feature,rank,bias=False)
        self.B = nn.Linear(rank,out_feature,bias=False)

        # 矩阵A高斯初始化
        torch.nn.init.normal_(self.A.weight,mean=0,std=0.02)
        # 矩阵B全0初始化
        torch.nn.init.zeros_(self.B.weight)
    
    def forward(self,x):
        return self.B(self.A(self.dropout(x))) * self.scaling
    
class LoRAWrapper(nn.Module):
    def __init__(self,base_layer,lora):
        super().__init__()
        self.base_layer = base_layer
        self.lora = lora
    def forward(self,x):
        # 原始输出 + LoRA修正量
        # with torch.no_grad():
        #     base_out = self.base_layer(x)
        base_out = self.base_layer(x)
        lora_out = self.lora(x)
        return base_out + lora_out
    
def apply_lora(model,rank=8,lora_alpha=16,device='cpu', target_modules=None):
    if target_modules is None:
        target_modules = target_modules = [
            "wq", "wk", "wv", "wo",  # 注意力层的Q/K/V/O投影
            "up_proj", "down_proj"  # 前馈网络层
        ]
    # 遍历模型，给目标层添加LoRA
    for name,module in model.named_modules():
        if isinstance(module,nn.Linear) and any(target in name for target in target_modules):
            # 如果已经加过 LoRA，就跳过
            if hasattr(module,'base_layer'):
                continue
            in_feature = module.weight.shape[1] # 输入特征维度
            out_feature = module.weight.shape[0] # 输出特征维度
            lora = LoRA(in_feature,out_feature,rank,lora_alpha) # 创建LoRA层
            lora.to(device)
            
            # 获取上一个模块
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = model.get_submodule(parent_name)
            else:
                parent_module = model
                child_name = name

            # 用包装类替换原始线性层
            setattr(parent_module, child_name, LoRAWrapper(module, lora))
    
    # 冻结原始参数，只训练 LoRA 参数
    for param in model.parameters():
        param.requires_grad = False
    for name, module in model.named_modules():
        if isinstance(module, LoRAWrapper):
            for param in module.lora.parameters():
                param.requires_grad = True

def count_lora_parameters(model):
    total_lora_params = 0
    lora_params = []
    for name,module in model.named_modules():
        if isinstance(module, LoRAWrapper):
            for param_name, param in module.lora.named_parameters():
                if param.requires_grad:
                    full_param_name = f"{name}.lora.{param_name}"
                    lora_params.append((full_param_name, param))
                    total_lora_params += param.numel()
    total_lora_params = total_lora_params / 1e6
    return total_lora_params,lora_params


def save_lora(model,path):
    state_dict={}
    for name,module in model.named_modules():
        if isinstance(module, LoRAWrapper):
            # 使用字典推导式创建一个新的字典，其中键是原有键加上 'lora'，值是对应的参数值
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            # 将 LoRA 层的状态字典添加到总的状态字典中
            state_dict.update(lora_state)
    torch.save(state_dict, path)

def load_lora(model,path):
    state_dict = torch.load(path,map_location=model.device)
    for name,module in model.named_modules():
        if isinstance(module, LoRAWrapper):
            lora_state = { k.replace(f'{name}.lora.',''): v for k,v in state_dict.items() if k.startswith(f'{name}.lora.')}
            # 加载 LoRA 层的状态字典
            module.lora.load_state_dict(lora_state)

