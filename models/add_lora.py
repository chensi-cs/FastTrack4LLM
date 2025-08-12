import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self,in_feature,out_feature,rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_feature,rank,bias=False)
        self.B = nn.Linear(rank,out_feature,bias=False)
        # 矩阵A高斯初始化
        torch.nn.init.normal_(self.A.weight,mean=0,std=0.02)
        # 矩阵B全0初始化
        torch.nn.init.zeros_(self.B.weight)
    
    def forward(self,x):
        return self.B(self.A(x))
    
def apply_lora(model,rank=8,device='cpu'):
    # 遍历模型所有模块,返回模块名称（name）和模块实例（module）
    for name,module in model.named_modules():
        # 如果模块是线性层
        if isinstance(module,nn.Linear) and 'lora' not in name:

            in_feature = module.weight.shape[1] # 输入特征维度
            out_feature = module.weight.shape[0] # 输出特征维度
            lora =  LoRA(in_feature,out_feature,rank) # 创建LoRA层
            lora.to(device)
            
            setattr(module,'lora',lora) # 将LoRA层添加到

            original_forward = module.forward

            def lora_forward(x,original_forward=original_forward,lora=lora):
                return original_forward(x) + lora(x)
            # 替换原有的forward方法
            module.forward = lora_forward
 

def save_lora(model,path):
    state_dict={}
    for name,module in model.named_modules():
        if hasattr(module,'lora'):
            # state_dict() 是 PyTorch 中 nn.Module 类的核心方法，用于返回一个参数字典,字典的键（key） 是参数的名称,值（value） 是参数的值
            # items() 是 Python 字典的内置方法，用于返回字典中所有键值对（key-value pairs）的迭代器
            # 这里使用了字典推导式来创建一个新的字典，其中键是原有键加上前缀 'name.lora.'，值是对应的参数值
            lora_state = { f'{name}.lora.{k}':v for k,v in module.lora.state_dict().items()}
            # 将 LoRA 层的状态字典添加到总的状态字典中
            state_dict.update(lora_state)

    torch.save(state_dict, path)

def load_lora(model,path):
    state_dict = torch.load(path,map_location=model.device)
    for name,module in model.named_modules():
        if hasattr(module,'lora'):
            lora_state = { k.replace(f'{name}.lora.',''): v for k,v in state_dict.items() if f'{name}.lora.' in k}
            # 加载 LoRA 层的状态字典
            module.lora.load_state_dict(lora_state)

