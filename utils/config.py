import torch
from transformers import PretrainedConfig

class Config(PretrainedConfig):
    """ 基础配置类，继承自PretrainedConfig，方便与transformers库集成"""
    def __init__(self):
        super().__init__()
        # 训练参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32 # batch size
        self.base_batch_size = 32 # base batch size
        self.num_epochs = 1 # number of epochs
        self.lr = 5e-4 # learning rate
        self.base_lr = 5e-4 # base learning rate
        self.accumulation_steps = 8 # gradient accumulation steps
        self.optimizer = 'adamw'
        self.patience = 5 # early stopping patience
        self.grad_clip = 1.0
        self.ddp = False
        self.use_wandb = False # use Weights & Biases for logging 
        self.use_tensorboard = False # use tensorboard for logging
        self.log_interval = 100 # log interval for training
        self.save_interval = 100
        self.evaluate_val = False
        self.evaluate_test = False
        self.use_amp = True
        self.loss_mask = False # 是否使用loss mask
        self.attn_mask = False # 是否使用attention mask
        # self.num_workers = 4

        # 模型参数
        self.model = 'llama1'
        self.d_model = 512 # embedding dimension for one token
        self.max_seq_len = 512 # max length of input sequence
        self.max_position_len = 32768 # max length of position embedding
        self.vocab_size = 6400 # vocabulary size
        self.hidden_dim = 1024 # hidden dimension for feedforward network
        self.num_layers = 8 # number of layers
        self.dropout = 0.0 # dropout rate
        self.position_type = 'rope'
        self.activation = 'silu'
        self.use_kv_cache = False
        self.flash_att = True
        self.iscausal = True
        self.istrain = True

        # MoE参数
        self.use_moe = False
        self.num_experts = 4
        self.num_independent_experts = 3
        self.num_shared_experts = 1
        self.experts_topk = 1
        self.norm_topk_prob = True # 是否标准化top-k概率
        self.aux_loss_alpha = 0.1
        self.add_aux_loss = False # 是否添加auxiliary loss


        # Attention参数
        self.attention_type = 'MHA'
        self.num_heads = 8 # number of heads for multi-head attention
        self.num_groups = 4

        
        # 路径参数
        self.data_path = 'data/model_data/train.json' # train data path
        self.val_path = 'data/model_data/val.json' # validation data path
        self.test_path = 'data/model_data/test.json' # test data path
        self.tokenizer_path = 'data/' # tokenizer path
        self.model_save_path = 'saved_models' # model save path
        self.model_result_path = 'all_models'
        self.log_dir = 'logs' # log directory
        self.checkpoint_path = 'checkpoints' # checkpoint path
        self.pretrain_path = 'pretrained_models' # pretrained model path
        
        # lora
        self.lora_rank = 8

        # 蒸馏
        self.kl_alpha = 0.1
        self.kl_temperature = 1

        # dpo
        self.dpo_beta = 0.1
        
        # 测试训练技巧
        # self.test_early_stopping = False
        # self.test_grad_accumulation = False
        # self.test_lr_scheduler = False
        # self.test_mixed_precision = False
        # self.test_weight_decay = False
        # self.test_gradient_clipping = False   

    def __str__(self):
        # 自定义打印格式，方便查看配置信息
        return f'{self.__class__.__name__}({self.__dict__})'
    

    def __repr__(self):
        return self.__str__()


class PretrainConfig(Config):
    """预训练配置，继承基础配置并覆盖特定参数"""
    def __init__(self):
        super().__init__()
        # 预训练通常需要更大的batch和更多的epoch
        self.batch_size = 128

        # 预训练学习率通常较高
        self.lr = 5e-4
        
        # 预训练数据路径通常不同
        self.data_path = 'data/llm_data/processed/pretrain_hq.json'
        

class SFTConfig(Config):
    """全量微调配置"""
    def __init__(self):
        super().__init__()
        # 全量微调batch通常较小
        self.batch_size = 32
        self.num_epochs = 2

        # 全量微调学习率低于预训练
        self.lr = 5e-7

        # 微调数据路径
        self.data_path = 'data/llm_data/processed/sft_mini_512.json'
        # 预训练模型路径
        self.pretrain_path = 'all_models/llama3_pretrain.pt'
        


class LoraConfig(Config):
    """LoRA微调配置"""
    def __init__(self):
        super().__init__()
        # LoRA可以使用更大的batch（因为计算量小）
        self.batch_size = 64
        self.num_epochs = 10
        
        # LoRA学习率高于全量微调
        self.lr = 1e-4
        
        # 微调数据路径
        self.data_path = 'data/llm_data/processed/sft_mini_512.json'
        # 预训练模型路径
        self.pretrain_path = 'all_models/llama3_pretrain.pt'


class DPOConfig(Config):
    """DPO（偏好优化）配置"""
    def __init__(self):
        super().__init__()
        # DPO数据量通常较小，batch也较小
        self.batch_size = 16
        self.num_epochs = 2
        
        # DPO学习率非常低
        self.lr = 1e-8
        
        # 微调数据路径
        self.data_path = 'data/llm_data/processed/dpo.json'
        # 预训练模型路径
        self.pretrain_path = 'all_models/llama3_pretrain.pt'


class DistillConfig(Config):
    """蒸馏配置"""
    def __init__(self):
        super().__init__()
        self.batch_size = 64
        self.num_epochs = 6
        self.accumulation_steps = 1

        # 蒸馏学习率介于预训练和微调之间
        self.lr = 5e-6

        # 蒸馏数据路径（通常由教师模型生成）
        self.data_path = 'data/llm_data/processed/sft_mini_512.json'
        # 预训练模型路径
        self.teacher_path = 'all_logs/train_20250813_010827_llama3/saved_models/llama3_model.pt'
        # 学生模型路径（通常是一个较小的模型）
        self.student_path = 'all_models/llama3_pretrain.pt'

        # 蒸馏特有参数
        self.teacher_d_model = 768
        self.teacher_hidden_dim = 3072
        self.teacher_num_layers = 16

  
class ChatConfig(PretrainConfig):
    def __init__(self):
        super().__init__()
        self.temperature = 0.85
        self.top_p = 0.85
        self.max_generate_len = 1024
        self.chat_mode = 0
        self.model_type = 'pretrain' # pretrain: 预训练, 'sft': 全量微调， 'lora': LoRA微调
        self.istrain = False
        self.pretrain_path = 'pretrained_models/llama3'  # 预训练模型路径
        self.lora_path = 'saved_models/llama3_lora.pt'

class Llama1Config(PretrainConfig):
    def __init__(self):
        super().__init__()
        self.model = 'llama1'
        self.num_layers = 8

class Llama3Config(PretrainConfig):
    def __init__(self):
        super().__init__()
        self.model = 'llama3'
        self.num_layers = 8
