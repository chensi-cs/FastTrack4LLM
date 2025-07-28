import torch
from transformers import PretrainedConfig

class TrainConfig(PretrainedConfig):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32 # batch size
        self.num_epochs = 2 # number of epochs
        self.lr = 5e-4 # learning rate
        self.num_heads = 8 # number of heads for multi-head attention
        self.hidden_dim = 1024 # hidden dimension for feedforward network
        self.num_layers = 8 # number of layers
        self.dropout = 0.0 # dropout rate
        self.d_model = 512 # embedding dimension for one token
        self.max_seq_len = 1024 # max length of input sequence
        self.max_position_len = 32768 # max length of position embedding
        self.vocab_size = 6400 # vocabulary size
        self.position_type = 'rope'
        self.activation = 'silu'
        self.kv_cache = False
        self.model = 'llama1'
        self.optimizer = 'adamw'
        self.iscausal = True
        self.patience = 5 # early stopping patience
        self.ddp = False
        self.use_wandb = False # use Weights & Biases for logging 
        self.use_tensorboard = True # use tensorboard for logging
        self.log_interval = 100 # log interval for training
        self.save_interval = 100
        self.evaluate_val = False
        self.evaluate_test = False
        
        # self.num_workers = 4
        self.data_path = 'data/model_data/train.json' # train data path
        self.val_path = 'data/model_data/val.json' # validation data path
        self.test_path = 'data/model_data/test.json' # test data path
        self.tokenizer_path = 'data/' # tokenizer path
        self.model_save_path = 'saved_models' # model save path
        self.log_dir = 'logs' # log directory
        self.checkpoint_path = 'checkpoints' # checkpoint path

        # 测试一些训练技巧
        self.test_early_stopping = False
        self.test_grad_accumulation = False
        self.test_lr_scheduler = False
        self.test_mixed_precision = False
        self.test_weight_decay = False
        self.test_gradient_clipping = False
        

    def __str__(self):
        # 自定义打印格式，方便查看配置信息
        return f"Config(device={self.device}, batch_size={self.batch_size}, num_epochs={self.num_epochs}, learning_rate={self.lr}, " \
               f"data_path={self.data_path}, tokenizer_path={self.tokenizer_path}, model_save_path={self.model_save_path}, log_dir={self.log_dir}, " \
               f"checkpoint_path={self.checkpoint_path}, d_model={self.d_model}, max_seq_len={self.max_seq_len}, vocab_size={self.vocab_size}, " \
               f"validation_path={self.val_path}, test_path={self.test_path}),early stopping patience = {self.patience}" 

    def __repr__(self):
        return self.__str__()

class ChatConfig():
    def __init__(self):
        self.tokenizer_path = 'data/'
        self.model_path = 'saved_models/'
        self.model_name = 'llama1'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = 0.7
        self.top_p = 0.9
        self.max_generate_len = 1024
        self.chat_mode = 0


class Llama1Config(TrainConfig):
    def __init__(self):
        super().__init__()
        self.model = 'llama1'
        self.num_layers = 8

class Llama2Config(TrainConfig):
    def __init__(self):
        super().__init__()
        self.model = 'llama2'
        self.num_layers = 12

class Llama3Config(TrainConfig):
    def __init__(self):
        super().__init__()
        self.model = 'llama3'
        self.num_layers = 16
