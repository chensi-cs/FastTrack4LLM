import torch

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32 # batch size
        self.num_epochs = 10 # number of epochs
        self.learning_rate = 0.001 # learning rate
        self.num_heads = 8 # number of heads for multi-head attention
        self.hidden_dim = 2024 # hidden dimension for feedforward network
        self.num_layers = 6 # number of layers
        self.dropout = 0.1 # dropout rate
        self.d_model = 128 # embedding dimension for one token
        self.max_len = 128 # max length of input sequence
        self.vocab_size = 10000 # vocabulary size
        self.position_type = 'sin_cos'
        self.activation = 'relu'
        self.kv_cache = True

        # self.num_workers = 4
        self.data_path = 'data.csv' # train data path
        self.validation_path = 'validation.csv' # validation data path
        self.test_path = 'test.csv' # test data path
        self.tokenizer_path = 'tokenizer.json' # tokenizer path
        self.model_save_path = 'saved_models' # model save path
        self.log_dir = 'logs' # log directory
        self.checkpoint_path = 'checkpoints' # checkpoint path
        

    def __str__(self):
        # 自定义打印格式，方便查看配置信息
        return f"Config(device={self.device}, batch_size={self.batch_size}, num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, " \
               f"data_path={self.data_path}, tokenizer_path={self.tokenizer_path}, model_save_path={self.model_save_path}, log_dir={self.log_dir}, " \
               f"checkpoint_path={self.checkpoint_path}, embedding_dim={self.embedding_dim}, max_length={self.max_length}, vocab_size={self.vocab_size}, " \
               f"validation_path={self.validation_path}, test_path={self.test_path})"

    def __repr__(self):
        return self.__str__()

config = Config()
print(config)