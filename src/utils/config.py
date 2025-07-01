import torch

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.n_head = 8
        # self.num_workers = 4
        self.data_path = 'data.csv'
        self.validation_path = 'validation.csv'
        self.test_path = 'test.csv'
        self.tokenizer_path = 'tokenizer.json'
        self.model_save_path = 'saved_models'
        self.log_dir = 'logs'
        self.checkpoint_path = 'checkpoints'
        self.embedding_dim = 128
        self.max_length = 128
        self.vocab_size = 10000

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