import torch
import logging
import random
import numpy as np

class EarlyStopping:
    def __init__(self,patience =10,delta=0.01):
        """
        Early stopping
        Args:
            patience: int, number of epochs to wait before stopping
            delta: float, the minimum improvements
        """
        self.patience = patience
        self.delta = delta
        self.counter =0 
        self.early_stop = False
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter =0 
        else:
            self.counter+=1
            if self.counter >= self.patience:
                self.early_stop = True
    
def plot_train_loss_mean(config, history, save_path='train_loss_mean.png'):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns  
    
    sns.set_theme(
        style="whitegrid",  
        font_scale=1.1,    
        palette="colorblind" 
    )
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        history, 
        label='Train Loss', 
        color='#2c7fb8',  
        linewidth=2.5,    
        linestyle='-',   
        markersize=5,    
        markeredgewidth=1.5,  
        markeredgecolor='white' 
    )
    
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold', labelpad=10)
    
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  
    ax.tick_params(axis='both', which='major', labelsize=12)  
    
    fig.suptitle(
        'Training Loss Over Epochs', 
        fontsize=16, 
        fontweight='bold', 
        y=0.98,  
        color='#333333',  
        ha='center'
    )
    ax.set_title(
        f'Epochs={config.num_epochs} | Batch Size={config.batch_size} | LR={config.lr:.6f}', 
        fontsize=12, 
        color='#666666',
        pad=15, 
        ha='center'
    )
    
    ax.legend(
        fontsize=12, 
        loc='upper right',  
        frameon=True,     
        framealpha=0.9,    
        edgecolor='#dddddd' 
    )
    
    ax.grid(color='#f0f0f0', linewidth=1.0, linestyle='--')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

    plt.tight_layout()
    

    plt.savefig(
        save_path,
        dpi=300, 
        bbox_inches='tight'  #
    )
    plt.close()

def plot_train_loss_all(config, history, save_path='train_loss_all.png'):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns  
    
    sns.set_theme(
        style="whitegrid",
        font_scale=1.1,
        palette="colorblind"
    )
    
    fig, ax = plt.subplots(figsize=(10,5))  
    

    ax.plot(
        history, 
        label='Train Loss', 
        color='#e41a1c',  
        linewidth=1.2,    
        alpha=0.8,        
        rasterized=True   
    )
    
    ax.set_xlabel('Batch Index', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold', labelpad=10)
    

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both'))  
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    fig.suptitle(
        'Training Loss Over Batches', 
        fontsize=16, 
        fontweight='bold', 
        y=0.98,
        color='#333333'
    )
    ax.set_title(
        f'Epochs={config.num_epochs} | Batch Size={config.batch_size} | LR={config.lr:.6f}', 
        fontsize=12, 
        color='#666666',
        pad=15
    )
    
    ax.legend(
        fontsize=12, 
        loc='upper right',
        frameon=True,
        framealpha=0.9,
        edgecolor='#dddddd'
    )
    

    ax.grid(color='#f0f0f0', linewidth=1.0, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 针对批次损失可能波动大的特点，添加y轴范围自动调整（可选）
    # ax.set_ylim(bottom=0, top=max(history)*1.1)  # 下限0，上限留10%余量
    
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

# 学习率调度函数：余弦退火
def get_lr(current_step,total_step,lr):
    return lr / 10 + 0.5 * lr * ( 1+ np.cos(np.pi * current_step / total_step))

# 配置日志
def setup_logger(log_dir):
    logger = logging.getLogger("global_logger")
    logger.setLevel(logging.INFO)
    # 确保只添加一次handler（避免重复输出）
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        # 文件输出
        log_filename = f"{log_dir}/training.log"
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

# 设置随机种子以确保实验可重复性
def set_seed(seed):
    torch.manual_seed(seed) #固定 PyTorch 框架自身的 CPU 随机数生成器的种子
    torch.cuda.manual_seed_all(seed) #固定所有 GPU（如果有多个 CUDA 设备）的随机数生成器的种子
    np.random.seed(seed) #固定 NumPy 库的随机数生成器种子
    random.seed(seed) #固定 Python 内置的 random 模块的随机数生成器种子
    torch.backends.cudnn.deterministic = True #启用 CuDNN 的确定性模式（CuDNN 是 NVIDIA 的深度学习加速库，PyTorch 的很多 GPU 操作（如卷积）会依赖它）
    torch.backends.cudnn.benchmark = False  #禁用 CuDNN 的自动调优功能（该功能会根据硬件和输入数据的特性选择最优的算法来加速计算，但会引入一定的随机性）
    
