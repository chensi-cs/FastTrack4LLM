# -*- coding: utf-8 -*-
import os
import json
import torch
import torch.nn as nn
from datetime import datetime
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader 
import sys
from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter

# 获取当前脚本的绝对路径
script_path = Path(__file__).resolve()  # 例如：/Users/cs/cs-work/llm_learning/trainner/train_pretrain.py

# 获取项目根目录（祖父目录）
project_root = str(script_path.parent.parent)  # 例如：/Users/cs/cs-work/llm_learning

# 添加项目根目录到sys.path
sys.path.append(project_root)


# 打印验证（绝对路径会清晰显示）
print("项目根目录:", project_root)
print("搜索路径:", sys.path)

from utils.config import Config
from utils.data import PretrainDataset  
from models.llama_model import Llama1Model
from utils.utils import EarlyStopping


# 配置日志
def setup_logger(log_dir='logs'):
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
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{log_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 初始化全局logger
logger = setup_logger()

# 初始化TensorBoard写入器
def setup_tensorboard():
    return SummaryWriter(f'logs/train_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

# 初始化TensorBoard写入器并保存实例
writer = setup_tensorboard()

def train_one_epoch(model,train_loader,optimizer,device,epoch,config):
    model.train()
    # 如果不指定reduction参数，交叉熵损失会对所有样本的损失值求平均（reduction='mean'）或求和（reduction='sum'）。
    # 当设置为reduction='none'时，损失函数会为每个样本单独计算损失值，不进行任何聚合操作（既不求和也不平均）。返回的是一个与输入样本数量相同的损失张量。
    criterion =  nn.CrossEntropyLoss(ignore_index=0)
    train_loss = 0.0
    
    avg_loss = 0.0
    
    accumulation_steps = 10
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        x, y, loss_mask = batch
        x = x.to(device)
        y = y.to(device)
        loss_mask = loss_mask.to(device)

        output = model(x)

        running_loss = criterion(output.view(-1,output.size(-1)),y.view(-1))

        # 梯度累积的操作之一
        loss = running_loss / accumulation_steps 

        loss.backward()

        if (batch_idx+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += running_loss.item()
        avg_loss = train_loss / (batch_idx+1)
        logger.info(f"Epoch {epoch}, batch {batch_idx}, loss: {avg_loss}")
    return avg_loss
        
def evaluate(model,val_loader,device,config):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    val_loss = 0.0
    for batch in val_loader:
        x, y, loss_mask = batch  
        x = x.to(device)
        y = y.to(device)
        loss_mask = loss_mask.to(device)
        with torch.no_grad():
            output = model(x)
            loss = criterion(output.view(-1,output.size(-1)),y.view(-1))
            val_loss += loss.item()
    val_loss = val_loss / len(val_loader)
    return val_loss



def train(config):
    os.makedirs(config.model_save_path,exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_path, exist_ok=True)

    

    # 训练逻辑
    device = config.device
    if config.model == 'llama1':
        model = Llama1Model(config)
    else:
        raise ValueError(f"Model {config.model} not supported")
    
    
    model.to(device)
    logger.info(f"Model {config.model} loaded")
    logger.info(f"config: {config}")
    logger.info("Loading datasets...")
    train_dataset = PretrainDataset( config.data_path,config.tokenizer_path,config.max_seq_len)
    train_loader = DataLoader(train_dataset,batch_size=config.batch_size, shuffle=True,num_workers=0)

    val_dataset = PretrainDataset( config.val_path,config.tokenizer_path,config.max_seq_len)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,num_workers=0)

    test_dataset = PretrainDataset( config.test_path,config.tokenizer_path,config.max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,num_workers=0)

    logger.info(f"Datasets loaded successfully...")
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataset)}")
    logger.info(f"Number of test samples: {len(test_dataset)}")

    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(),lr=config.lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(config.patience)

    for epoch in range(1,1+config.num_epochs):
        train_loss = train_one_epoch(model,train_loader,optimizer,device,epoch,config)
        history['train_loss'].append(train_loss)
        logger.info(f"Epoch {epoch}, average train loss: {train_loss}")
        val_loss = evaluate(model,val_loader,device,config)
        history['val_loss'].append(val_loss)
        logger.info(f"Epoch {epoch}, average val loss: {val_loss}")
        logger.info("-"*100)

        # 记录训练损失
        writer.add_scalar("Loss/Train", train_loss, epoch)  # 标签路径建议分类，方便TB中分组查看
        # 记录验证损失
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }
            checkpoint_saved_path = os.path.join(config.checkpoint_path, f"checkpoint_epoch_{epoch}.pt")
            torch.save(checkpoint, checkpoint_saved_path)
            logger.info(f"New best model saved at {config.checkpoint_path} with val loss: {val_loss}")
        
        # 记录参数分布
        for name, param in model.named_parameters():
            # 记录参数值分布
            writer.add_histogram(f"Parameters/{name}", param, epoch)  # 参数值的直方图
            # 记录参数梯度分布（仅训练阶段有梯度）
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

        # 记录当前学习率（取第一个参数组的学习率即可，默认所有组一致）
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("LearningRate", current_lr, epoch)

        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch}")
            break

        
    logger.info("Training completed.")
    writer.close()  # 必须添加，否则可能导致日志未完全写入



if __name__ == "__main__":
    config = Config()
    logging.info("Start training")
    

    # config.data_path = "data/model_data/train.json"
    # config.val_path = "data/model_data/val.json"
    # config.test_path = "data/model_data/test.json"
    # config.tokenizer_path = "data/"
    # config.vocab_size = 6400
    # config.model = 'llama1'
    # config.d_model = 512
    # config.num_heads = 1
    # config.num_layers = 2
    # config.hidden_dim = 10
    # config.batch_size = 1
    # config.max_seq_len = 10
    config.num_epochs = 1
    # config.kv_cache = True  
    train(config)
