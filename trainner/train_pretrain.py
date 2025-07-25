# -*- coding: utf-8 -*-
import os
import json
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader 
import sys
from utils.config import Config
from utils.data import PretrainDataset  
from models.llama_model import Llama1Model
from utils.utils import EarlyStopping   
import logging


from pathlib import Path

# 添加项目根目录到sys.path
sys.path.append(str(Path(__file__).parent.parent))


# 基础配置：输出到终端，级别为INFO，格式包含时间、级别、消息
logging.basicConfig(
    level=logging.INFO,  # 日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 格式
    datefmt="%Y-%m-%d %H:%M:%S"  # 时间格式
)

# 获取日志器（通常用__name__表示当前模块）
logger = logging.getLogger(__name__)


def  train_one_epoch(model,train_loader,optimizer,device,epoch,config):
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
        logger.info(f"Output shape: {output.shape}")
        logger.info(f"Output: {output}")
        logger.info(f"Y shape: {y.shape}")
        logger.info(f"Y: {y}")

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
            logger.info(f"Output shape: {output.shape}")
            logger.info(f"Y shape: {y.shape}")
            loss = criterion(output.view(-1,output.size(-1)),y.view(-1))
            val_loss += loss.item()
    val_loss = val_loss / len(val_loader)
    return val_loss



def train(config):
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
    train_dataset = PretrainDataset( config.data_path,config.tokenizer_path,config.max_len)
    train_loader = DataLoader(train_dataset,batch_size=config.batch_size, shuffle=True,num_workers=0)

    val_dataset = PretrainDataset( config.val_path,config.tokenizer_path,config.max_len)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,num_workers=0)

    test_dataset = PretrainDataset( config.test_path,config.tokenizer_path,config.max_len)
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }
            torch.save(checkpoint, config.checkpoint_path)
            logger.info(f"New best model saved at {config.checkpoint_path} with val loss: {val_loss}")
        
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch}")
            break
    logger.info("Training completed.")
    



if __name__ == "__main__":
    config = Config()
    logging.info("Start training")
    config.data_path = "data/model_data/train.json"
    config.val_path = "data/model_data/val.json"
    config.test_path = "data/model_data/test.json"
    config.tokenizer_path = "data/"
    config.vocab_size = 6400
    config.model = 'llama1'
    config.d_model = 4
    config.num_heads = 1
    config.num_layers = 2
    config.hidden_dim = 10
    config.batch_size = 1
    config.max_len = 10
    config.num_epochs = 1
    # config.kv_cache = True  
    train(config)