# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

__package__ = "trainer"
script_path = Path(__file__).resolve()  # 获取当前脚本的绝对路径
project_root = str(script_path.parent.parent)  # 获取项目根目录（祖父目录）
sys.path.append(project_root) # 添加项目根目录到sys.path

import torch
import argparse
import logging
import wandb
import torch.nn as nn
from datetime import datetime
import torch.optim as optim
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from utils.config import PretrainConfig
from utils.data import PretrainDataset  
from models.llama_model import  Llama1ForCausalLM,Llama3ForCausalLM
from utils.utils import EarlyStopping, plot_train_loss_mean, plot_train_loss_all,get_lr,set_seed, setup_logger,evaluate


def train_one_epoch(model,train_loader,optimizer,device,epoch,config):
    model.train()
    
    criterion =  nn.CrossEntropyLoss(ignore_index=0)
    train_loss = 0.0
    avg_loss = 0.0

    accumulation_steps = config.accumulation_steps  # 梯度累积步数
    optimizer.zero_grad()
    iter_per_epoch = len(train_loader)

    start_time = datetime.now()
    epoch_start_time = start_time

    for batch_idx, batch in enumerate(train_loader):
        x, y, attention_mask = batch
        x = x.to(device)
        y = y.to(device)
        attention_mask = attention_mask.to(device)
        if config.attn_mask == False:
            attention_mask = None
        
        lr = get_lr((epoch-1)* iter_per_epoch + batch_idx, config.num_epochs * iter_per_epoch , config.lr )

        if writer is not None:
            writer.add_scalar("LearningRate", lr, batch_idx)

        # 遍历优化器中的所有参数组，然后把每个参数组的学习率（'lr'）都设定为新计算得出的值（lr）
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if config.use_amp:
            with autocast():
                output = model(input_ids=x, attention_mask=attention_mask)    
        else:
            output = model(input_ids=x, attention_mask=attention_mask)   

        logits = output.logits  

        running_loss = criterion(logits.view(-1,logits.size(-1)),y.view(-1))

        if config.add_aux_loss and output.loss is not None:
            running_loss += output.loss
        train_loss_all.append(running_loss.item())


        loss = running_loss / accumulation_steps 
        if config.use_amp:
            scaler.scale(loss).backward() #缩放损失值，计算梯度
        else :
            loss.backward() 
        
        # 梯度累积，参数更新
        if (batch_idx+1) % accumulation_steps == 0:
            if config.use_amp:
                scaler.unscale_(optimizer) # 反向梯度缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = config.grad_clip) #裁剪梯度
                scaler.step(optimizer) #更新模型参数
                scaler.update() # 动态更新缩放因子
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = config.grad_clip)
                optimizer.step()

            if writer is not None:
                # 记录参数分布
                for name, param in model.named_parameters():
                    # 记录参数值分布
                    writer.add_histogram(f"Parameters/{name}", param, batch_idx)  # 参数值的直方图
                    # 记录参数梯度分布（仅训练阶段有梯度）
                    if param.grad is not None and torch.isfinite(param.grad).any():
                        writer.add_histogram(f"Gradients/{name}", param.grad, batch_idx)
                        
            optimizer.zero_grad(set_to_none=True) # 梯度清零

        # 日志记录
        if (batch_idx+1) %  config.log_interval == 0 or (batch_idx+1) == len(train_loader):
            if wandb:
                wandb.log({"train_loss": running_loss.item(), "epoch": epoch, "batch_idx": batch_idx})
            if writer:
                writer.add_scalar('Batch Loss', running_loss.item(), batch_idx)
            end_time = datetime.now()
            logger.info(f"Epoch [{epoch}/{config.num_epochs}] ({batch_idx+1}/{iter_per_epoch}) Train Loss: {running_loss.item():.5f} LR: {lr:.12f} Duration: {(end_time - start_time).total_seconds() / 60 :.2f} minutes")
            start_time = datetime.now()

        # 模型检查点保存
        if (batch_idx+1) %  config.save_interval == 0 or (batch_idx+1) == len(train_loader):
            model.eval()  # 切换到推理模式
            checkpoint = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': running_loss,
                }
            checkpoint_saved_path = os.path.join(config.checkpoint_path, f"checkpoint_epoch_{epoch}.pt")
            torch.save(checkpoint, checkpoint_saved_path)
            model.train()  # 切换回训练模式

        train_loss += running_loss.item()
        avg_loss = train_loss / (batch_idx+1)
    
    epoch_end_time = datetime.now()
    logger.info(f"Epoch [{epoch}/{config.num_epochs}] Total Train Loss: {running_loss.item():.5f} LR: {lr:.12f} Duration: {(epoch_end_time - epoch_start_time).total_seconds() / 60 :.2f} minutes")

    return avg_loss
        
def train(config):
    # 1. 预训练设置
    os.makedirs(config.model_save_path,exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_path, exist_ok=True)
    global writer
    global wandb
    # 初始化TensorBoard写入器
    if config.use_tensorboard:
        writer = SummaryWriter(config.log_dir)
    else:
        writer = None

    if config.use_wandb:
        wandb.init(project="llm_pretrain", name=f"train_{config.model}_{config.num_epochs}_{config.batch_size}_{config.lr}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else :
        wandb = None
    
    device = config.device
    logger.info(f"模型配置: {config}")

    # 2. 加载模型
    if config.model == 'llama1':
        model =  Llama1ForCausalLM(config)
    elif config.model == 'llama3':
        model =  Llama3ForCausalLM(config)
    else:
        raise ValueError(f"Model {config.model} not supported")
    model.to(device)
    logger.info(f"Model infomation: {model}")
    # 计算模型总参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"模型参数量: {total_params:.3f} M") 
    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f'LLM可训练参数量: {trainable_params:.3f} M')


    # 3. 加载数据集
    logger.info("加载数据集...")
    train_dataset = PretrainDataset( config.data_path,config.tokenizer_path,config.max_seq_len)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory = True, # 内存锁页
        drop_last=True,  # 丢弃最后一个不完整的batch
        prefetch_factor=2,  # 用来控制每个工作进程预先加载的样本数量
        persistent_workers=True  # 保持工作进程持续存在
    )
    logger.info(f"预训练样本数: {len(train_dataset)}")

    if config.evaluate_val:
        val_dataset = PretrainDataset( config.val_path,config.tokenizer_path,config.max_seq_len)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,num_workers=16)
        logger.info(f"验证集样本数: {len(val_dataset)}")

    # 4. 优化器和训练设置
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(),lr=config.lr)
    
    global scaler
    if config.use_amp:
        scaler = GradScaler()
    else :
        scaler = None
    global train_loss_all
    train_loss_all = []
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(config.patience)
    
    # 5. 预训练开始
    try:
        for epoch in range(1,1+config.num_epochs):
            # 一个epoch训练
            train_loss = train_one_epoch(model,train_loader,optimizer,device,epoch,config)
            history['train_loss'].append(train_loss)
            logger.info(f"Epoch {epoch}, average train loss: {train_loss}")
            
            if config.evaluate_val:
                val_loss = evaluate(model,val_loader,device,config)
                history['val_loss'].append(val_loss)
                logger.info(f"Epoch {epoch}, average val loss: {val_loss}")

                # 记录验证损失
                if writer is not None:
                    writer.add_scalar("Loss/Validation", val_loss, epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.eval()  # 切换到推理模式
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': train_loss,
                    }
                    checkpoint_saved_path = os.path.join(config.checkpoint_path, f"checkpoint_epoch_{epoch}.pt")
                    torch.save(checkpoint, checkpoint_saved_path)
                    logger.info(f"新的最佳模型保存至： {config.checkpoint_path} ，验证集损失: {val_loss}")
                if early_stopping(val_loss):
                    logger.info(f"Early stopping!!!当前轮次： {epoch}")
                    break

            if writer is not None:
                writer.add_scalar("Loss/Train", train_loss, epoch)  # 标签路径建议分类，方便TB中分组查看

            # 保存模型
            model.eval()  # 切换到推理模式
            logger.info(f" {epoch} 轮次模型保存中...")
            model_save_path = os.path.join(config.model_save_path, f"{config.model}_model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), model_save_path)
            logger.info("-"*100)
        
        # 训练结束，绘制损失曲线
        plot_train_loss_mean(config,history['train_loss'],save_path=f"{config.log_dir}/train_loss_mean.png")
        plot_train_loss_all(config,train_loss_all, save_path=f"{config.log_dir}/train_loss_all.png")
        logger.info("训练完成!!!")
    except KeyboardInterrupt:
        logger.info("训练被用户中断!!!")            
    finally:
        model.eval()  
        logger.info(f"最终模型保存中...")
        model_save_path = os.path.join(config.model_save_path, f"{config.model}_model.pt")
        torch.save(model.state_dict(), model_save_path)
        if writer is not None:
            writer.close()

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="预训练")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--model", type=str, default='llama1')
    parser.add_argument("--attn_mask", action='store_true', default=False)
    parser.add_argument("--use_moe", action='store_true', default=False)
    parser.add_argument("--add_aux_loss", action='store_true', default=False)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    config = PretrainConfig()
    # config.data_path = 'data/model_data/demo/train.json'
    for key, value in vars(args).items():
        if hasattr(config, key):  
            setattr(config, key, value)

    # 初始化全局logger
    logger = setup_logger(config.log_dir)

    # 创建包含当前时间的日志目录
    now_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.log_dir = os.path.join("logs", f"pretrain_{now_timestamp}_{config.model}")
    os.makedirs(config.log_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
    
    # 开始训练
    start_time = datetime.now()
    logging.info("开启预训练...")
    logger.info(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    train(config)
    end_time = datetime.now()
    logger.info(f"训练结束时间:  {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"训练耗时: {end_time - start_time}")
