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
from utils.config import LoraConfig
from utils.data import SFTDataset 
from models.add_lora import apply_lora,save_lora,count_lora_parameters
from models.llama_model import  Llama1ForCausalLM,Llama3ForCausalLM
from utils.utils import EarlyStopping, plot_train_loss_mean, plot_train_loss_all,get_lr,set_seed, setup_logger,evaluate


def train_one_epoch(model,train_loader,optimizer,device,epoch,config):
    model.train()
    
    criterion = nn.CrossEntropyLoss(reduction='none') # 为每个样本单独计算损失值,不进行任何聚合操作
    train_loss = 0.0
    avg_loss = 0.0

    accumulation_steps = config.accumulation_steps  # 梯度累积步数
    optimizer.zero_grad()
    iter_per_epoch = len(train_loader)

    start_time = datetime.now()
    epoch_start_time = start_time

    for batch_idx, batch in enumerate(train_loader):
        x, y, attention_mask, loss_mask = batch
        x = x.to(device)
        y = y.to(device)
        attention_mask = attention_mask.to(device)
        loss_mask = loss_mask.to(device)
        if config.attn_mask== False:
            attention_mask = None

        lr = get_lr((epoch-1)* iter_per_epoch + batch_idx, config.num_epochs * iter_per_epoch , config.lr )
        if writer is not None:
            writer.add_scalar("LearningRate", lr, batch_idx)


        # 遍历优化器中的所有参数组，然后把每个参数组的学习率（'lr'）都设定为新计算得出的值（lr）
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast():
            output = model(input_ids=x, attention_mask=attention_mask)    

        logits = output.logits # [batch_size, seq_len, vocab_size]
        
        # logits.view(-1,logits.size(-1)) : [batch_size, seq_len, vocab_size] > [batch_size * seq_len, vocab_size]
        # y.view(-1) :[batch_size, seq_len] > [batch_size * seq_len]
        # running_loss  [batch_size * seq_len]
        running_loss = criterion(logits.view(-1,logits.size(-1)),y.view(-1)) 
        running_loss = running_loss.view(y.size())  # 将损失值的形状调整为 [batch_size, seq_len]

        running_loss = (running_loss * loss_mask )
        running_loss = running_loss.sum() / loss_mask.sum()  # 计算有效损失的平均值

        if config.add_aux_loss and output.loss is not None:
            running_loss += output.loss

        train_loss_all.append(running_loss.item())
        loss = running_loss / accumulation_steps 

        scaler.scale(loss).backward() #缩放损失值，计算梯度

        if (batch_idx+1) % accumulation_steps == 0:
            scaler.unscale_(optimizer) # 反向梯度缩放
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = config.grad_clip) #裁剪梯度
            scaler.step(optimizer) #更新模型参数
            scaler.update() # 动态更新缩放因子

            if writer is not None:
                # 记录参数分布
                for name, param in model.named_parameters():
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
            logger.info(f"Epoch [{epoch}/{config.num_epochs}] ({batch_idx+1}/{iter_per_epoch}) Train Loss: {running_loss.item():.5f} LR: {lr:.12f} Duration: {(end_time - start_time).total_seconds() / 60 :.5f} minutes")
            start_time = datetime.now()
        # 模型检查点保存
        if (batch_idx+1) %  config.save_interval == 0 or (batch_idx+1) == len(train_loader):
            model.eval()  # 切换到推理模式
            lora_save_path = os.path.join(config.checkpoint_path, f"lora_checkpoint_epoch_{epoch}.pt")
            save_lora(model, lora_save_path)
            model.train()  # 切换回训练模式

        train_loss += running_loss.item()
        avg_loss = train_loss / (batch_idx+1)
        
    epoch_end_time = datetime.now()
    logger.info(f"Epoch [{epoch}/{config.num_epochs}] Total Train Loss: {running_loss.item():.5f} LR: {lr:.12f} Duration: {(epoch_end_time - epoch_start_time).total_seconds() / 60 :.5f} minutes")
    
    return avg_loss

def train(config):
    # 1. 训练设置
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
        wandb.init(project="lora", name=f"lora_{config.model}_{config.num_epochs}_{config.batch_size}_{config.lr}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else :
        wandb = None
    device = config.device
    logger.info(f"参数配置: {config}")

    # 2. 加载模型
    if config.model == 'llama1':
        model =  Llama1ForCausalLM(config)
    elif config.model == 'llama3':
        model =  Llama3ForCausalLM(config)
    else:
        raise ValueError(f"Model {config.model} not supported")
    
    # 加载预训练模型
    model.load_state_dict(torch.load(config.pretrain_path, map_location=device),strict=True)
    model.to(device)
    logger.info(f"预训练模型加载成功...")
    logger.info(f"预训练模型信息: {model}")
    
    # 应用LoRA
    logger.info(f"在预训练模型上应用LoRA...")
    apply_lora(model, rank=config.lora_rank, device=config.device)  

    logger.info(f"应用LoRA后模型信息: {model}")

    # 计算模型总参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"模型总参数量: {total_params:.3f} M")  # 保留3位小数
    
    lora_params_count , name_lora_params = count_lora_parameters(model)

    logger.info(f"LoRA参数量: {lora_params_count:.3f} M")  # 保留3位小数
    logger.info(f"LoRA参数占比: {lora_params_count / total_params * 100:.3f} %")  # 保留2位小数

    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f'LLM可训练参数量: {trainable_params:.3f} M')

    logger.info(f"模型参数介绍:")
    for name, param in model.named_parameters():
        logger.info(f"{name}: {param.size()}")

    logger.info(f"LoRA参数介绍:")
    for name, param in name_lora_params:
        logger.info(f"{name}: {param.size()}")
    lora_params = [name_param[1] for name_param in name_lora_params]

    # 3. 加载数据集
    logger.info("加载数据集...")
    train_dataset = SFTDataset(config.data_path,config.tokenizer_path,config.max_seq_len)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory = True,# 内存锁页
        drop_last=True,  # 丢弃最后一个不完整的batch
        prefetch_factor=2,  # 用来控制每个工作进程预先加载的样本数量
        persistent_workers=True  # 保持工作进程持续存在
    )
    logger.info(f"LoRA微调样本数: {len(train_dataset)}")

    
    if config.evaluate_val:
        val_dataset = SFTDataset( config.val_path,config.tokenizer_path,config.max_seq_len)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,num_workers=16)
        logger.info(f"验证集样本数: {len(val_dataset)}")

    # 4. 优化器和训练设置
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(lora_params,lr=config.lr)
    global scaler
    scaler = GradScaler()
    
    global train_loss_all
    train_loss_all = []    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(config.patience)
    
    # 5. 微调开始
    try:
        for epoch in range(1,1+config.num_epochs):
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
                if early_stopping(val_loss):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if writer is not None:
                writer.add_scalar("Loss/Train", train_loss, epoch)  # 标签路径建议分类，方便TB中分组查看

            # 保存模型
            model.eval()  # 切换到推理模式
            logger.info(f" {epoch} 轮次LoRA参数保存中...")
            lora_save_path = os.path.join(config.model_save_path, f"{config.model}_lora_epoch_{epoch}.pt")
            save_lora(model, lora_save_path)
            logger.info("-"*100)

        # 训练结束，绘制损失曲线
        plot_train_loss_mean(config,history['train_loss'],save_path=f"{config.log_dir}/train_loss_mean.png")
        plot_train_loss_all(config,train_loss_all, save_path=f"{config.log_dir}/train_loss_all.png")
        logger.info("LoRA微调训练完成!!!")

    except KeyboardInterrupt:
        logger.info("训练被用户中断!!!")              
    finally:
        model.eval()  # 切换到推理模式
        logger.info(f"最终LoRA参数保存中...")
        lora_save_path = os.path.join(config.model_save_path, f"{config.model}_lora.pt")
        save_lora(model, lora_save_path)
        if writer is not None:
            writer.close()

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="LoRA高效微调")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--loss_mask", action='store_true', default=False)
    parser.add_argument("--attn_mask", action='store_true', default=False)
    parser.add_argument("--model", type=str, default='llama1')
    parser.add_argument("--pretrain_path", type=str, default='all_models/llama1_pretrain.pt')
    parser.add_argument("--use_moe", action='store_true', default=False)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--add_aux_loss", action='store_true', default=False)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    defaults = parser.parse_args([])

    config = LoraConfig()
    # config.data_path = "data/llm_data/processed/demo_data/sft_mini_512.json"
    for key, value in vars(args).items():
        if value != getattr(defaults, key) and hasattr(config, key):
            setattr(config, key, value)
    config.data_path = "data/llm_data/processed/lora_medical.json"
    config.pretrain_path = "all_logs/sft_20251105_175730_llama1/saved_models/llama1_sft.pt"
    config.loss_mask = True
    # 创建包含当前时间的日志目录
    now_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.log_dir = os.path.join("logs", f"lora_{now_timestamp}_{config.model}")
    os.makedirs(config.log_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错

    # 初始化全局logger
    logger = setup_logger(config.log_dir)

    # 开始训练
    start_time = datetime.now()
    logging.info("开启LoRA高效微调...")
    logger.info(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    train(config)
    end_time = datetime.now()
    logger.info(f"训练结束时间:  {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"训练耗时: {end_time - start_time}")

