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
import wandb
import random
import numpy as np
import argparse
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler


# 获取当前脚本的绝对路径
script_path = Path(__file__).resolve()  # 例如：/Users/cs/cs-work/llm_learning/trainner/train_pretrain.py

# 获取项目根目录（祖父目录）
project_root = str(script_path.parent.parent)  # 例如：/Users/cs/cs-work/llm_learning

# 添加项目根目录到sys.path
sys.path.append(project_root)


# 打印验证（绝对路径会清晰显示）
print("项目根目录:", project_root)
print("搜索路径:", sys.path)

from utils.config import DPOConfig
from utils.data import DPODataset
from models.llama_model import Llama3ForCausalLM, Llama1ForCausalLM
from utils.utils import EarlyStopping

# 配置日志
def setup_logger(log_dir):
    global logger
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

# 初始化TensorBoard写入器
def setup_tensorboard(log_dir):
    return SummaryWriter(log_dir)

def get_lr(current_step,total_step,lr):
    return lr / 10 + 0.5 * lr * ( 1+ np.cos(np.pi * current_step / total_step))

def logits_2_probs(logits,y):
    # logits = [batch_size, seq_len, vocab_size]
    # probs = [batch_size, seq_len, vocab_size]
    probs = F.log_softmax(logits,dim=2)
    # y = [batch_size, seq_len]
    # y.unsqueeze(2) = [batch_size, seq_len, 1]
    # torch.gather 根据标签索引，从对数概率中提取每个位置对应真实标签的对数概率
    # probs_target = [batch_size, seq_len]
    probs_target = torch.gather(probs,dim=2,index=y.unsqueeze(2)).squeeze(-1)

    return probs_target

def dpo_loss(probs,ref_probs,loss_mask,beta):
    # probs = [batch_size, seq_len]
    batch_size, _ = probs.shape
    # loss_mask : [batch_size, seq_len]
    if loss_mask is not None:
        seq_len = loss_mask.sum(dim=1,keepdim=True)
        seq_len = seq_len.squeeze(-1)  # seq_len is [batch_size, 1], so we need to squeeze it to [batch_size]

    probs = (probs * loss_mask).sum(dim=1) /seq_len

    ref_probs = (ref_probs * loss_mask).sum(dim=1) /seq_len

    y_chosen = probs[:batch_size//2 ]

    y_rejected = probs[batch_size//2:]

    ref_y_chosen = ref_probs[:batch_size//2]

    ref_y_rejected = ref_probs[batch_size//2:]
    reward = y_chosen - y_rejected
    reward_ref = ref_y_chosen - ref_y_rejected
    # loss = - torch.log(torch.sigmoid(beta*(reward - reward_ref)))
    loss = - F.logsigmoid(beta * (reward - reward_ref))
    loss = loss.mean()  # 平均损失
    return loss



def train_one_epoch(train_loader,optimizer,device,epoch,config):
    model.train()
    ref_model.eval()
    # 如果不指定reduction参数，交叉熵损失会对所有样本的损失值求平均（reduction='mean'）或求和（reduction='sum'）， 默认是 reduction: str = "mean"
    # 当设置为reduction='none'时，损失函数会为每个样本单独计算损失值，不进行任何聚合操作（既不求和也不平均）。返回的是一个与输入样本数量相同的损失张量。
    # 因为设置了loss_mask,所以将reduction设置为'none'，在计算损失时，使用loss_mask来计算需要忽略的损失值
    if config.loss_mask:
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion =  nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充的token，填充的token通常是0
    train_loss = 0.0
    avg_loss = 0.0

    accumulation_steps = config.accumulation_steps  # 梯度累积步数

    optimizer.zero_grad()
    global epoch_loss_list
    epoch_loss_list = []
    iter_per_epoch = len(train_loader)
    start_time = datetime.now()
    for batch_idx, batch in enumerate(train_loader):

        # x_chosen: [batch_size, seq_len]
        x_chosen = batch['x_chosen'].to(device)
        y_chosen = batch['y_chosen'].to(device)
        x_rejected = batch['x_rejected'].to(device)
        y_rejected = batch['y_rejected'].to(device)
        chosen_loss_mask = batch['chosen_loss_mask'].to(device)
        rejected_loss_mask = batch['rejected_loss_mask'].to(device)
        chosen_attn_mask = batch['chosen_attn_mask'].to(device)
        rejected_attn_mask = batch['rejected_attn_mask'].to(device)
        
        # x : [2*batch_size, seq_len]
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected],dim=0)
        loss_mask = torch.cat([chosen_loss_mask, rejected_loss_mask],dim=0)
        attention_mask = torch.cat([chosen_attn_mask, rejected_attn_mask],dim=0)

        if config.attn_mask== False:
            attention_mask = None

        loss_mask = loss_mask.to(device)

        lr = get_lr((epoch-1)* iter_per_epoch + batch_idx, config.num_epochs * iter_per_epoch , config.lr )
        if writer is not None:
            writer.add_scalar("LearningRate", lr, batch_idx)

        # 遍历优化器中的所有参数组，然后把每个参数组的学习率（'lr'）都设定为新计算得出的值（lr）
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if config.use_amp:
            with autocast():
                output = model(input_ids=x, attention_mask=attention_mask)    
                with torch.no_grad():
                    ref_output = ref_model(input_ids=x, attention_mask=attention_mask)
        else:
            output = model(input_ids=x, attention_mask=attention_mask)  
            with torch.no_grad():
                ref_output = ref_model(input_ids=x, attention_mask=attention_mask)

        # output_probs = [batch_size, seq_len]

        output_probs = logits_2_probs(output.logits,y)

        ref_output_probs = logits_2_probs(ref_output.logits,y)

        if config.loss_mask:
            running_loss = dpo_loss(output_probs,ref_output_probs,loss_mask,config.dpo_beta)
        else:
            running_loss = dpo_loss(output_probs,ref_output_probs,None,config.dpo_beta)

        epoch_loss_list.append(running_loss.item())

        # 梯度累积的操作之一
        loss = running_loss / accumulation_steps 

        if config.use_amp:
            #  使用scaler.scale()方法来缩放损失值
            #  使用.backward()方法来计算梯度
            scaler.scale(loss).backward()
        else :
            loss.backward() 


        if (batch_idx+1) % accumulation_steps == 0:
            if config.use_amp:
                # 将已经缩放的梯度 反向缩放
                scaler.unscale_(optimizer)
                # 使用clip_grad_norm_()方法来裁剪梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = config.grad_clip)
                # 使用scaler.step()方法来更新模型参数
                scaler.step(optimizer)
                # 动态更新缩放因子
                scaler.update()
            else:
                # 使用clip_grad_norm_()方法来裁剪梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = config.grad_clip)
                optimizer.step()

            if writer is not None:
                # 记录参数分布
                for name, param in model.named_parameters():
                    writer.add_histogram(f"Parameters/{name}", param, batch_idx)  # 参数值的直方图
                    # 记录参数梯度分布（仅训练阶段有梯度）
                    if param.grad is not None and torch.isfinite(param.grad).any():
                        writer.add_histogram(f"Gradients/{name}", param.grad, batch_idx)
                        
            # 梯度清零,set_to_none=True会将梯度设置为 None，而不是将其设置为 0,节省内存
            optimizer.zero_grad(set_to_none=True)

        if (batch_idx+1) %  config.log_interval == 0 or (batch_idx+1) == len(train_loader):
            if wandb:
                wandb.log({"train_loss": running_loss.item(), "epoch": epoch, "batch_idx": batch_idx})
            if writer:
                writer.add_scalar('Batch Loss', running_loss.item(), batch_idx)
            end_time = datetime.now()
            logger.info(f"Epoch [{epoch}/{config.num_epochs}] ({batch_idx+1}/{iter_per_epoch}) Train Loss: {running_loss.item():.5f} LR: {lr:.12f} Duration: {(end_time - start_time).total_seconds() / 60} minutes")
            start_time = datetime.now()

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


def plot_batch_loss(history, epoch_idx =0, save_path='loss_plot.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(history, label='Train Loss', color='blue')
    plt.xlabel('Batch Index')
    plt.ylabel('Train Loss')
    plt.title(f'Epoch {epoch_idx} Training  Loss')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def plot_train_loss(history, save_path='train_loss_plot.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(history, label='Train Loss', color='blue')
    plt.xlabel('Epoch Index')
    plt.ylabel('Train Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def init_model(config):
    if config.model == 'llama1':
        model = Llama1ForCausalLM(config)
    elif config.model == 'llama3':
        model = Llama3ForCausalLM(config)

    else:
        raise ValueError(f"Model {config.model} not supported")
    state_dict = torch.load(config.pretrain_path,map_location=config.device)
    model.load_state_dict(state_dict,strict=True)
    
    return model


def train(config):
    os.makedirs(config.model_save_path,exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_path, exist_ok=True)
    global writer
    global wandb
    
    if config.use_tensorboard:
        writer = setup_tensorboard(config.log_dir)
    else:
        writer = None

    if config.use_wandb:
        wandb.init(project="llm_dpo", name=f"dpo_{config.model}_{config.num_epochs}_{config.batch_size}_{config.lr}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else :
        wandb = None

    logger.info("Loading datasets...")
    train_dataset = DPODataset(config.data_path,config.tokenizer_path,config.max_seq_len)
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
    logger.info(f"Number of training samples: {len(train_dataset)}")

    if config.evaluate_val:
        val_dataset = DPODataset( config.val_path,config.tokenizer_path,config.max_seq_len)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,num_workers=16)
        logger.info(f"Number of validation samples: {len(val_dataset)}")
    if config.evaluate_test:
        test_dataset = DPODataset( config.test_path,config.tokenizer_path,config.max_seq_len)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,num_workers=16)
        logger.info(f"Number of test samples: {len(test_dataset)}")

    logger.info(f"Datasets loaded successfully...")


    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(),lr=config.lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(config.patience)
    
    global scaler
    if config.use_amp:
        scaler = GradScaler()
    else :
        scaler = None

    logger.info(f"Model parameters:")
    for name, param in model.named_parameters():
        logger.info(f"{name}: {param.size()}")
    
    try:
        for epoch in range(1,1+config.num_epochs):
            train_loss = train_one_epoch(train_loader,optimizer,config.device,epoch,config)
            history['train_loss'].append(train_loss)
            logger.info(f"Epoch {epoch}, average train loss: {train_loss}")
            
            plot_batch_loss(epoch_loss_list, epoch_idx=epoch, save_path=f"{config.log_dir}/train_loss_epoch_{epoch}.png")
            np.save(os.path.join(config.log_dir, f'{epoch}_training_history.npy'), epoch_loss_list)

            if config.evaluate_val:
                val_loss = evaluate(model,val_loader,config.device,config)
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
            model_save_path = os.path.join(config.model_save_path, f"{config.model}_dpo_epoch_{epoch}.pt")
            torch.save(model.state_dict(), model_save_path)
            logger.info("-"*100)
        plot_train_loss(history['train_loss'],save_path=f"{config.log_dir}/train_loss.png")
        logger.info("Fine-tuning completed.")
    except KeyboardInterrupt:
        logger.info("Fine-tuning interrupted by user.")            
    finally:
        # 保存模型
        model.eval()  # 切换到推理模式
        model_save_path = os.path.join(config.model_save_path, f"{config.model}_dpo.pt")
        torch.save(model.state_dict(), model_save_path)

        if writer is not None:
            writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model DPO")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--loss_mask", action='store_true', default=False)
    parser.add_argument("--attn_mask", action='store_true', default=False)
    parser.add_argument("--model", type=str, default='llama3')
    parser.add_argument("--pretrain_path", type=str, default='all_models/llama1_pretrain.pt')
    parser.add_argument("--use_moe", action='store_true', default=False)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--add_aux_loss", action='store_true', default=False)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-8)


    args = parser.parse_args()
    print(f"Arguments: {args}")
    set_seed(42)
    config = DPOConfig()
    config.data_path = "data/llm_data/processed/dpo.json"
    # config.data_path = "data/llm_data/processed/demo/dop.json"

    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.accumulation_steps = args.accumulation_steps
    config.loss_mask = args.loss_mask
    config.attn_mask = args.attn_mask
    config.model = args.model
    config.use_moe = args.use_moe
    config.pretrain_path = "all_logs/sft_20250818_020142_llama3/saved_models/llama3_sft.pt"
    config.lora_rank = args.lora_rank
    config.add_aux_loss = args.add_aux_loss
    config.d_model = args.d_model
    config.hidden_dim = args.hidden_dim
    config.num_layers = args.num_layers
    config.lr = args.lr
    

    # 创建包含当前时间的日志目录
    now_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.log_dir = os.path.join("logs", f"dpo_{now_timestamp}_{config.model}")
    os.makedirs(config.log_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错

    # 初始化全局logger
    logger = setup_logger(config.log_dir)
    logger.info(f"args: {args}")
    logger.info(f"config: {config}")

    start_time = datetime.now()
    logger.info(f"DPO started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("Start DPO...")
    model = init_model(config)
    logger.info(f"Model infomation: {model}")
    total_params_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"已成功加载原始模型，模型的参数量大小：{total_params_count:.3f} M")
    
    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f'LLM可训练参数量: {trainable_params:.3f} M')
    model.to(config.device)

    ref_model = init_model(config)
    # 设置ref_model为不可训练
    ref_model.requires_grad_(False)
    ref_model.to(config.device)

    train(config)
    end_time = datetime.now()
    logger.info(f"DPO completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total DPO time: {end_time - start_time}")
