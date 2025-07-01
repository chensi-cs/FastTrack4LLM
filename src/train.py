import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import  DataLoader
from utils.config import Config
from utils.data import CustomDataset
from models.model import TransformerModel
from tokenizers import Tokenizer

def train_one_epoch(model, train_loader, optimizer, device, epoch, config):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
    criterion = nn.CrossEntropyLoss()
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        print(f"Batch {batch_idx+1}/{len(train_loader)}: input_ids shape: {input_ids.shape}, labels shape: {labels.shape}")
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        output_logits = model(input_ids,labels)
        # print(f"Output logits shape: {output_logits.shape}")
        
        predicted_indices = torch.argmax(output_logits, dim=-1)
        # print(f"Predicted indices shape: {predicted_indices.shape}")
        # print(predicted_indices)
        # print(f"Labels shape: {labels.shape}")
        # print(labels)
        # 计算损失
        loss = criterion(output_logits.view(-1, config.vocab_size), labels.view(-1))
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        total_loss += loss.item()

        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    average_loss = total_loss / len(train_loader)
    return average_loss


def evaluate(model, val_loader, device, config):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            output_logits = model(input_ids,labels)
            loss = criterion(output_logits.view(-1, config.vocab_size), labels.view(-1))
            total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train(config):
    os.makedirs(config.model_save_path,exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_path, exist_ok=True)

    # 打印配置信息
    print(f"Training with config: {config}")

    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    config.vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {config.vocab_size}")

    device = config.device
    model= TransformerModel(
        d_model=config.embedding_dim,
        n_head=config.n_head,
        vocab_size=config.vocab_size
    )
    model.to(device)

    # 加载数据
    print("Loading datasets...")
    train_dataset = CustomDataset(config.data_path, config.tokenizer_path, config.max_length)
    train_loader =  DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,num_workers=0)


    val_dataset = CustomDataset(config.validation_path, config.tokenizer_path, config.max_length)
    val_loader =  DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,num_workers=0)
    
    print("Datasets loaded successfully.")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练循环
    history = {"loss": [], "val_loss": []}
    best_val_loss = float('inf')

    for epoch in range(1,1+config.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, config)
        history["loss"].append(train_loss)

        val_loss = evaluate(model, val_loader, device, config)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(config.checkpoint_path, 'best_model.pth'))
            print(f"New best model saved with val loss: {val_loss:.4f}")

        np.save(os.path.join(config.log_dir, 'training_history.npy'), history)


     # 加载最佳模型
    checkpoint = torch.load(os.path.join(config.checkpoint_path, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 保存最终模型
    model.save_pretrained(config.model_save_path)


if __name__ == "__main__":
    config= Config()
    config.data_path="../data/processed/train.csv"
    config.validation_path = '../data/processed/val.csv'
    config.test_path = '../data/processed/test.csv'
    config.tokenizer_path = './data/tokenizer.json'
    config.model_save_path = '../data/saved_models'
    config.log_dir = '../data/logs'
    config.checkpoint_path = '../data/checkpoints'
    config.embedding_dim = 128
    config.max_length = 256
    config.learning_rate = 0.001
    config.batch_size = 32
    config.num_epochs = 10

    train(config)



