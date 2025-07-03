import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import  DataLoader
from utils.config import Config
from utils.data import CustomDataset
from models.model import TransformerModel
from tokenizers import Tokenizer
import matplotlib.pyplot as plt

def train_one_epoch(model, train_loader, optimizer,scheduler, device, epoch, config):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
    criterion = nn.CrossEntropyLoss()
    accumulation_steps = 32
    for batch_idx, batch in enumerate(train_loader):
        accumulation_steps+=1
        encoder_input_ids = batch['encoder_input_ids'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        decoder_labels = batch['decoder_labels'].to(device)
        # print("encoder_input_ids:",encoder_input_ids)
        # print("decoder_input_ids:",decoder_input_ids)
        # print("decoder_labels:",decoder_labels)
        print(f"Batch {batch_idx+1}/{len(train_loader)}: encoder_input_ids shape: {encoder_input_ids.shape},decoder_input_ids shape: {decoder_input_ids.shape}, decoder_labels shape: {decoder_labels.shape}")
        
        if (batch_idx+1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # 更新学习率
            # 梯度清零
            optimizer.zero_grad()

        # 前向传播
        output_logits = model(encoder_input_ids, decoder_input_ids)
        # print(f"Output logits shape: {output_logits.shape}")
        
        predicted_indices = torch.argmax(output_logits, dim=-1)
        # print(f"Predicted indices shape: {predicted_indices.shape}")
        # print(predicted_indices)
        # print(f"Labels shape: {labels.shape}")
        # print(labels)
        # 计算损失
        loss = criterion(output_logits.view(-1, config.vocab_size), decoder_labels.view(-1))
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
        encoder_input_ids = batch['encoder_input_ids'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        decoder_labels = batch['decoder_labels'].to(device)
        with torch.no_grad():
            output_logits = model(encoder_input_ids, decoder_input_ids)
            loss = criterion(output_logits.view(-1, config.vocab_size), decoder_labels.view(-1))
            total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def load_model_for_translation(model_path, tokenizer_path):
    # 加载分词器
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # 获取词汇表大小
    vocab_size = tokenizer.get_vocab_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型
    model= TransformerModel(
        d_model=128,
        n_head=8,
        vocab_size=vocab_size
    )
    
    # 加载模型权重
    # 正确的方式：将map_location参数放在torch.load()中
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    
    return model, tokenizer, device
    
def translate_sentence(model, sentence, tokenizer, device, max_length=500):
    model.eval()
    input_ids = tokenizer.encode(sentence)
    input_ids = input_ids.ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    else:
        padding_length = max_length - len(input_ids)
        input_ids += [2] * padding_length
        

    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    # print(f"input_ids: {input_ids}")
    # print(f"input_ids shape: {input_ids.shape}")

    # 初始化解码器输入，以 <sos> 开始
    decoder_input_ids = torch.tensor([[0]], dtype=torch.long).to(device)


    predicted_indices = []
    for i in range(max_length):
        # print(f"预测第 {i+1} 个词")
        with torch.no_grad():
            # print(f"decoder_input_ids: {decoder_input_ids}")
            # print(f"decoder_input_ids shape: {decoder_input_ids.shape}")
            output = model(input_ids, decoder_input_ids)
            # print(f"output: {output}")
            # print(f"output shape: {output.shape}")

            next_token = torch.argmax(output[:, -1, :], dim=-1)
            # output[:, -1, :]：从 output 中取出最后一个时间步的预测得分，形状为 (batch_size, vocab_size)。
            # dim=-1：指定在最后一个维度（即词汇表维度）上进行最大值索引的查找。

            predicted_indices.append(next_token.item())
            # next_token.item()：将 next_token 张量转换为 Python 标量值，因为 append 方法需要一个标量值作为参数。

            # print(f"next_token: {next_token}")
            # print(f"next_token shape: {next_token.shape}")

            # print(f"next_token.unsqueeze(0): {next_token.unsqueeze(0)}")
            # print(f"next_token.unsqueeze(0) shape: {next_token.unsqueeze(0).shape}")

            decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == 1:  # 如果预测到 <eos> 则停止
                break
        # print("-"*100)
    # print(f"predicted_indices: {predicted_indices}")
    predicted_sentence = tokenizer.decode(predicted_indices)
    # print(f"predicted_sentence: {predicted_sentence}")
    return predicted_sentence


def translate_test(model_path, tokenizer_path,test_path,out_path):
    model, tokenizer, device = load_model_for_translation(model_path, tokenizer_path)
    data= pd.read_csv(test_path)
    results = []
    for index, row in data.iterrows():
        print(f"预测第 {index} 句 :")
        en_sentence = row['en']
        zh_sentence = row['zh']
        predict_setence = translate_sentence(model, en_sentence, tokenizer, device)
        tmp={
            "index": index,
            "en": en_sentence,
            "zh": zh_sentence,
            "predict": predict_setence
        }
        print(tmp)
        print("-"*100)
        results.append(tmp)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


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
        hidden_dim=config.hidden_dim,
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
    
    test_dataset = CustomDataset(config.test_path, config.tokenizer_path, config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,num_workers=0)

    print("Datasets loaded successfully.")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    optimizer = optim.Adam(
        model.parameters(),
        betas=(0.9, 0.98),          # 设置一阶矩和二阶矩估计的指数衰减率
        eps=1e-09                   # 设置数值稳定性参
        )

    def lr_lambda(epoch):
        if (epoch == 0):
            return 0
        d_model=config.embedding_dim
        warmup_steps=4000
        lr= d_model ** -0.5 * min(epoch ** -0.5, epoch * warmup_steps ** -1.5)
        return lr
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


    # 训练循环
    history = {"loss": [], "val_loss": []}
    best_val_loss = float('inf')

    # 初始化早停相关变量
    patience = 10  # 容忍周期数
    no_improve_count = 0  # 验证集损失未改善的周期数

    for epoch in range(1,1+config.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,device, epoch, config)
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
            no_improve_count = 0  # 重置计数器
        else:
            no_improve_count += 1  # 验证集损失未改善
            print(f"Validation loss did not improve! Current patience: {no_improve_count}/{patience}")
    

        np.save(os.path.join(config.log_dir, 'training_history.npy'), history)

        # 检查是否需要早停
        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch} all epochs and {patience} epochs without improvement.")
            break

        
     # 加载最佳模型
    checkpoint = torch.load(os.path.join(config.checkpoint_path, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(config.model_save_path, 'final_model.pth'))
    print("Training complete. Best model saved.")
    print("Final training history:", history)
    # 绘制训练历史
    plot_training_history(history)

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('../logs/training_history.png')

if __name__ == "__main__":
    config= Config()
    # config.data_path="../../llm_data/translate_data/processed/train.csv"
    # config.validation_path = '../../llm_data/translate_data/processed/val.csv'
    # config.test_path = '../../llm_data/translate_data/processed/test.csv'
    config.data_path="../data/processed/train.csv"
    config.validation_path = "../data/processed/val.csv"
    config.test_path = "../data/processed/test.csv"
    config.tokenizer_path = './data/tokenizer.json'
    config.model_save_path = '../saved_models'
    config.log_dir = '../logs'
    config.checkpoint_path = '../checkpoints'
    config.embedding_dim = 256
    config.hidden_dim = 1024
    config.max_length = 1000
    config.learning_rate = 0.001
    config.batch_size = 1
    config.num_epochs =30

    train(config)
    # model, tokenizer, device = load_model_for_translation('../saved_models/final_model.pth', './data/tokenizer.json')
    # print(model)
    # sentence = "Hello, how are you?"
    # translate_sentence(model, sentence, tokenizer, device)
    translate_test('../saved_models/final_model.pth', './data/tokenizer.json', '../data/processed/test.csv', '../data/processed/test_predict.json')
