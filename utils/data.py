import torch
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_seq_len):
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        self.data = self.load_data()
        print(len(self.data),"samples loaded from",self.data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
    
    def load_data(self):
        with open(self.data_path, 'r') as file:
            data = json.load(file)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['text']
        encoding = self.tokenizer(
            str(sentence),
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # torch.Tensor.squeeze() 用于移除张量中所有尺寸为 1 的维度，如形状 [1, max_length] → [max_length]（移除了尺寸为 1 的 batch 维度）。
        input_ids = encoding.input_ids.squeeze()
        attention_mask = ( input_ids != self.tokenizer.pad_token_id)
        x = input_ids[:-1].clone().detach()
        y = input_ids[1:].clone().detach()
        attention_mask = attention_mask[:-1].clone().detach()

        return x,y,attention_mask
    
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_seq_len):
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        self.data = self.load_data()
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def load_data(self):
        with open(self.data_path, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        # 对话列表
        conversations = self.data[index]['conversations']
        messages = []
        for i , conversation in enumerate(conversations):
            role = 'user' if i % 2== 0 else 'assistant'
            messages.append({'role':role,'content':conversation['content']})
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False , # 是否添加生成下一轮回复的提示标记
            tokenize = True  # 是否直接返回token ID（False则返回文本字符串）
        )
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        x = input_ids[:-1].clone().detach()
        y = input_ids[1:].clone().detach()
        loss_mask = ( input_ids != self.tokenizer.pad_token_id)
        loss_mask = loss_mask[1:].clone().detach()
        return x,y,loss_mask

    
