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
        self.bos_id = self.tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = self.tokenizer('<|im_end|>', add_special_tokens=False).input_ids


    def load_data(self):
        with open(self.data_path, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def _get_loss_mask(self,input_ids):
        # 为了让模型专注学习 “如何生成正确的回答” ，loss只关注输出
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end+len(self.eos_id)]== self.eos_id:
                        break
                    end += 1
                # range右开
                for j in range(start+1,min(end + len(self.eos_id)+1 ,len(input_ids))):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

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
        # 将pad_token的部分mask为0，不参与自注意力得分计算
        attention_mask = ( input_ids != self.tokenizer.pad_token_id)
        attention_mask = attention_mask[:-1].clone().detach()

        # 仅计算assistant回答部分的loss
        loss_mask = self._get_loss_mask(input_ids)
        loss_mask = loss_mask[1:].clone().detach()
        return x,y,attention_mask,loss_mask

    
