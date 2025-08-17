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
        input_ids = input_ids.tolist()  # 转换为列表
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
                for j in range(start+1,min(end + len(self.eos_id)+1 ,self.max_seq_len)):
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
            tokenize = False  # 是否直接返回token ID（False则返回文本字符串）
        )

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        # print("input_ids:", input_ids)
        # input_ids[:-1].clone()是用来创建张量的一个副本（深拷贝），与直接赋值（x = input_ids[:-1]）的区别是：副本的修改不会影响原张量，而直接赋值是共享内存的
        # .detach()将张量从计算图中分离出来，不再跟踪其梯度（gradient）
        x = input_ids[:-1].clone().detach()
        y = input_ids[1:].clone().detach()
        # 将pad_token的部分mask为0，不参与自注意力得分计算
        attention_mask = ( input_ids != self.tokenizer.pad_token_id)
        attention_mask = attention_mask[:-1].clone().detach()

        # 仅计算assistant回答部分的loss
        loss_mask = self._get_loss_mask(input_ids)


        loss_mask = torch.tensor(loss_mask[1:]).clone().detach()
        # print("loss_mask:", loss_mask)
        return x,y,attention_mask,loss_mask

class DPODataset(Dataset):
    def __init__(self,data_path,tokenizer_path,max_seq_len):
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.data= self.load_data()
        print(len(self.data),"samples loaded from",self.data_path)
        self.bos_id = self.tokenizer('<|im_start|>assistant',add_special_tokens=False).input_ids
        self.eos_id = self.tokenizer('<|im_end|>',add_special_tokens=False).input_ids

    def load_data(self):
        with open(self.data_path, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def _get_loss_mask(self,input_ids):
        input_ids = input_ids.tolist()  # 转换为列表
        loss_mask = [0] * len(input_ids)
        i = 0
        while i <len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
         
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start+1,min(end + len(self.eos_id)+1, self.max_seq_len)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end + len(self.eos_id) < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self,index):
        conversations = self.data[index]
        chosen =  conversations['chosen']
        rejected = conversations['rejected']

        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen,
            add_generation_prompt=False , # 是否添加生成下一轮回复的提示标记
            tokenize = False  # 是否直接返回token ID（False则返回文本字符串）
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected,
            add_generation_prompt=False , # 是否添加生成下一轮回复的提示标记
            tokenize = False  # 是否直接返回token ID（False则返回文本字符串）
        )

        chosen_encoding = self.tokenizer(
            chosen_prompt,
            max_length=self.max_seq_len,
            padding= 'max_length',
            truncation=True,
            return_tensors='pt'
        )

        rejected_encoding = self.tokenizer(
            rejected_prompt,
            max_length=self.max_seq_len,
            padding= 'max_length',
            truncation=True,
            return_tensors='pt'
        )

        # [1,seq_len] > [seq_len]
        chosen_ids = chosen_encoding.input_ids.squeeze()
        chosen_loss_mask = self._get_loss_mask(chosen_ids)

        rejected_ids = rejected_encoding.input_ids.squeeze()
        rejected_loss_mask = self._get_loss_mask(rejected_ids)

        x_chosen = chosen_ids[:-1].clone().detach()
        y_chosen = chosen_ids[1:].clone().detach()

        x_rejected = rejected_ids[:-1].clone().detach()
        y_rejected = rejected_ids[1:].clone().detach()

        chosen_loss_mask = torch.tensor(chosen_loss_mask[1:]).clone().detach()
        rejected_loss_mask = torch.tensor(rejected_loss_mask[1:]).clone().detach()

        chosen_attn_mask = (chosen_ids != self.tokenizer.pad_token_id)[:-1].clone().detach()
        rejected_attn_mask = (rejected_ids != self.tokenizer.pad_token_id)[:-1].clone().detach()


        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'chosen_loss_mask': chosen_loss_mask,
            'rejected_loss_mask': rejected_loss_mask,
            'chosen_attn_mask': chosen_attn_mask,
            'rejected_attn_mask': rejected_attn_mask
        }
    