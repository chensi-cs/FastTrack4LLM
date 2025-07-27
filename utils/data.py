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
        input_ids = encoding.input_ids.squeeze()
        loss_mask = ( input_ids != self.tokenizer.pad_token_id)
        x = torch.tensor(input_ids[:-1],dtype=torch.long)
        y = torch.tensor(input_ids[1:],dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:],dtype=torch.long)
        return x,y,loss_mask

    