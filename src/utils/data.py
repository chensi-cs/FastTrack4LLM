import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

class CustomDataset(Dataset):
    def __init__(self,data_path,tokenizer_path,max_length):
        self.data_path = data_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.max_length = max_length
        self.data = self.load_data()
        print(self.data.head())  # Print the first few rows of the dataset for debugging

    def load_data(self):
        """Load the dataset from the specified data path."""
        data=pd.read_csv(self.data_path)
        return data

    def __len__(self):
        return len(self.data)-1
    

    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        en_sentence = item['en']
        zh_sentence = item['zh']
        en_tokens = self.tokenizer.encode(en_sentence)
        zh_tokens = self.tokenizer.encode(zh_sentence)
        input_ids = en_tokens.ids
        attention_mask = en_tokens.attention_mask

        labels = zh_tokens.ids
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        else:
            padding_length = self.max_length - len(input_ids)
            input_ids += [2] * padding_length
            attention_mask += [0] * padding_length

        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        else:
            padding_length = self.max_length - len(labels)
            labels += [2] * padding_length
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


