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
        encoder_input_ids = en_tokens.ids

        
        if len(encoder_input_ids) > self.max_length:
            encoder_input_ids = encoder_input_ids[:self.max_length]
        else:
            padding_length = self.max_length - len(encoder_input_ids)
            encoder_input_ids += [2] * padding_length

        decoder_input_ids = [0] + zh_tokens.ids  # <sos> id 为 0

        # 标签添加 <eos> 标记
        decoder_labels = zh_tokens.ids  # <eos> id 为 1

        if len(decoder_input_ids) > self.max_length:
            decoder_input_ids = decoder_input_ids[:self.max_length]
        else:
            padding_length = self.max_length - len(decoder_input_ids)
            decoder_input_ids += [2] * padding_length  # <padding> id 为 2

        if len(decoder_labels) > self.max_length:
            decoder_labels = decoder_labels[:self.max_length-1]
            decoder_labels += [1]
        else:
            padding_length = self.max_length - len(decoder_labels)-1
            decoder_labels += [2] * padding_length  # <padding> id 为 2
            decoder_labels += [1]



        if len(decoder_labels) > self.max_length:
            decoder_labels = decoder_labels[:self.max_length]
        else:
            padding_length = self.max_length - len(decoder_labels)
            decoder_labels += [2] * padding_length
        
        encoder_input_ids = torch.tensor(encoder_input_ids, dtype=torch.long)
        decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
        decoder_labels = torch.tensor(decoder_labels, dtype=torch.long)
        return {
            'encoder_input_ids': encoder_input_ids,
            'decoder_input_ids': decoder_input_ids,
            'decoder_labels': decoder_labels
        }


