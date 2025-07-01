from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import json 
import pandas as pd
import re

def clean_data(input_file, output_file):
    """
    清洗数据
    """
    
    df = pd.read_csv(input_file)

    print(len(df))

    corresponding_data = []
    for i in range(len(df)):
        s = df['translation'][i]  # 获取每行的字符串数据

        en = re.search(r"'en':\s*['\"](.*?)['\"]\s*,\s*'zh':", s)
        zh = re.search(r"'zh':\s*['\"](.*?)['\"]}", s)

        if en and zh:
            corresponding_data.append([en.group(1), zh.group(1)])
    print(len(corresponding_data))

    # 将中英文对应数据保存到新的 CSV 文件
    result_df = pd.DataFrame(corresponding_data, columns=['en', 'zh'])
    # result_df.to_csv(output_file, index=False)
    # 保存英文和中文到文件
    # result_df['en'].to_csv('test_en.txt', index=False, header=False)
    # result_df['zh'].to_csv('test_zh.txt', index=False, header=False)


# input_file_list=[
#     '../../data/raw/train-00005-of-00013.csv',
#     '../../data/raw/train-00006-of-00013.csv',
#     '../../data/raw/train-00007-of-00013.csv',
#     '../../data/raw/train-00009-of-00013.csv',
#     '../../data/raw/train-00010-of-00013.csv',
#     '../../data/raw/train-00011-of-00013.csv',
#     '../../data/raw/train-00012-of-00013.csv',
# ]
# output_file_list = [
#     '../../data/processed/train-00005-of-00013.csv',
#     '../../data/processed/train-00006-of-00013.csv',
#     '../../data/processed/train-00007-of-00013.csv',
#     '../../data/processed/train-00009-of-00013.csv',
#     '../../data/processed/train-00010-of-00013.csv',
#     '../../data/processed/train-00011-of-00013.csv',
#     '../../data/processed/train-00012-of-00013.csv',
# ]
# for input_file, output_file in zip(input_file_list, output_file_list):
#     clean_data(input_file, output_file)
#     print(f"Data cleaning completed and saved to {output_file}")

# input_file = '../../data/raw/test-00000-of-00001.csv'
# output_file = '../../data/processed/test-00000-of-00001.csv'
# clean_data(input_file, output_file)
# print(f"Data cleaning completed and saved to {output_file}")

def train_bpe_tokenizer(file_path, language="en", vocab_size=10000):
    tokenizer = Tokenizer(models.BPE())
    
    # 预分词器（英文按空格，中文按字符）
    if language == "en":
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 训练配置
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"]
    )
    
    # 训练并保存
    tokenizer.train([file_path], trainer)
    tokenizer.save(f"{language}_bpe_tokenizer.json")
    return tokenizer

# 训练英文BPE分词器
# en_tokenizer = train_bpe_tokenizer("../../data/processed/test_en.txt", "en", 1000)
# # 训练中文BPE分词器
# zh_tokenizer = train_bpe_tokenizer("../../data/processed/test_zh.txt", "zh", 1000)

# 加载英文分词器
# en_tokenizer = Tokenizer.from_file("en_bpe_tokenizer.json")

# # 加载中文分词器
# zh_tokenizer = Tokenizer.from_file("zh_bpe_tokenizer.json")
# text_en = "Hello world!"
# text_zh = "你好，世界！"

# # 英文编码
# encoded = en_tokenizer.encode(text_en)
# print(encoded.tokens)  #
# print(encoded.ids)     

# # 中文编码
# encoded = zh_tokenizer.encode(text_zh)
# print(encoded.tokens)  # 输出分词结果：["Hello", "world", "!"]
# print(encoded.ids)     # 输出对应的Token ID序列

# # # 单条解码
# decoded_text = zh_tokenizer.decode(encoded.ids)
# print(decoded_text)  # 输出"你好世界 Hello world"

tokenizer = Tokenizer.from_file("../data/tokenizer.json")
encoded = tokenizer.encode('Hello world!你好世界，我是实习生')
print(encoded.tokens)  #
print(encoded.ids) 
print(encoded.attention_mask)
decoded_text = tokenizer.decode(encoded.ids)
print(decoded_text)  # 输出"你好世界 Hello world"


