import json
import random
import os
from typing import List, Dict, Any, Optional

def random_split_data(
    file_path: str, 
    output_dir: str = "./",
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
    shuffle: bool = True,
    overwrite: bool = False
) :
    """
    将单个JSON文件随机分割为训练集、验证集和测试集
    
    参数:
        file_path: 输入JSON文件路径
        output_dir: 输出文件目录
        train_size: 训练集比例
        val_size: 验证集比例
        test_size: 测试集比例
        random_seed: 随机种子值
        shuffle: 是否打乱数据
        overwrite: 是否覆盖已存在的输出文件
    
    返回:
        包含三个数据集的字典: {"train": [...], "val": [...], "test": [...]}
    """
    # 参数验证
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"输入文件不存在: {file_path}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_ratio = train_size + val_size + test_size
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"训练集、验证集和测试集比例之和应为1.0，当前为{total_ratio}")
    
    # 读取数据
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析错误: {str(e)}")
    
    if not isinstance(data, list):
        raise TypeError("JSON数据格式必须是数组类型")
    
    total_samples = len(data)
    print(f"总样本数: {total_samples}")
    
    if total_samples == 0:
        raise ValueError("输入数据为空")
    
    # 设置随机种子并打乱数据
    if shuffle:
        random.seed(random_seed)
        random.shuffle(data)
    
    # 计算各数据集样本数量
    train_count = int(total_samples * train_size)
    val_count = int(total_samples * val_size)
    test_count = total_samples - train_count - val_count
    
    # 分割数据
    train_data = data[:train_count]
    val_data = data[train_count:train_count+val_count]
    test_data = data[train_count+val_count:]
    
    print(f"训练集样本数: {len(train_data)} ({train_size*100:.1f}%)")
    print(f"验证集样本数: {len(val_data)} ({val_size*100:.1f}%)")
    print(f"测试集样本数: {len(test_data)} ({test_size*100:.1f}%)")
    
    # 保存到文件
    datasets = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }
    
    for name, dataset in datasets.items():
        output_path = os.path.join(output_dir, f"{name}.json")
        
        if os.path.exists(output_path) and not overwrite:
            print(f"警告: 文件已存在，跳过保存 {output_path}")
            continue
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        print(f"已保存 {name} 集到 {output_path}")
    



def random_sample(input_file, output_file, n=2, seed=42):
    """从列表或JSON文件中随机抽取n条数据并可保存到文件"""
    # 如果输入是文件路径，读取JSON数据
    if isinstance(input_file, str):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    random.seed(seed)
    samples = random.sample(data, min(n, len(data)))
    
    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=4, ensure_ascii=False)
    

if __name__ == "__main__":
    random_split_data(
        file_path="data/llm_data/processed/pretrain_hq.json",
        output_dir="data/model_data/",
        train_size=0.9,
        val_size=0.05,
        test_size=0.05,
        random_seed=42,
        shuffle=True,
        overwrite=True
    )
    # random_sample("data/llm_data/processed/pretrain_hq.json","data/model_data/train1.json",2)
    # random_sample("data/model_data/train.json","data/model_data/demo/train.json",2000)
    # random_sample("data/model_data/val.json","data/model_data/demo/val.json",2000)
    # random_sample("data/model_data/test.json","data/model_data/demo/test.json",2000)
