import pandas as pd
import glob
import os

# 获取../../data/目录下所有.parquet文件路径
parquet_files = glob.glob('../../data/*.parquet')

# 遍历所有parquet文件
for file in parquet_files:
    # 构造对应的csv文件名
    csv_file = os.path.splitext(file)[0] + '.csv'
    
    # 读取Parquet文件
    df = pd.read_parquet(file, engine='pyarrow')  # 或'fastparquet'
    
    # 保存为CSV
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f'已转换: {file} -> {csv_file}')
    
# # 读取Parquet文件
# df = pd.read_parquet('../../data/test-00000-of-00001.parquet', engine='pyarrow')  # 或'fastparquet'
# # 保存为CSV
# df.to_csv('../../data/test-00000-of-00001.csv', index=False, encoding='utf-8')