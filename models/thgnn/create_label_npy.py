import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('/home/ljh/RGEN_B/train/data/label/2024_Q2_3_label.csv')

# 提取从第4列(current_ratio)开始的所有列数据
numerical_data = df.iloc[:, 2:].values

# 保存为npy文件
save_path = '/home/ljh/RGEN_B/train/data/label/2024_Q2_3_label.npy'
np.save(save_path, numerical_data)

# 验证
loaded_data = np.load(save_path)
print(loaded_data)
print(f"保存成功! 数组形状: {loaded_data.shape}")