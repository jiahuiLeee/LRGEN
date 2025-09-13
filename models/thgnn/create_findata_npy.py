import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('/home/ljh/RGEN_B/train/data/financial_indicators/2024_Q2_indicators_auto_log_scaled.csv')

# 提取从第4列(current_ratio)开始的所有列数据
numerical_data = df.iloc[:, 3:].values

# 保存为npy文件
save_path = '/home/ljh/RGEN_B/train/data/financial_indicators/2024_Q2_indicators_auto_log_scaled.npy'
np.save(save_path, numerical_data)

# 验证
loaded_data = np.load(save_path)
print(f"保存成功! 数组形状: {loaded_data.shape}")