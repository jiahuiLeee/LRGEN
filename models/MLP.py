import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=18, hidden_dims=[128, 64], num_classes=2):
        super(MLP, self).__init__()
        
        # 定义全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])  # 输入层到第一隐藏层
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])  # 第一隐藏层到第二隐藏层
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)  # 第二隐藏层到第三隐藏层
        # self.fc4 = nn.Linear(hidden_dims[2], num_classes)  # 第三隐藏层到输出层
        
        # 定义激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 前向传播
        x = self.relu(self.fc1(x))  # 第一层 + ReLU
        x = self.relu(self.fc2(x))  # 第二层 + ReLU
        # x = self.relu(self.fc3(x))  # 第三层 + ReLU
        x = self.fc3(x)  # 第三层 + ReLU
        # x = self.fc4(x)  # 输出层（无激活函数，通常配合交叉熵损失函数使用）
        return x
