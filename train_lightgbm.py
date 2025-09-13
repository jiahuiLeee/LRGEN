import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from utils import *

def evaluate(model, X_test, y_test):
    # 直接使用 LightGBM 模型进行预测
    y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)  # 获取预测的概率
    
    # 对于二分类问题，y_pred_prob 是每个样本属于正类的概率
    if y_pred_prob.ndim == 1:  # 二分类时
        y_pred_class = (y_pred_prob >= 0.5).astype(int)  # 概率大于 0.5 则为正类 (1)，否则为负类 (0)
    else:  # 多分类时
        y_pred_class = np.argmax(y_pred_prob, axis=1)  # 对于多分类，选择概率最大的类别作为预测标签
    
    # 计算各类指标
    accuracy = accuracy_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class, average='macro')
    precision = precision_score(y_test, y_pred_class, average='macro')
    recall = recall_score(y_test, y_pred_class, average='macro')
    
    # AUC 计算
    if y_pred_prob.ndim == 1:  # 二分类
        auc = roc_auc_score(y_test, y_pred_prob)  # AUC 只考虑正类的概率
    else:  # 多分类
        auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')  # 使用 One-vs-Rest 计算 AUC
    
    return accuracy, f1, precision, recall, auc


# 训练LightGBM模型的代码
def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"device:{device}")

    # 构建训练集和测试集的特征矩阵
    train_fin_index_emb = read_financial_index(config['train_financial_index'])
    test_fin_index_emb = read_financial_index(config['test_financial_index'])
    
    # 假设数据已经准备好，包含了节点特征
    X_train = train_fin_index_emb  # 特征矩阵
    X_test = test_fin_index_emb    # 测试集特征矩阵
    
    # 读取标签
    train_labels = read_labels(config['train_label'])
    test_labels = read_labels(config['test_label'])

    # 标准化特征数据
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    X_train_scaled = X_train
    X_test_scaled = X_test

    # 训练和测试数据集
    y_train = list(train_labels)
    y_test = list(test_labels)

    # LightGBM模型参数
    seed = np.random.randint(0, 10**9)

    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 30,
        'learning_rate': 1e-4,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'num_class': 3,
        'seed': seed,
        'bagging_seed': seed,
        'feature_fraction_seed': seed,
        'deterministic': False  # 明确禁用确定性（有些版本可用）
    }
    # params = {
    #     'objective': 'multiclass',          # 二分类问题
    #     'metric': 'multi_logloss',       # 评估指标是二分类错误率
    #     'boosting_type': 'gbdt',        # GBDT模型
    #     'num_leaves': 30,               # 树的最大叶子节点数
    #     'learning_rate': 0.0001,          # 学习率
    #     'feature_fraction': 0.9,        # 每次迭代使用的特征比例
    #     'bagging_fraction': 0.8,        # 每次迭代使用的样本比例
    #     'bagging_freq': 5,              # 每 5 次迭代进行一次采样
    #     'verbose': 0,
    #     'num_class': 3                      # 设置类别数为3
    # }

    # 创建LightGBM的Dataset对象
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

    # 训练LightGBM模型
    print("Training LightGBM model...")
    model = lgb.train(
        params, 
        train_data, 
        num_boost_round=500,  # 最大迭代次数
        valid_sets=[test_data], 
        # early_stopping_rounds=50  # 早停机制，避免过拟合
    )

    # 评估模型
    print("Evaluating model on test data...")
    test_accuracy, test_f1, test_precision, test_recall, test_auc = evaluate(model, X_test_scaled, y_test)
    
    # 打印评估结果
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # 保存最佳模型
    model.save_model("best_lightgbm_model.txt")

if __name__ == "__main__":
    main()
