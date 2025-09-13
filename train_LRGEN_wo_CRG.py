from utils import *
from models.LRGEN import LRGEN_wo_CRG
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch
import torch.nn.functional as F

import csv
import yaml
from datetime import datetime
import shutil

def evaluate(model, test_graph, test_dataloader, test_fin_index_emb, test_labels, classify_criterion):
    model.eval()
    emb_list = []
    last_batch = False
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch_idx, subgraph_triple in enumerate(test_dataloader):
            if batch_idx == len(test_dataloader) - 1:
                last_batch = True
            subgraph_triple = subgraph_triple.to('cuda')
            out, q, _ = model(test_graph, subgraph_triple, test_fin_index_emb, last_batch, emb_list)
            emb_list.append(q.to('cpu'))
            
            if out is not None:
                # 计算损失
                fuse_loss = 1
                classify_loss = classify_criterion(out, test_labels)
                # 计算准确率
                preds = torch.argmax(out, dim=1)
                accuracy = accuracy_score(test_labels.cpu().numpy(), preds.cpu().numpy())
                f1 = f1_score(test_labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
                precision = precision_score(test_labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
                recall = recall_score(test_labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
                # 计算 AUC
                # 获取预测概率
                probs = F.softmax(out, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(test_labels.cpu().numpy())
                # 如果是二分类（只需要考虑正类的概率）
                if probs.shape[1] == 2:
                    auc = roc_auc_score(test_labels.cpu().numpy(), probs[:, 1])  # 只需要第二类的概率
                else:
                    # 多分类（使用 One-vs-Rest 方式）
                    auc = roc_auc_score(test_labels.cpu().numpy(), probs, multi_class='ovr')
    
                return accuracy, fuse_loss, classify_loss, f1, precision, recall, auc

# 加载配置文件
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
# 设置设备
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
# 数据集和数据加载器
train_emb_dataset = NpyDataset(root_dir=config['train_embedding'])
test_emb_dataset = NpyDataset(root_dir=config['test_embedding'])
train_dataloader = DataLoader(train_emb_dataset, batch_size=config['batch_size'], shuffle=False) # must False
test_dataloader = DataLoader(test_emb_dataset, batch_size=config['batch_size'], shuffle=False)  # must False
# 读取训练数据和测试数据
rel_graph_train_data = load_graph_data_from_file(config['train_graph_data'])
rel_graph_test_data = load_graph_data_from_file(config['test_graph_data'])
# 构建训练集和测试集的异质图
train_graph = build_dgl_hetero_graph(rel_graph_train_data).to(device)
test_graph = build_dgl_hetero_graph(rel_graph_test_data).to(device)
# 标签
train_labels = read_labels(config['train_label']).to(device)
test_labels = read_labels(config['test_label']).to(device)
# 读取财务指标数据
train_fin_index_emb = read_financial_index(config['train_financial_index']).to(device)
test_fin_index_emb = read_financial_index(config['test_financial_index']).to(device)

# 实例化模型
model = LRGEN_wo_CRG(
    group=config['LRGEN']['group'],
    drop_ratio=config['LRGEN']['drop_ratio'],
    in_channels=config['LRGEN']['in_channels'],
    hidden_channels=config['LRGEN']['hidden_channels'],
    shufunit_out_channel=config['LRGEN']['shufunit_out_channels'],
    out_kg_dim=config['LRGEN']['out_kg_dim'],
    in_rgcn_dim=config['LRGEN']['in_rgcn_dim'],
    out_rel_dim=config['LRGEN']['out_rel_dim'],
    num_relations=config['LRGEN']['num_relations'],
    num_bases=config['LRGEN']['num_bases'],
    n_layers=config['LRGEN']['n_layers'],
    fin_dim=config['LRGEN']['fin_dim'],
    class_dim=config['LRGEN']['class_dim'],
    n_class=config['LRGEN']['class_num']
).to(device)

# 损失函数
classify_criterion = nn.CrossEntropyLoss()
optimizer_fuse_triple_layer = torch.optim.Adam(model.parameters(), lr=config['fuse_lr'])
optimizer_classify = torch.optim.Adam(model.parameters(), lr=config['classify_lr'])


# =================================================================================================================
# 开始训练
# 创建当前时间戳文件夹
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'./logs/{timestamp}'
os.makedirs(log_dir, exist_ok=True)
# 复制配置文件到日志文件夹
shutil.copy('config.yaml', log_dir)
# 创建CSV文件来保存训练日志
log_file = os.path.join(log_dir, 'training_log.csv')
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'epoch', 
        'train_accuracy', 'fusion_loss', 'classification_loss', 
        'test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_auc',
        'best_epoch',
    ])  # CSV表头

    best_test_accuracy = 0
    best_epoch = 0
    best_params = None
    # 训练循环
    num_epochs = config['num_epochs']
    emb_list = []
    
    # plt
    train_loss1 = []
    test_loss1 = []
    train_loss2 = []
    test_loss2 = []
    train_acc = []
    test_acc = []
    
    for epoch in range(num_epochs):
        last_batch = False
        model.train()
        emb_list = []
        running_loss = 0.0
        for batch_idx, subgraph_triple in enumerate(train_dataloader):
            if batch_idx == len(train_dataloader) - 1:
                last_batch = True
            # 将批次数据移到GPU
            subgraph_triple = subgraph_triple.to(device)
            optimizer_fuse_triple_layer.zero_grad()
            optimizer_classify.zero_grad()
            out, q, _ = model(train_graph, subgraph_triple, train_fin_index_emb, last_batch, emb_list)
            emb_list.append(q.detach().to('cpu'))
            if out is None:
                # 计算损失
                fuse_loss = 0
            else:
                classify_loss = classify_criterion(out, train_labels)
                total_loss = classify_loss
                total_loss.backward()
                optimizer_classify.step()
                # 计算训练准确率
                preds = torch.argmax(out, dim=1)
                train_accuracy = accuracy_score(train_labels.cpu().numpy(), preds.cpu().numpy())
                # 测试
                test_accuracy, test_fuse_loss, test_classify_loss, f1, precision, recall, auc = \
                    evaluate(model, test_graph, test_dataloader, test_fin_index_emb, test_labels, classify_criterion)

                train_loss1.append(fuse_loss)
                train_loss2.append(classify_loss.detach().cpu().numpy())
                test_loss1.append(test_fuse_loss)

                test_loss2.append(test_classify_loss.detach().cpu().numpy())
                train_acc.append(train_accuracy)
                test_acc.append(test_accuracy)

                plot_metrics(train_loss1, test_loss1, train_loss2, test_loss2, train_acc, test_acc, log_dir)
                
                # 如果当前测试准确率更好，更新best_test_accuracy
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    best_auc = auc
                    best_epoch = epoch + 1  # epoch从0开始，因此需要加1
                    best_params = model.state_dict()  # 保存当前模型的参数
                    # 将最好的指标写入文件
                    with open(os.path.join(log_dir, "best_metrics.txt"), mode='w') as metrics_file:
                        metrics_file.write(f"Best Epoch: {best_epoch}\n")
                        metrics_file.write(f"Best Test Accuracy: {best_test_accuracy}\n")
                        metrics_file.write(f"Best F1: {best_f1}\n")
                        metrics_file.write(f"Best Precision: {best_precision}\n")
                        metrics_file.write(f"Best Recall: {best_recall}\n")
                        metrics_file.write(f"Best AUC: {best_auc}\n")

                
                # 打印信息
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"Train Loss1: {fuse_loss:.4f}, Train Loss2: {classify_loss:.4f}, Train Accuracy: {train_accuracy}")
                print("*****test*****")
                print(f"Test Loss1: {test_fuse_loss:.4f}, Test Loss2: {test_classify_loss:.4f}, Test Accuracy: {test_accuracy}")
                print(f"Test F1: {f1}")
                print(f"Test Precision: {precision}")
                print(f"Test Recall: {recall}")
                print(f"Test AUC: {auc}")
                print("="*20)

                # 保存训练数据到CSV
                writer.writerow([
                    epoch + 1, train_accuracy, fuse_loss.item(), classify_loss.item(),
                    test_accuracy, f1, precision, recall, auc,
                    best_epoch
                ])

# 保存最佳模型
torch.save(best_params, os.path.join(log_dir, "best_model.pth"))

# 训练完成后重命名文件夹
final_log_dir = f'{log_dir}_{best_test_accuracy:.4f}'
os.rename(log_dir, final_log_dir)

print(f"Training complete. Achieved at Epoch [{best_epoch}/{num_epochs}]")
print(f"Best Test Accuracy: {best_test_accuracy}")
print(f"Best Accuracy: {best_test_accuracy}")
print(f"Best F1: {best_f1}")
print(f"Best Precision: {best_precision}")
print(f"Best Recall: {best_recall}")
print(f"Best AUC: {best_auc}")
print(f"Logs saved to {final_log_dir}")
