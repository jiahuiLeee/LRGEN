from utils import *

import os
from models.model import THGNN
from criterions import *

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch
import torch.nn.functional as F

import csv
import yaml
from datetime import datetime
import shutil

def train_gnn(g, x, y, model, optimizer, weight_node=None, Alpha=0.01, local_graph_loader=None,  epoch=None):
    # print(g.device, x.device, y.device, train_idx.device)
    model.train()
    probs, contrast_loss, logits, _ = model(g, x, local_graph_loader)
    # 计算准确率
    preds = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
    optimizer.zero_grad()
    loss = torch.nn.CrossEntropyLoss()(logits, y)        
    loss_final = loss + contrast_loss * Alpha
    loss_final.backward()
    optimizer.step()
    return contrast_loss, loss, accuracy


def test_gnn(g, x, y, model, local_graph_loader=None):
    with torch.no_grad():
        model.eval()
        _, contrast_loss, logits, feature = model(g, x, local_graph_loader)
        classify_loss = torch.nn.CrossEntropyLoss()(logits, y)
        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        accuracy = accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        precision = precision_score(y.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        recall = recall_score(y.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        # 计算 AUC
        # 获取预测概率
        probs = F.softmax(logits, dim=1).cpu().numpy()
        # 如果是二分类（只需要考虑正类的概率）
        if probs.shape[1] == 2:
            auc = roc_auc_score(y.cpu().numpy(), probs[:, 1])  # 只需要第二类的概率
        else:
            # 多分类（使用 One-vs-Rest 方式）
            auc = roc_auc_score(y.cpu().numpy(), probs, multi_class='ovr')
        return accuracy, contrast_loss, classify_loss, f1, precision, recall, auc


def evaluate(model, test_graph, test_dataloader, test_fin_index_emb, test_labels, fuse_triple_criterion, classify_criterion):
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
            # out, q, k, _ = model(test_graph, subgraph_triple, test_fin_index_emb, last_batch, emb_list)
            out, q, _ = model(test_graph, subgraph_triple, test_fin_index_emb, last_batch, emb_list)
            emb_list.append(q.to('cpu'))
            
            if out is not None:
                # 计算损失
                # fuse_loss = fuse_triple_criterion(q, k)
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
    
                return fuse_loss, classify_loss, accuracy, f1, precision, recall, auc

# 加载配置文件
with open('/home/ljh/RGEN_B/train/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
# 设置设备
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

# load datasets
train_data_mat, train_label_mat, train_local_graph_loader, train_g = build_dataset(
    path_X=config['THGNN']['train_path_x'],
    path_Y=config['THGNN']['train_path_y'],
    path_tribes=config['THGNN']['train_path_tribe_files'],
    path_tribes_order=config['THGNN']['path_tribe_order'],
    g_path=config['THGNN']['train_path_g'])
train_x = torch.from_numpy(train_data_mat).float().to(device)
train_y = read_labels(config['train_label']).to(device)
train_g = train_g.to(device)

test_data_mat, test_label_mat, test_local_graph_loader, test_g = build_dataset(
    path_X=config['THGNN']['test_path_x'],
    path_Y=config['THGNN']['test_path_y'],
    path_tribes=config['THGNN']['test_path_tribe_files'],
    path_tribes_order=config['THGNN']['path_tribe_order'],
    g_path=config['THGNN']['test_path_g'])
test_x = torch.from_numpy(test_data_mat).float().to(device)
test_y = read_labels(config['test_label']).to(device)
test_g = test_g.to(device)

# 实例化模型
model = THGNN(
    in_size=config['THGNN']['in_size'],
    out_size=config['THGNN']['class_num'],
    hidden=config['THGNN']['hidden_channels'],
    norm=config['THGNN']['norm'],
    use_attnFusioner=config['THGNN']['use_attnFusioner'],
    tribe_encoder_type=config['THGNN']['tribe_encoder_type'],
    local_layer_num=config['THGNN']['local_layer_num'],
).to(device)

# params = "/home/ljh/RGEN_B/train/logs/RGEN_20250518_161542_0.8833/best_model.pth"
# 加载模型参数
# model.load_state_dict(torch.load(params))
# 损失函数
# fuse_triple_criterion = InfoNCELoss(temperature=0.07)
# classify_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['THGNN']['lr'])
# 定义 ReduceLROnPlateau 调度器，监控验证集的损失，并在没有提升时减少学习率
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True, cooldown=5)

# =================================================================================================================
# 开始训练
# 创建当前时间戳文件夹
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f"/home/ljh/RGEN_B/train/logs/THGNN_{timestamp}"
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
    a = config['THGNN']['a']
    b = config['THGNN']['b']
    # 训练循环
    num_epochs = config['THGNN']['num_epochs']
    emb_list = []
    
    # plt
    train_loss1 = []
    test_loss1 = []
    train_loss2 = []
    test_loss2 = []
    train_acc = []
    test_acc = []
    
    for epoch in range(num_epochs):
        train_loss_contrast, train_loss_clf, train_accuracy = train_gnn(train_g, train_x, train_y, model, 
                                                            optimizer,
                                                            weight_node=None, Alpha=config['THGNN']['a'], 
                                                            local_graph_loader=train_local_graph_loader, 
                                                            epoch=epoch)
        
        test_accuracy, test_loss_contrast, test_loss_clf, f1, precision, recall, auc = test_gnn(test_g, test_x, test_y, model,
                                                                                 local_graph_loader=test_local_graph_loader)
        
        train_loss1.append(train_loss_contrast.detach().cpu().numpy())
        train_loss2.append(train_loss_clf.detach().cpu().numpy())
        test_loss1.append(test_loss_contrast.detach().cpu().numpy())
        test_loss2.append(test_loss_clf.detach().cpu().numpy())
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
        print(f"Train Loss1: {train_loss_contrast:.4f}, Train Loss2: {train_loss_clf:.4f}, Train Accuracy: {train_accuracy}")
        print("*****test*****")
        print(f"Test Loss1: {test_loss_contrast:.4f}, Test Loss2: {test_loss_clf:.4f}, Test Accuracy: {test_accuracy}")
        print(f"Test F1: {f1}")
        print(f"Test Precision: {precision}")
        print(f"Test Recall: {recall}")
        print(f"Test AUC: {auc}")
        print("="*20)

        # 保存训练数据到CSV
        writer.writerow([
            epoch + 1, train_accuracy, train_loss_contrast.item(), train_loss_clf.item(),
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
