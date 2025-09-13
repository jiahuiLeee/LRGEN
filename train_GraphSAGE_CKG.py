import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from utils import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import csv
import yaml
from datetime import datetime
import shutil

def evaluate(model, graph, fin_data, labels, criterion):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        edge_index = torch.stack(graph.edges()).long().to('cuda')
        out, _ = model(fin_data, edge_index)
        loss = criterion(out, labels)
        # 计算训练准确率
        preds = torch.argmax(out, dim=1)
        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        # 计算 AUC
        # 获取预测概率
        probs = F.softmax(out, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
        
        # 如果是二分类（只需要考虑正类的概率）
        if probs.shape[1] == 2:
            auc = roc_auc_score(labels.cpu().numpy(), probs[:, 1])  # 只需要第二类的概率
        else:
            # 多分类（使用 One-vs-Rest 方式）
            auc = roc_auc_score(labels.cpu().numpy(), probs, multi_class='ovr')
        
        return loss, accuracy, f1, precision, recall, auc
    
# 定义模型
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1):
        super(GraphSAGEModel, self).__init__()
        self.sage = GraphSAGE(in_channels, hidden_channels, num_layers, hidden_channels)
        self.out_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.sage(x, edge_index)
        out = self.out_layer(x)
        return out, x


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        # 创建当前时间戳文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"./logs/GraphSAGE_mean_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    # 复制配置文件到日志文件夹
    shutil.copy('config.yaml', log_dir)
    # 创建CSV文件来保存训练日志
    log_file = os.path.join(log_dir, 'training_log.csv')
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'epoch', 
            'train_accuracy', 'train_loss',
            'test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_auc',
            'best_epoch',
        ])  # CSV表头

        best_test_accuracy = 0
        best_epoch = 0
        best_params = None
        # 训练循环
        emb_list = []
        # plt
        train_loss1 = []
        test_loss1 = []
        train_loss2 = []
        test_loss2 = []
        train_acc = []
        test_acc = []
        
        # 设置设备
        device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"device:{device}")
        
        train_ckg_emb = torch.load(config['train_embedding_mean']).to(device)
        test_ckg_emb = torch.load(config['test_embedding_mean']).to(device)
        
        # 读取训练数据和测试数据
        rel_graph_train_data = load_graph_data_from_file(config['train_graph_data'])
        rel_graph_test_data = load_graph_data_from_file(config['test_graph_data'])

        # 构建训练集和测试集的异质图
        train_graph = build_homo_graph(rel_graph_train_data).to(device)
        test_graph = build_homo_graph(rel_graph_test_data).to(device)
        
        train_labels = read_labels(config['train_label']).to(device)
        test_labels = read_labels(config['test_label']).to(device)

        train_fin_index_emb = read_financial_index(config['train_financial_index']).to(device)
        test_fin_index_emb = read_financial_index(config['test_financial_index']).to(device)
        
        # 拼接train_emb和train_fin_index_emb
        train_feature = torch.cat([train_ckg_emb, train_fin_index_emb], dim=1)
        test_feature = torch.cat([test_ckg_emb, test_fin_index_emb], dim=1)
        # train_graph.nodes['company'].data['feat'] = train_fin_index_emb
        # test_graph.nodes['company'].data['feat'] = test_fin_index_emb
        
        # 初始化模型
        model = GraphSAGEModel(
            in_channels=config['GraphSAGE']['mean_ckg_insize'], 
            hidden_channels=config['GraphSAGE']['hidden_size'], 
            out_channels=config['GraphSAGE']['class_num'],
            num_layers=config['GraphSAGE']['n_layers']
            ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        best_test_accuracy = 0
        num_epochs = config['num_epochs']
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            edge_index = torch.stack(train_graph.edges()).long().to(device)
            out, _ = model(train_feature, edge_index)
            
            loss = criterion(out, train_labels)
            loss.backward()
            optimizer.step()
            # 计算训练准确率
            preds = torch.argmax(out, dim=1)
            train_accuracy = accuracy_score(train_labels.cpu().numpy(), preds.cpu().numpy())
            
            test_loss, test_accuracy, f1, precision, recall, auc = evaluate(
                model,
                test_graph,
                test_feature,
                test_labels,
                criterion
                )
            
            train_loss1.append(loss.detach().cpu().numpy())
            train_loss2.append(0)
            test_loss1.append(test_loss.detach().cpu().numpy())
            test_loss2.append(0)
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
            print(f"Train Loss1: {loss:.4f}, Train Accuracy: {train_accuracy}")
            print("*****test*****")
            print(f"Test Loss1: {test_loss:.4f}, Test Accuracy: {test_accuracy}")
            print(f"Test F1: {f1}")
            print(f"Test Precision: {precision}")
            print(f"Test Recall: {recall}")
            print(f"Test AUC: {auc}")
            print("="*20)

            # 保存训练数据到CSV
            writer.writerow([
                epoch + 1, train_accuracy, loss.item(),
                test_accuracy, f1, precision, recall, auc,
                best_epoch
            ])

        
    # 保存最佳模型
    torch.save(best_params, os.path.join(log_dir, "best_model.pth"))

    # 训练完成后重命名文件夹
    final_log_dir = f'{log_dir}_{best_test_accuracy:.4f}'
    os.rename(log_dir, final_log_dir)
    
    print(f"Best Accuracy: {best_test_accuracy}")
    print(f"Best F1: {best_f1}")
    print(f"Best Precision: {best_precision}")
    print(f"Best Recall: {best_recall}")
    print(f"Best AUC: {best_auc}")
    print("="*20)

    
    
if __name__ == "__main__":
    main()