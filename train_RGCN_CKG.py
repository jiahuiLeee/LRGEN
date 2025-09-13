from dgl.nn import RelGraphConv
from utils import *
import dgl
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import csv
import yaml
from datetime import datetime
import shutil

def evaluate(model, graph, labels, criterion):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        out, _ = model(graph)
        loss = criterion(out, labels)
        # 计算训练准确率
        preds = torch.argmax(out, dim=1)
        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
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
    
class RGCN(nn.Module):
    def __init__(self, in_size, hidden_size, out_feats, num_relations, num_bases, n_layers):
        """_summary_

        Args:
            in_feats (int): _description_
            hidden_size (int): _description_
            out_feats (int): _description_
            relation_mapping (_type_): _description_
            n_layers (int): _description_
        """
        super(RGCN, self).__init__()
        # self.input_layer = nn.Sequential(
        #     nn.Linear(in_size, hidden_size),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(hidden_size)
        #     )
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                RelGraphConv(
                    in_feat=in_size, 
                    out_feat=hidden_size, 
                    num_rels=num_relations, 
                    regularizer='basis', 
                    num_bases=num_bases,
                    activation=torch.relu,
                    )
            )
        self.out_layer = nn.Linear(hidden_size, out_feats)


    def forward(self, graph):
        graph = dgl.to_homogeneous(graph, ndata=['feat'])
        x = graph.ndata['feat']
        etypes = graph.edata[dgl.ETYPE]
        # x = self.input_layer(x).relu_()
        for conv in self.convs:
            x = conv(graph, x, etypes)
        out = self.out_layer(x)
        return out, x
    
def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        # 创建当前时间戳文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"./logs/RGCN_mean_{timestamp}"
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
    train_graph = build_dgl_hetero_graph(rel_graph_train_data).to(device)
    test_graph = build_dgl_hetero_graph(rel_graph_test_data).to(device)
    
    train_labels = read_labels(config['train_label']).to(device)
    test_labels = read_labels(config['test_label']).to(device)

    train_fin_index_emb = read_financial_index(config['train_financial_index']).to(device)
    test_fin_index_emb = read_financial_index(config['test_financial_index']).to(device)
    
    # 拼接train_emb和train_fin_index_emb
    train_feature = torch.cat([train_ckg_emb, train_fin_index_emb], dim=1)
    test_feature = torch.cat([test_ckg_emb, test_fin_index_emb], dim=1)
    
    train_graph.nodes['company'].data['feat'] = train_feature
    test_graph.nodes['company'].data['feat'] = test_feature
    
    model = RGCN(
        in_size=config['RGCN']['mean_ckg_insize'],
        hidden_size=config['RGCN']['hidden_size'],
        out_feats=config['RGCN']['class_num'],
        num_relations=config['RGCN']['num_relations'],
        num_bases=config['RGCN']['num_bases'],
        n_layers=config['RGCN']['n_layers']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_test_accuracy = 0
    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out, _ = model(train_graph)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()
        # 计算训练准确率
        preds = torch.argmax(out, dim=1)
        train_accuracy = accuracy_score(train_labels.cpu().numpy(), preds.cpu().numpy())
        
        test_loss, test_accuracy, f1, precision, recall, auc = evaluate(
            model,
            test_graph,
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
        print(f"Train Loss: {loss}, Train Accuracy: {train_accuracy}")
        print("*****test*****")
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        print(f"Test F1: {f1}")
        print(f"Test Precision: {precision}")
        print(f"Test Recall: {recall}")
        print(f"Test AUC: {auc}")
        print("="*20)
    
    # 保存最佳模型
    torch.save(best_params, os.path.join(log_dir, "best_model.pth"))

    # 训练完成后重命名文件夹
    final_log_dir = f'{log_dir}_{best_test_accuracy:.4f}'
    os.rename(log_dir, final_log_dir)
    
    print(model)
    print(f"Best Accuracy: {best_test_accuracy}")
    print(f"Best F1: {best_f1}")
    print(f"Best Precision: {best_precision}")
    print(f"Best Recall: {best_recall}")
    print(f"Best AUC: {best_auc}")
    print("="*20)
    

if __name__ == "__main__":
    main()