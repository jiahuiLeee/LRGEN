from dgl.nn import RelGraphConv
from utils import *
import yaml
import dgl
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math

import csv
from datetime import datetime
import shutil

class HGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(HGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.use_RTE       = use_RTE
        self.att           = None
        
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        '''
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
        '''
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        
        if self.use_RTE:
            self.emb            = RelTemporalEncoding(in_dim)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time = edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        '''
            j: source, i: target; <j, i>
        '''
        data_size = edge_index_i.size(0)
        '''
            Create Attention and Message tensor beforehand.
        '''
        res_att     = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg     = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type] 
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    '''
                        idx is all the edges with meta relation <source_type, relation_type, target_type>
                    '''
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    '''
                        Get the corresponding input node representations by idx.
                        Add tempotal encoding to source representation (j)
                    '''
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    if self.use_RTE:
                        source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    '''
                        Step 1: Heterogeneous Mutual Attention
                    '''
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att[relation_type]).transpose(1,0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    '''
                        Step 2: Heterogeneous Message Passing
                    '''
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1,0), self.relation_msg[relation_type]).transpose(1,0)   
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)


    def update(self, aggr_out, node_inp, node_type):
        '''
            Step 3: Target-specific Aggregation
            x = W[node_type] * gelu(Agg(x)) + x
        '''
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            '''
                Add skip connection with learnable weight self.skip[t_id]
            '''
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
    
    
    
class DenseHGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(DenseHGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.use_RTE       = use_RTE
        self.att           = None
        
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()

        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        '''
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
        '''
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.drop           = nn.Dropout(dropout)
        
        if self.use_RTE:
            self.emb            = RelTemporalEncoding(in_dim)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)
        
        
        self.mid_linear  = nn.Linear(out_dim,  out_dim * 2)
        self.out_linear  = nn.Linear(out_dim * 2,  out_dim)
        self.out_norm    = nn.LayerNorm(out_dim)
        
    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time = edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        '''
            j: source, i: target; <j, i>
        '''
        data_size = edge_index_i.size(0)
        '''
            Create Attention and Message tensor beforehand.
        '''
        res_att     = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg     = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type] 
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    '''
                        idx is all the edges with meta relation <source_type, relation_type, target_type>
                    '''
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    '''
                        Get the corresponding input node representations by idx.
                        Add tempotal encoding to source representation (j)
                    '''
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    if self.use_RTE:
                        source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    '''
                        Step 1: Heterogeneous Mutual Attention
                    '''
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att[relation_type]).transpose(1,0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    '''
                        Step 2: Heterogeneous Message Passing
                    '''
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1,0), self.relation_msg[relation_type]).transpose(1,0)   
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)


    def update(self, aggr_out, node_inp, node_type):
        '''
            Step 3: Target-specific Aggregation
            x = W[node_type] * Agg(x) + x
        '''
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx])) + node_inp[idx]
            '''
                Add skip connection with learnable weight self.skip[t_id]
            '''
            if self.use_norm:
                trans_out = self.norms[target_type](trans_out)
                
            '''
                Step 4: Shared Dense Layer
                x = Out_L(gelu(Mid_L(x))) + x
            '''
                
            trans_out     = self.drop(self.out_linear(F.gelu(self.mid_linear(trans_out)))) + trans_out
            res[idx]      = self.out_norm(trans_out)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 240, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
    def forward(self, x, t):
        return x + self.lin(self.emb(t))
    
    
    
class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm = True, use_RTE = True):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'hgt':
            self.base_conv = HGTConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm, use_RTE)
        elif self.conv_name == 'dense_hgt':
            self.base_conv = DenseHGTConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm, use_RTE)
        elif self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)
    def forward(self, meta_xs, node_type, edge_index, edge_type, edge_time):
        if self.conv_name == 'hgt':
            return self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_time)
        elif self.conv_name == 'gcn':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'gat':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'dense_hgt':
            return self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_time)
    

class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid    = n_hid
        self.n_out    = n_out
        self.linear   = nn.Linear(n_hid,  n_out)
    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)
    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)

class Matcher(nn.Module):
    '''
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    '''
    def __init__(self, n_hid):
        super(Matcher, self).__init__()
        self.left_linear    = nn.Linear(n_hid,  n_hid)
        self.right_linear   = nn.Linear(n_hid,  n_hid)
        self.sqrt_hd  = math.sqrt(n_hid)
        self.cache      = None
    def forward(self, x, y, infer = False, pair = False):
        ty = self.right_linear(y)
        if infer:
            '''
                During testing, we will consider millions or even billions of nodes as candidates (x).
                It's not possible to calculate them again for different query (y)
                Since the model is fixed, we propose to cache them, and dirrectly use the results.
            '''
            if self.cache != None:
                tx = self.cache
            else:
                tx = self.left_linear(x)
                self.cache = tx
        else:
            tx = self.left_linear(x)
        if pair:
            res = (tx * ty).sum(dim=-1)
        else:
            res = torch.matmul(tx, ty.transpose(0,1))
        return res / self.sqrt_hd
    def __repr__(self):
        return '{}(n_hid={})'.format(
            self.__class__.__name__, self.n_hid)
    

class GNN(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout = 0.2, conv_name = 'hgt', prev_norm = True, last_norm = True, use_RTE = True):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.adapt_ws  = nn.ModuleList()
        self.drop      = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = prev_norm, use_RTE = use_RTE))
        self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm = last_norm, use_RTE = use_RTE))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs  


def evaluate(gnn, classifier, test_graph, test_fin_index_emb, labels, criterion, device):
    test_total_edges = 0
    for etype in test_graph.etypes:
        test_total_edges += test_graph.number_of_edges(etype)
    node_type = torch.LongTensor([0]*4112).to(device)
    test_edge_time = torch.LongTensor([1]*test_total_edges).to(device)
    edge_index = torch.stack(dgl.to_homogeneous(test_graph).edges()).long().to(device)
    gnn.eval()
    classifier.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        node_rep = gnn.forward(test_fin_index_emb, node_type, test_edge_time, edge_index, test_graph.etypes)
        res  = classifier.forward(node_rep)
        loss = criterion(res, labels)
        # 计算训练准确率
        preds = torch.argmax(res, dim=1)
        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        # 计算 AUC
        # 获取预测概率
        probs = F.softmax(res, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
        
        # 如果是二分类（只需要考虑正类的概率）
        if probs.shape[1] == 2:
            auc = roc_auc_score(labels.cpu().numpy(), probs[:, 1])  # 只需要第二类的概率
        else:
            # 多分类（使用 One-vs-Rest 方式）
            auc = roc_auc_score(labels.cpu().numpy(), probs, multi_class='ovr')
        
        return loss, accuracy, f1, precision, recall, auc
    
    
def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        # 创建当前时间戳文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"./logs/HGT_mean_{timestamp}"
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
    # print(len(train_graph.edges))
    train_total_edges = 0
    for etype in train_graph.etypes:
        train_total_edges += train_graph.number_of_edges(etype)

    # print(total_edges)
    # exit()
    train_labels = read_labels(config['train_label']).to(device)
    test_labels = read_labels(config['test_label']).to(device)

    train_fin_index_emb = read_financial_index(config['train_financial_index']).to(device)
    test_fin_index_emb = read_financial_index(config['test_financial_index']).to(device)
    
    train_feature = torch.cat([train_ckg_emb, train_fin_index_emb], dim=1)
    test_feature = torch.cat([test_ckg_emb, test_fin_index_emb], dim=1)
    
    gnn = GNN(conv_name = 'hgt',
              in_dim = config['HGT']['mean_ckg_insize'],
              n_hid = config['HGT']['hidden_size'], 
              n_heads = 8, 
              n_layers = config['HGT']['n_layers'], 
              dropout = 0.2,
              num_types = config['HGT']['num_node_type'], 
              num_relations = config['HGT']['num_relations']
    ).to(device)
    classifier = Classifier(config['HGT']['hidden_size'], config['HGT']['class_num']).to(device)

    model = nn.Sequential(gnn, classifier)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    
    best_test_accuracy = 0
    num_epochs = config['num_epochs']
    node_type = torch.LongTensor([0]*4112).to(device)
    train_edge_time = torch.LongTensor([1]*train_total_edges).to(device)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        edge_index = torch.stack(dgl.to_homogeneous(train_graph).edges()).long().to(device)

        node_rep = gnn.forward(train_feature, node_type, train_edge_time, edge_index, train_graph.etypes)
        res  = classifier.forward(node_rep)
        
        loss = criterion(res, train_labels)
        loss.backward()
        optimizer.step()
        # 计算训练准确率
        preds = torch.argmax(res, dim=1)
        train_accuracy = accuracy_score(train_labels.cpu().numpy(), preds.cpu().numpy())
        
        test_loss, test_accuracy, test_f1, test_precision, test_recall, test_auc = evaluate(
            gnn,
            classifier,
            test_graph,
            test_feature,
            test_labels,
            criterion,
            device
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
            best_f1 = test_f1
            best_precision = test_precision
            best_recall = test_recall
            best_auc = test_auc
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
        print(f"Test F1: {test_f1}")
        print(f"Test Precision: {test_precision}")
        print(f"Test Recall: {test_recall}")
        print(f"Test AUC: {test_auc}")
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