import torch.nn as nn
import torch
import dgl
from models.RGCN import RGCN
from models.ShuffleUint import TripleFusionNet

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

class LRGEN(nn.Module):
    def __init__(self, 
                 group, drop_ratio, in_channels, hidden_channels, shufunit_out_channel, out_kg_dim,
                 hidden_dim, num_types, num_relations, n_heads, n_layers,
                 fin_dim, 
                 class_dim, n_class):
        super(LRGEN, self).__init__()
        # stage1: fuse triple
        self.fuse_triple_layer = nn.Sequential(
            TripleFusionNet(
                groups=group,
                drop_ratio=drop_ratio,
                in_channels=in_channels,
                hidden_channel=hidden_channels,
                shufunit_out_channel=shufunit_out_channel,
                out_channels=out_kg_dim),
        )
        # stage2: fuse neighbors 
        self.fuse_neighbors_layer = GNN(conv_name = 'hgt',
                in_dim = out_kg_dim+fin_dim,
                n_hid = hidden_dim, 
                n_heads = n_heads, 
                n_layers = n_layers, 
                dropout = 0.2,
                num_types = num_types, 
                num_relations = num_relations
        )
        # stage3: fuse financial indexs
        self.fuse_fin_indexs_layer = nn.Sequential(
            nn.Linear(hidden_dim, class_dim),
        )
        # stage4: classify
        self.fc = nn.Sequential(
            nn.Linear(class_dim, n_class)
        )
        # 初始化模型权重
        self.initialize_weights()
        # 中间变量
        self.emb = []
    
    def forward(self, rel_graph, subgraph_triple, fin_emb, last_batch, emb_list):
        q = self.fuse_triple_layer(subgraph_triple)
        if last_batch:
            emb_list.append(q.detach().to('cpu'))
            emb_stack = torch.cat(emb_list, dim=0).to('cuda')
            
            train_total_edges = 0
            for etype in rel_graph.etypes:
                train_total_edges += rel_graph.number_of_edges(etype)
            node_type = torch.LongTensor([0]*4112).to('cuda')
            train_edge_time = torch.LongTensor([1]*train_total_edges).to('cuda')
            edge_index = torch.stack(dgl.to_homogeneous(rel_graph).edges()).long().to('cuda')
            
            # ✅ 新逻辑：直接拼接结构嵌入 + 财务特征作为图节点特征
            node_features = torch.cat((emb_stack, fin_emb), dim=1)

            # 使用拼接后特征作为输入，走 GNN
            x = self.fuse_neighbors_layer(node_features, node_type, train_edge_time, edge_index, rel_graph.etypes)

            # 下游分类任务
            x = self.fuse_fin_indexs_layer(x)  # 不再拼接 fin_emb
            out = self.fc(x)
            return out, q,  x
        else:
            return None, q,  None
    
    def initialize_weights(self):
        # 初始化卷积层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class LRGEN_wo_CKG(nn.Module):
    def __init__(self, 
                 group, drop_ratio, in_channels, hidden_channels, shufunit_out_channel, out_kg_dim,
                 hidden_dim, num_types, num_relations, n_heads, n_layers,
                 fin_dim, 
                 class_dim, n_class):
        super(LRGEN_wo_CKG, self).__init__()
        self.fuse_neighbors_layer =GNN(conv_name = 'hgt',
                in_dim = fin_dim,
                n_hid = hidden_dim, 
                n_heads = n_heads, 
                n_layers = n_layers, 
                dropout = 0.2,
                num_types = num_types, 
                num_relations = num_relations

        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, n_class)
        )
        # 初始化模型权重
        self.initialize_weights()
    
    def forward(self, rel_graph, subgraph_triple, fin_emb, last_batch, emb_list):
        rel_graph.ndata['feat'] = fin_emb.to('cuda')
        train_total_edges = 0
        for etype in rel_graph.etypes:
            train_total_edges += rel_graph.number_of_edges(etype)
        node_type = torch.LongTensor([0]*4112).to('cuda')
        train_edge_time = torch.LongTensor([1]*train_total_edges).to('cuda')
        edge_index = torch.stack(dgl.to_homogeneous(rel_graph).edges()).long().to('cuda')
        
        # ✅ 新逻辑：直接拼接结构嵌入 + 财务特征作为图节点特征
        node_features = fin_emb
        
        x = self.fuse_neighbors_layer(node_features, node_type, train_edge_time, edge_index, rel_graph.etypes)
        out = self.fc(x)
        return out
    
    def initialize_weights(self):
        # 初始化卷积层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class LRGEN_wo_CRG(nn.Module):
    def __init__(self, 
                 group, drop_ratio, in_channels, hidden_channels, shufunit_out_channel, out_kg_dim,
                 in_rgcn_dim, out_rel_dim, num_relations, num_bases, n_layers,
                 fin_dim, 
                 class_dim, n_class):
        super(LRGEN_wo_CRG, self).__init__()
        self.fuse_triple_layer = nn.Sequential(
            TripleFusionNet(
                groups=group,
                drop_ratio=drop_ratio,
                in_channels=in_channels,
                hidden_channel=hidden_channels,
                shufunit_out_channel=shufunit_out_channel,
                out_channels=out_kg_dim),
        )
        self.fuse_fin_indexs_layer = nn.Sequential(
            nn.Linear(out_kg_dim + fin_dim, class_dim),
        )
        # stage4: classify
        self.fc = nn.Sequential(
            nn.Linear(class_dim, n_class)
        )
        # 初始化模型权重
        self.initialize_weights()
        # 中间变量
        self.emb = []
    
    def forward(self, rel_graph, subgraph_triple, fin_emb, last_batch, emb_list):
        q = self.fuse_triple_layer(subgraph_triple)
        if last_batch:
            emb_list.append(q.detach().to('cpu'))
            emb_stack = torch.cat(emb_list, dim=0).to('cuda')
            x = torch.cat((emb_stack, fin_emb), dim=1).to('cuda')
            x = self.fuse_fin_indexs_layer(x)
            out = self.fc(x)
            return out, q,  x
        else:
            return None, q,  None

    def initialize_weights(self):
        # 初始化卷积层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class LRGEN_wo_Fin(nn.Module):
    def __init__(self, 
                 group, drop_ratio, in_channels, hidden_channels, shufunit_out_channel, out_kg_dim,
                 hidden_dim, num_types, num_relations, n_heads, n_layers,
                 fin_dim, 
                 class_dim, n_class):
        super(LRGEN_wo_Fin, self).__init__()
        # stage1: fuse triple
        self.fuse_triple_layer = nn.Sequential(
            TripleFusionNet(
                groups=group,
                drop_ratio=drop_ratio,
                in_channels=in_channels,
                hidden_channel=hidden_channels,
                shufunit_out_channel=shufunit_out_channel,
                out_channels=out_kg_dim),
        )
        # stage2: fuse neighbors 
        self.fuse_neighbors_layer = GNN(conv_name = 'hgt',
                in_dim = out_kg_dim,
                n_hid = hidden_dim, 
                n_heads = n_heads, 
                n_layers = n_layers, 
                dropout = 0.2,
                num_types = num_types, 
                num_relations = num_relations

        )
        # stage3: fuse financial indexs
        self.fuse_fin_indexs_layer = nn.Sequential(
            nn.Linear(hidden_dim, class_dim),

        )
        # stage4: classify
        self.fc = nn.Sequential(
            nn.Linear(class_dim, n_class)
        )
        # 初始化模型权重
        self.initialize_weights()
        # 中间变量
        self.emb = []
    
    def forward(self, rel_graph, subgraph_triple, fin_emb, last_batch, emb_list):
        q = self.fuse_triple_layer(subgraph_triple)
        if last_batch:
            emb_list.append(q.detach().to('cpu'))
            emb_stack = torch.cat(emb_list, dim=0).to('cuda')
            
            train_total_edges = 0
            for etype in rel_graph.etypes:
                train_total_edges += rel_graph.number_of_edges(etype)
            node_type = torch.LongTensor([0]*4112).to('cuda')
            train_edge_time = torch.LongTensor([1]*train_total_edges).to('cuda')
            edge_index = torch.stack(dgl.to_homogeneous(rel_graph).edges()).long().to('cuda')
            
            # ✅ 新逻辑：直接拼接结构嵌入 + 财务特征作为图节点特征
            node_features = emb_stack

            # 使用拼接后特征作为输入，走 GNN
            x = self.fuse_neighbors_layer(node_features, node_type, train_edge_time, edge_index, rel_graph.etypes)

            # 下游分类任务
            x = self.fuse_fin_indexs_layer(x)  # 不再拼接 fin_emb
            out = self.fc(x)
            return out, q,  x
        else:
            return None, q,  None
    
    def initialize_weights(self):
        # 初始化卷积层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)