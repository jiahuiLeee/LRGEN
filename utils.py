import os
import time
import pickle
import numpy as np
import torch
import pandas as pd
import dgl
from scipy.sparse import linalg
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.thgnn.dataset import ourGraphClassificationDataset

def plot_metrics(train_loss1, test_loss1, train_loss2, test_loss2, train_acc, test_acc, log_dir):
    epochs = range(1, len(train_loss1) + 1)
    
    # Plot Loss1
    plt.figure()
    plt.plot(epochs, train_loss1, 'b', label='Train Loss1')
    plt.plot(epochs, test_loss1, 'r', label='Test Loss1')
    plt.title('Train and Test Loss1')
    plt.xlabel('Epochs')
    plt.ylabel('Loss1')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss1.png'))
    plt.close()
    
    # Plot Loss2
    plt.figure()
    plt.plot(epochs, train_loss2, 'b', label='Train Loss2')
    plt.plot(epochs, test_loss2, 'r', label='Test Loss2')
    plt.title('Train and Test Loss2')
    plt.xlabel('Epochs')
    plt.ylabel('Loss2')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss2.png'))
    plt.close()
    
    # Plot Accuracy
    plt.figure()
    plt.plot(epochs, train_acc, 'b', label='Train Accuracy')
    plt.plot(epochs, test_acc, 'r', label='Test Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'accuracy.png'))
    plt.close()

# 读取三元组数据
class NpyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

        # 构建样本列表
        for fname in sorted(os.listdir(root_dir)):
            if fname.endswith('.npy'):
                self.samples.append(os.path.join(root_dir, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        # 将 NumPy 数组转换为 PyTorch 张量
        data = torch.from_numpy(data).float()
        return data  # 仅返回数据，无标签
    
# 读取 relation graph 原始数据
def load_graph_data_from_file(filename="graph_data.pkl"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# 将原始数据转换为同质图
# def build_homo_graph(data, num_nodes=4112):
#     # 从原始数据中提取节点和边信息
#     nodes = data["nodes"]
#     node_features = data["node_features"]
#     edge_index = data["edge_index"]
#     edge_type = data["edge_type"]
#     labels = data["labels"]
    
#     # 节点特征（这里只有一种节点类型：company）
#     node_feats = {
#         'company': torch.tensor(node_features['company'], dtype=torch.float)
#     }
    
#     # 合并所有关系类型的边
#     all_edges = []
#     for etype in edge_index['company'].values():
#         all_edges.extend(etype)  # 将所有关系的边连接起来
    
#     # 将边列表转换为张量格式
#     src_nodes = torch.tensor([edge[0] for edge in all_edges], dtype=torch.long)
#     dst_nodes = torch.tensor([edge[1] for edge in all_edges], dtype=torch.long)
    
#     # 创建同质图
#     graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
    
#     # 添加节点特征
#     graph.ndata['feat'] = node_feats['company']
    
#     # 添加节点标签
#     graph.ndata['label'] = labels

#     # 1. 添加自环给没有边的节点
#     # 找出所有没有边的节点
#     # all_nodes = set(range(num_nodes))
#     # connected_nodes = set(src_nodes.tolist()) | set(dst_nodes.tolist())
#     # isolated_nodes = list(all_nodes - connected_nodes)
    
#     # 对没有边的节点添加自环
#     # src_nodes_self_loop = torch.tensor(isolated_nodes, dtype=torch.long)
#     # dst_nodes_self_loop = torch.tensor(isolated_nodes, dtype=torch.long)
    
#     # 将自环的边加入到图中
#     # graph.add_edges(src_nodes_self_loop, dst_nodes_self_loop)

#     return graph

def build_homo_graph(data):
    """
    将异质图数据（来自 txt 构建的 pkl）转换为同质图 DGLGraph。
    会将所有边类型统一为一类，用于同质图学习。
    """
    node_features = data["node_features"]               # shape: [N, F]
    edge_index = data["edge_index"]                     # dict: (src_type, rel, dst_type) -> tensor(2, E)
    labels = data["labels"]                             # shape: [N]

    # 合并所有异质边为统一边
    all_src = []
    all_dst = []

    for key, edge_tensor in edge_index.items():
        src, dst = edge_tensor[0], edge_tensor[1]
        all_src.append(src)
        all_dst.append(dst)

    src_nodes = torch.cat(all_src)
    dst_nodes = torch.cat(all_dst)

    num_nodes = node_features.shape[0]

    # 构建同质图
    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)

    # 添加节点特征和标签
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = labels

    return graph

# 将 relation graph 原始数据转换为 DGL 异质图的函数
# def build_dgl_hetero_graph(data):
#     # 从数据中提取节点、边、特征
#     nodes = data["node_list"]
#     node_features = data["node_features"]
#     edge_index = data["edge_index"]
#     edge_type = data["edge_type"]
#     relation_mapping = data["relation_mapping"]
#     labels = data["labels"]
    
#     # 节点类型映射（假设这里只有一个节点类型：company）
#     node_types = ['company']
    
#     # 创建节点数据
#     node_feats = {
#         'company': torch.tensor(node_features, dtype=torch.float)
#     }
    
#     # 创建边数据
#     src_dst = {}
#     # edges_type_dict = {}
    
    
#     for etype, etype_id in relation_mapping.items():
#         src_nodes = []
#         dst_nodes = []
#         edge_types = []
#         for edge in edge_index['company'][etype]:
#             src_nodes.append(edge[0])
#             dst_nodes.append(edge[1])
#             edge_types.append(etype_id)
#         src_dst[etype] = [src_nodes, dst_nodes]
#         # edges_type_dict[etype] = edge_types
    
#     # 将所有数据转换为DGL的异质图格式
#     graph = dgl.heterograph({
#         ('company', etype, 'company'): (torch.tensor(src_dst[etype][0]), torch.tensor(src_dst[etype][1]))
#         for etype in relation_mapping.keys()
#     }, num_nodes_dict={'company':4112})
    
#     # 添加节点特征
#     # graph.nodes['company'].data['feat'] = node_feats['company']
#     graph.nodes['company'].data['label'] = labels
    
#     return graph
import torch
import dgl

def build_dgl_hetero_graph(data):
    """
    将从txt构建的graph_data.pkl转换为DGL异质图
    """
    # 读取数据
    node_list = data["node_list"]                     # 股票代码列表
    node_features = data["node_features"]             # torch.Tensor [num_nodes, feat_dim]
    edge_index = data["edge_index"]                   # dict: (src_type, relation, dst_type) → tensor([2, num_edges])
    relation_mapping = data["relation_mapping"]       # dict: 中文关系 → 数字
    labels = data["labels"]                           # torch.Tensor [num_nodes]

    edge_data = {}
    for etype, adj in edge_index.items():
        if isinstance(adj, torch.Tensor) and adj.shape[0] == 2:
            src, dst = adj[0], adj[1]
            edge_data[etype] = (src, dst)
        else:
            raise ValueError(f"边索引格式错误: {etype}")
        
    # 创建异构图结构
    graph = dgl.heterograph(edge_data, num_nodes_dict={"company": len(node_list)})

    # 添加节点特征和标签
    graph.nodes["company"].data["feat"] = node_features
    graph.nodes["company"].data["label"] = labels

    return graph
           

def read_financial_index(file_path):
    # 打开并读取 JSON 文件
    data = pd.read_csv(file_path)
    
    # 提取所有的 1D 列表并转换为 tensor
    tensor_data = []
    for index, row in data.iterrows():
        tensor_data.append(torch.tensor(list(row.iloc[3:]), dtype=torch.float32))  # 转换为浮点型 tensor
    
    # 将所有行的 tensor 合并成一个大的 tensor
    tensor = torch.stack(tensor_data)
    
    return tensor

def read_labels(file_path):
    risk_label_series = pd.read_csv(file_path)['risk_label']
    return torch.tensor(risk_label_series.values, dtype=torch.long)


def batcher():
    def batcher_dev(batch):
        graph_q, graph_k, graph_idx = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k, graph_idx

    return batcher_dev

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.graphs, _ = dgl.data.utils.load_graphs(
        dataset.dgl_graphs_file, dataset.jobs[worker_id]
    )
    dataset.length = sum([g.number_of_nodes() for g in dataset.graphs])
    np.random.seed(worker_info.seed % (2 ** 32))
    
def build_dataset(
    path_X,
    path_Y,
    path_tribes,
    path_tribes_order,
    g_path):
    # prepare node attributes (financial statements) and labels (company financial risks) .
    data_mat = np.load(path_X)
    label_mat = np.load(path_Y)
    if len(label_mat.shape) == 2:
        lbl_dim=label_mat.shape[1]
    else:
        lbl_dim = 1
    print('label_dim:', lbl_dim)

    # prepare tribe sub-graph dataloader
    print('preparing subgraph dataloader...')
    t0 = time.time()
    local_graph_dataset = ourGraphClassificationDataset(
            graph_file_dir=path_tribes,
            rw_hops=30,
            subgraph_size=256,
            restart_prob=0.8,
            positional_embedding_size=32,
            entire_graph=True,
            order_graph_files_path=path_tribes_order,
        )
    local_graph_loader = torch.utils.data.DataLoader(
            dataset=local_graph_dataset,
            batch_size=4112,
            collate_fn=batcher(),
            shuffle=False,
            num_workers=0,
            worker_init_fn=worker_init_fn,
        )
    print('Subgraph dataloader completed, using {:.3f}s'.format(time.time() - t0))
    # # valid check
    # for graph_idx in local_graph_dataset.mapper_idx2name:
    #     company_name = local_graph_dataset.mapper_idx2name[graph_idx].split('.json')[0]
    #     assert graph_idx == mapper_name2node_id[company_name]

    # prepare 
    g = dgl.load_graphs(g_path)[0][0]
    return data_mat, label_mat, local_graph_loader, g

def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x

def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    n = g.number_of_nodes()
    # adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
    # adj = g.adjacency_matrix(transpose=False, scipy_fmt="csr").astype(float)
    # adj = g.adj_tensors(fmt='csr')
    adj = g.adj_external(transpose=False, scipy_fmt="csr")
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g




if __name__ == "__main__":

    tensor = read_financial_index("/home/ljh/RGEN_B/train/data/financial_indicators/2023_Q4_indicators_auto_log_scaled.csv")
    print(tensor.shape)