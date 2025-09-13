import pickle
import dgl
import torch

# 读取 .pkl 文件
def load_graph_data_from_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# 将异构图转换为同质图
def convert_to_homogeneous_graph(data):
    # 提取节点和特征
    nodes = data['nodes']['company']
    node_features = torch.tensor(data['node_features']['company'], dtype=torch.float32)

    # 提取边
    edge_index = data['edge_index']['company']
    edge_type = data['edge_type']

    # 合并所有类型的边
    src_nodes = []
    dst_nodes = []
    edge_types = []

    for etype, edges in edge_index.items():
        for edge in edges:
            src_nodes.append(edge[0])
            dst_nodes.append(edge[1])
            edge_types.append(edge_type[etype][0])  # 假设每种关系类型只有一个 ID

    # 将节点 ID 统一转换为字符串
    nodes = [str(node_id) for node_id in nodes]
    src_nodes = [str(node_id) for node_id in src_nodes]
    dst_nodes = [str(node_id) for node_id in dst_nodes]

    # 将节点 ID 映射为连续整数
    node_id_map = {node_id: i for i, node_id in enumerate(nodes)}
    src_nodes = [node_id_map[node_id] for node_id in src_nodes]
    dst_nodes = [node_id_map[node_id] for node_id in dst_nodes]

    # 构建同质图
    g = dgl.graph((src_nodes, dst_nodes), num_nodes=len(nodes))
    g.ndata['feat'] = node_features  # 添加节点特征
    g.edata['etype'] = torch.tensor(edge_types, dtype=torch.long)  # 添加边类型

    return g

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

# 保存为 .bin 文件
def save_graph_to_bin_file(graph, filename):
    dgl.save_graphs(filename, [graph])
    print(f"Graph saved to {filename}")

# 主函数
def main():
    # 读取 .pkl 文件
    year = "2023"
    data = load_graph_data_from_file(f"/home/ljh/RGEN_B/train/data/CRG/graph_2024Q2.pkl")
    save_path = f"/home/ljh/RGEN_B/train/data/CRG/graph_2024Q2.bin"
    # 转换为同质图
    homogeneous_graph = build_homo_graph(data)

    # 保存为 .bin 文件
    save_graph_to_bin_file(homogeneous_graph, save_path)
    
    g = dgl.load_graphs(save_path)[0][0]
    print(g)

if __name__ == "__main__":
    main()