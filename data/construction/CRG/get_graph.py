from neo4j import GraphDatabase
import torch
import pandas as pd
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import pickle

# 连接到 Neo4j 数据库
uri = "bolt://localhost:7687"  # Neo4j 地址
username = "neo4j"  # Neo4j 用户名
password = "gzhu@ljh12138"  # Neo4j 密码
driver = GraphDatabase.driver(uri, auth=(username, password))

# 读取图数据的函数
def read_graph_from_neo4j(driver):
    # 建立会话
    session = driver.session()

    # 读取节点信息（假设只有一种节点类型 company）
    nodes_query = "MATCH (n:company) RETURN n"
    nodes_result = session.run(nodes_query)

    nodes = {'company': []}
    node_features = {'company': []}
    
    # 读取节点
    for record in nodes_result:
        node = record['n']
        nodes['company'].append(node.element_id.split(':')[-1])
        print(node.element_id.split(':')[-1])
        # 假设每个公司节点有一个特征属性，如行业、规模等，存储为向量
        feature = node.get('feature', [0.0]*2)  # 默认特征值
        node_features['company'].append(feature)
    
    # 关系类型映射（中文到ID的映射）
    relation_mapping = {
        "供应商": 0,
        "合作": 1,
        "技术转让": 2,
        "投资": 3,
        "收购": 4
    }

    # 读取边信息，假设有五种关系类型：供应商、合作、技术转让、投资、收购
    edge_index = {'company': {etype: [] for etype in relation_mapping.keys()}}
    edge_type = {etype: [] for etype in relation_mapping.keys()}

    # 读取不同类型的边
    for etype, etype_id in relation_mapping.items():
        edges_query = f"MATCH (a:company)-[r:{etype}]->(b:company) RETURN a.CID AS start, b.CID AS end"
        edges_result = session.run(edges_query)

        for record in edges_result:
            start = record['start']
            end = record['end']
            edge_index['company'][etype].append([start, end])
            edge_type[etype].append(etype_id)

    # 返回读取到的节点、边信息
    return nodes, node_features, edge_index, edge_type, relation_mapping

# 保存图数据到文件
def save_graph_data_to_file(nodes, node_features, edge_index, edge_type, relation_mapping, labels, filename):
    data = {
        "nodes": nodes,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "relation_mapping": relation_mapping,
        "labels": labels
    }
    
    # 保存为 pickle 文件
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")


# 读取数据
nodes, node_features, edge_index, edge_type, relation_mapping = read_graph_from_neo4j(driver)

year = "2023"
# 读取 CSV 文件中的标签
labels_df = pd.read_csv(f'/home/ljh/RGEN_A/data/label/{year}_3_label.csv')
# 创建一个字典将公司 ID 映射到标签
labels_dict = dict(zip(labels_df.iloc[:,0], labels_df['risk_label']))
# 假设 nodes['company'] 存储了公司节点的 ID（从 Neo4j 中读取的）
company_ids = nodes['company']  # Neo4j 中读取的公司节点 ID 列表
# 根据从 CSV 中读取的标签，为每个公司节点分配标签
labels = torch.tensor([labels_dict.get(int(company_id), -1) for company_id in company_ids], dtype=torch.long)

# 保存图数据
# 训练：2023_graph_data.pkl
# 测试：2024_graph_data.pkl
save_graph_data_to_file(nodes, node_features, edge_index, edge_type, relation_mapping, labels, filename=f"{year}_graph_data.pkl")