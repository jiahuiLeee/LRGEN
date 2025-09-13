from neo4j import GraphDatabase
import json
import os

# 连接到 Neo4j 数据库
uri = "bolt://localhost:7687"  # 修改为您的 Neo4j URI
username = "neo4j"             # 修改为您的用户名
password = "gzhu@ljh12138"     # 修改为您的密码
driver = GraphDatabase.driver(uri, auth=(username, password))

year = "2024"
# 查询 1 到 2 跳路径
query = """
MATCH path = (start:company)-[*1..2]-(end)
WHERE start.stock_code = $code
RETURN path
"""
codes = []
with open('/home/ljh/RGEN_A/original_data/common_selected_ts_codes.txt', 'r') as f:
    for line in f:
        codes.append(line.strip())
        
for code in codes:
    # 打开数据库会话
    with driver.session() as session:
        # 执行查询
        results = session.run(query, code=code)
        
        # 打印完整的查询结果
        print("Full results object:")
        print(results)
        
        triples = []  # 用于存储三元组
        
        # 遍历查询结果
        for record in results:
            # 打印每条记录
            print("Record:")
            print(record)
            path = record["path"]
            # 遍历路径中的关系
            for rel in path.relationships:
                # print(rel)
                print(rel.nodes[0])
                print(rel.nodes[1])
                print(rel.type)
                print('----------------')
                # 构造三元组
                # start_node = rel.nodes[0].properties['full_name']
                # end_node = rel.nodes[1].properties['full_name'] if 'full_name' in rel.nodes[1].properties else rel.nodes[1].properties['mainbz']
                # 获取头实体和尾实体属性
                start_node = rel.start_node.get("full_name", rel.start_node.get("CID", "Unknown"))
                end_node = rel.end_node.get("full_name", rel.end_node.get("bz_item", "Unknown"))
                triple = (
                    start_node,  # 头实体
                    rel.type,              # 关系
                    end_node    # 尾实体
                )
                triples.append(triple)

    # 保存结果为 TXT 文件
    os.makedirs(f"triple/{year}", exist_ok=True)
    output_file = f"triple/{year}/{code}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for triple in triples:
            f.write("\t".join(triple) + "\n")

    print(f"三元组已保存到 {output_file}")
