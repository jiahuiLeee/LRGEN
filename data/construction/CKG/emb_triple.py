from transformers import BertModel, BertTokenizer, BertForPreTraining
import torch
import os
import pandas as pd
import json
import numpy as np

# 步骤1: 加载预训练模型和分词器
model_name = '/home/ljh/hf-models/google-bert/bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
# model = BertForPreTraining.from_pretrained(model_name)

# 将模型移动到 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# print(torch.cuda.is_available())
# print(device)
# exit()
# 读取 JSON 文件
with open('holder_description.json', 'r', encoding='utf-8') as file:
    holder_descript = json.load(file)

with open('manager_description.json', 'r', encoding='utf-8') as file:
    manager_descript = json.load(file)
    
with open('mainbz_description.json', 'r', encoding='utf-8') as file:
    mainbz_descript = json.load(file)

with open('company_description.json', 'r', encoding='utf-8') as file:
    company_descript = json.load(file)    

with open('relations_description.json', 'r', encoding='utf-8') as file:
    rel_descript = json.load(file)

length = []
codes = []
# year = "2023"
with open ("/home/ljh/RGEN_A/original_data/common_selected_ts_codes.txt", "r") as f:
    for line in f:
        codes.append(line.strip())
for year in ['2024']:
    for code in codes:
        cls_embeddings = []
        with open(f'triple_select/{year}/{code}.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                r_descript = rel_descript.get(r, "")
                if r == "参股":
                    h_descript = holder_descript.get(h, "")
                    t_descript = company_descript.get(t, "")
                elif r == "管理":
                    h_descript = manager_descript.get(h, "")
                    t_descript = company_descript.get(t, "")
                elif r == "主营业务":
                    h_descript = company_descript.get(h, "")
                    t_descript = mainbz_descript.get(t, "")
                else:
                    print(f"Error relation {r}")
                    exit(0)
                # 将描述与实体组合
                input_text = f"[CLS] {h} {h_descript} {r} {r_descript} {t} {t_descript} [SEP]"
                # Tokenize 并转为张量
                encoding = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
                input_ids = encoding['input_ids'].to(device)
                # print(f"input_text len:{len(input_text)}\ninput_ids:{input_ids.shape}")
                attention_mask = encoding['attention_mask'].to(device)
                # 输入模型，获取输出
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # 提取 [CLS] 的嵌入
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # 取第一个位置 ([CLS]) 的向量
                # 将每个 [CLS] 嵌入添加到列表中
                cls_embeddings.append(cls_embedding.cpu())
                # print(f"Entity pair ({h}, {t}) and relation {r}: CLS embedding shape: {cls_embedding.shape}")
        # 将嵌入矩阵合并成一个大矩阵
        cls_embeddings = np.vstack(cls_embeddings)
        # 如果 cls_embeddings.shape[0] 小于 80，则使用 0 填充到 80
        if cls_embeddings.shape[0] < 80:
            padding = np.zeros((80 - cls_embeddings.shape[0], cls_embeddings.shape[1]))
            cls_embeddings = np.vstack((cls_embeddings, padding))
        # length.append(cls_embeddings.shape[0])
        # 保存结果到 .npy 文件
        print(cls_embeddings.shape)
        save_space = f"fina_emb/{year}"
        os.makedirs(save_space, exist_ok=True)
        np.save(f"{save_space}/{code}.npy", cls_embeddings)
        print(f"Embeddings for file {code} saved to {save_space}/{code}.npy")

# # 将列表中的整数保存到文件
# with open('fina_emb_length_statistic.txt', 'w') as file:
#     for number in length:
#         file.write(f"{number}\n")

# import statistics
# print("最小长度:", min(length))
# print("最大长度:", max(length))
# print("平均长度:", statistics.mean(length))