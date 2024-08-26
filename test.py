import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn

with open("./scGPT_vocab.json", "r", encoding="utf-8") as f:
    content = json.load(f)
genes_scGPT = sorted(content, key=content.get)
# genes_scGPT = np.array(genes_scGPT)
# print(len(genes_scGPT))
#
embedding_layer = nn.Embedding(36574, 512)
pretrained_model_path = './scGPT_best_model.pt'
pretrained_model = torch.load(pretrained_model_path)
pretrained_embedding_weights = pretrained_model['encoder.embedding.weight']
# print(pretrained_embedding_weights.shape)
#
scGPT_embed = pd.DataFrame(np.array(pretrained_embedding_weights.cpu()),index = genes_scGPT,columns=['dim'+str(i+1) for i in range(512)])
scGPT_embed.rename_axis('Gene',inplace=True)
# print(scGPT_embed)
scGPT_embed.to_csv('scGPT_gene_embed.csv')
