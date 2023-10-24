"""
    load raw_data user_id item_id score
    train irt and save theta ,alpha, beta
    
"""
import json
import torch
from tqdm import tqdm
import numpy as np
from CAT.distillation.MFI.model import distillModel 
from CAT.distillation.tool import get_label_and_k, split_data, transform
import os
dataset='eedi'
cdm='irt'
stg='MFI'
path_prefix = os.path.abspath('.')
trait = json.load(open(f'{path_prefix}/data/{dataset}/{stg}/trait.json', 'r'))
utrait = trait['user']
itrait = trait['item']
label = trait['label']
k_info = trait['k_info']
torch.manual_seed(0)

k=50
embedding_dim=15
dMFI = distillModel(k,embedding_dim,1,device='cuda:1')
postfix='11'
dMFI.load(f'{path_prefix}/ckpt/{dataset}/{cdm}_{stg}_ip{postfix}.pt')
# ball_embs=[]
ball_trait = {}
for k,v in tqdm(itrait.items()):
    i_emb = dMFI.model.itn(torch.tensor(v).to('cuda:1')).tolist()
    ball_trait[int(k)]=(-3,i_emb)
    # ball_embs.append(i_emb)

with open(f"{path_prefix}/data/{dataset}/{stg}/ball_trait{postfix}.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(ball_trait, ensure_ascii=False))
    

# i_label={}
# # for i in itrait.keys():
# #     i_label[int(i)]=[]
# user_kinfo = [(theta,top_k) for (_, theta), top_k in zip(utrait.items(), k_info)]
# idx = [int(len(user_kinfo)/8*(i+0.5)) for i in range(8)]
# [user_kinfo[i] for i in idx]
# user_kinfo.sort()
# label=[sum(i)/len(i)  if len(i)!=0 else -3 for i in i_label.values()]    
# with open(f"{path_prefix}item_label.json", "w", encoding="utf-8") as f:
#     f.write(json.dumps(label, ensure_ascii=False))