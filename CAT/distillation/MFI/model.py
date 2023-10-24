from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
    
class distill(nn.Module):
    def __init__(self,embedding_dim,user_dim):
        # self.prednet_input_len =1
        super(distill, self).__init__()
        self.utn = nn.Sequential(
            nn.Linear(user_dim, 256), nn.Sigmoid(
            ),
            nn.Linear(256, 128), nn.Sigmoid(
            ), 
            nn.Linear(128, embedding_dim),nn.Softplus())
        # nn.Dropout(p=0.5)
        
        self.itn = nn.Sequential(
            nn.Linear(2, 256), nn.Sigmoid(
            ),
            nn.Linear(256, 128), nn.Sigmoid(
            ), 
            nn.Linear(128, embedding_dim),nn.Softplus())
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def forward(self,u,i):
        user =self.utn(u)
        item =self.itn(i)
        return (user * item).sum(dim=-1, keepdim=True)
        # return user*item
    
class distillModel(object):
    def __init__(self, k, embedding_dim, user_dim, device):
        self.model = distill(embedding_dim,user_dim)
        # 20 1 1 
        self.k = k
        self.device=device
        
        
    def train_rank(self,train_data,test_data,item_pool,lr=0.01,epoch=2):
        self.model=self.model.to(self.device)
        train_data=list(train_data)
        test_data=list(test_data)
        self.eval(test_data,item_pool)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch_i in range(epoch):
            loss = []
            for data in tqdm(train_data,f'Epoch {epoch_i+1} '):
                utrait,itrait,label,k_items=data
                itrait = itrait.squeeze()
                indices = torch.tensor(k_items).to(self.device)
                u_loss: torch.Tensor  = torch.tensor(0.).to(self.device)
                utrait:torch.Tensor = utrait.to(self.device)
                itrait: torch.Tensor = itrait.to(self.device)
                label: torch.Tensor = label.to(self.device)
                kutrait = torch.index_select(utrait, 0, indices)
                kitrait = torch.index_select(itrait, 0, indices)
                # klabel = torch.index_select(label, 0, indices)
                score = self.model(kutrait,kitrait).squeeze(-1)
                r = torch.arange(1,self.k+1).to(self.device)
                a=torch.tensor(20.).to(self.device)
                score1 = torch.cat([score[1:],score[49:]],dim=0)
                # u_loss = (-torch.exp(-r/a)*torch.log(torch.sigmoid(score-score1))).sum()
                u_loss = (-torch.log(torch.sigmoid(score-score1))).sum()
                # u_loss=((score-label)**2).sum()
                loss.append(u_loss.item())
                optimizer.zero_grad()
                u_loss.backward()
                optimizer.step()
                # print(float(np.mean(loss)))
                # self.eval(valid_data,item_pool)
            print('Loss: ',float(np.mean(loss)))
            self.eval(test_data,item_pool)
    
    def train(self,train_data,test_data,item_pool,lr=0.01,epoch=2):
        self.model=self.model.to(self.device)
        train_data=list(train_data)
        test_data=list(test_data)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.save(f'/data/yutingh/CAT/ckpt/ifytek/irt_MFI_ip21.pt')
        for epoch_i in range(epoch):
            loss = []
            for data in tqdm(train_data,f'Epoch {epoch_i+1} '):
                utrait,itrait,label,_=data
                itrait = itrait.squeeze()
                u_loss: torch.Tensor  = torch.tensor(0.).to(self.device)
                utrait:torch.Tensor = utrait.to(self.device)
                itrait: torch.Tensor = itrait.to(self.device)
                label: torch.Tensor = label.to(self.device)
                score = self.model(utrait,itrait).squeeze(-1)
                u_loss=((score-label)**2).sum()
                loss.append(u_loss.item())
                optimizer.zero_grad()
                u_loss.backward()
                optimizer.step()
                # print(float(np.mean(loss)))
                # self.eval(valid_data,item_pool)
            print('Loss: ',float(np.mean(loss)))
            self.eval(test_data,item_pool)
    
    def load(self, path):
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(path), strict=False)
    
    def save(self, path):
        model_dict = self.model.state_dict()
        model_dict = {k: v for k, v in model_dict.items()
                        if 'utn' in k or 'itn' in k}
        torch.save(model_dict, path)
    
    def eval(self,valid_data,item_pool):
        self.model=self.model.to(self.device)
        k_nums=[1,3,5,10,20]
        recall = [[]for i in k_nums]
        for data in tqdm(valid_data,'testing'):
            utrait,_,__,k_info=data
            k_items,k_DCG = self.getkitems(utrait,item_pool)
            for i,k in enumerate(k_nums):
                i_kitems = set(k_items[:k]).intersection(set(k_info[:k]))
                recall[i].append(len(i_kitems)/k)
        for i,k in enumerate(k_nums):
            print(f'recall@{k}: ',np.mean(recall[i]))
    
    def getkitems(self, utrait,item_pool):
        with torch.no_grad():
            self.model.eval()
            utrait:torch.Tensor = (utrait[0].repeat(len(item_pool.keys()),1)).to(self.device)
            itrait:torch.Tensor = torch.tensor(list(item_pool.values())).to(self.device)
            scores = self.model(utrait,itrait).squeeze(-1)
            tmp = list(zip(scores.tolist(),item_pool.keys()))
            tmp_sorted = sorted(tmp, reverse=True)
            self.model.train()
        return [int(i[1]) for i in tmp_sorted[:self.k]],[e[0]/np.log(i+2) for i,e in enumerate(tmp_sorted[:self.k])]
        

    
    
    
    
        