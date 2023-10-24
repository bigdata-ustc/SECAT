import torch
from torch import Tensor
from tqdm import tqdm
import numpy as np
import vegas
from scipy import integrate
import random
import functools
from random import randint, sample
from copy import deepcopy


def split_data(utrait,label,k_fisher,rate=1,tested_info=None):
    max_len=int(rate*len(utrait.keys()))
    u_train={}
    u_test={}
    for i, (u,trait) in enumerate(utrait.items()):
        if i>=max_len:
            u_test[u]=trait
        else:
            u_train[u]=trait
    if tested_info:
        return ((u_train,label[:max_len],k_fisher[:max_len],tested_info[:max_len]),(u_test,label[max_len:],k_fisher[max_len:],tested_info[max_len:]))
    else:
        return ((u_train,label[:max_len],k_fisher[:max_len]),(u_test,label[max_len:],k_fisher[max_len:]))
    
def get_label_and_k(user_trait,item_trait,k,stg="MFI", model=None):
    labels=[]
    k_infos=[]
    tested_infos=[]
    for sid, theta in tqdm(user_trait.items(),f'get top {k} items'):
        if stg=='MFI':
            k_info, label, tested_info = get_k_fisher(k, theta, item_trait)
            tested_infos.append(tested_info)
        elif stg=='KLI':
            k_info,label,tested_info= get_k_kli(k, theta, item_trait)
            tested_infos.append(tested_info)
        elif stg=='MAAT':
            k_info,label,tested_info= get_k_emc(k, sid,theta, item_trait, model)
            tested_infos.append(tested_info)
        k_infos.append(k_info)
        labels.append(label)
    return labels,k_infos,tested_infos

def get_ncd_label_and_k(user_trait, item_trait, k, stg="MFI", model=None, data=None):
    labels=[]
    k_infos=[]
    tested_infos=[]
    pred_all = model.get_pred_all(data)
    for sid, theta in tqdm(user_trait.items(),f'get top {k} items'):
        k_info,label,tested_info= get_ncd_k_emc(k, sid, theta, item_trait, model,data,pred_all)
        tested_infos.append(tested_info)
        k_infos.append(k_info)
        labels.append(label)
    return labels,k_infos,tested_infos


def transform(item_trait, user_trait,labels,k_fishers,tested_infos=None):
    if tested_infos:
        for theta, label, k_fisher,tested_info in zip(user_trait.values(),labels,k_fishers,tested_infos):
            itrait = list(item_trait.values())
            item_n = len(itrait)
            user_embs=[]
            for tmp in tested_info:
                user_emb = [theta]
                if type(tmp) == list:
                    user_emb.extend(tmp)
                else:
                    user_emb.append(tmp)
                user_embs.append(user_emb)
            yield pack_batch([
                torch.tensor(user_embs),
                # torch.tensor(user_embs*item_n),
                # torch.tensor(list(zip([theta]*item_n,tested_info))),
                itrait,
                label,
                k_fisher
            ])
    else:
        for theta, label, k_fisher in zip(user_trait.values(),labels,k_fishers):
            itrait = list(item_trait.values())
            item_n = len(itrait)
            if type(theta) == list:
                yield pack_batch([
                    torch.tensor(theta),
                    itrait,
                    label,
                    k_fisher
                ])
            else:
                yield pack_batch([
                    torch.tensor([theta]*item_n).unsqueeze(-1),
                    itrait,
                    label,
                    k_fisher
                ])

def pack_batch(batch):
    theta, itrait, label, k_fisher= batch
    return (
        theta, Tensor(itrait), Tensor(label), k_fisher
    )

def get_k_fisher(k,theta,items):
    fisher_arr = []
    for qid,(alpha,beta) in items.items():
        pred = 1.702*(alpha * (theta - beta))
        pred = torch.sigmoid(torch.tensor(pred))
        # pred = 1 / (1 + np.exp(-pred))
        q = 1 - pred
        fisher_info = float((q*pred*(alpha ** 2)).numpy())
        fisher_arr.append((fisher_info,qid))
    fisher_arr_sorted = sorted(fisher_arr, reverse=True)
    return [i[1] for i in fisher_arr_sorted[:k]],[i[0]for i in fisher_arr],[]

def get_k_emc(k,sid,theta,items,vanilla_model):
    epochs = vanilla_model.config['num_epochs']
    lr = vanilla_model.config['learning_rate']
    device = vanilla_model.config['device']
    item_n = len(items.keys())
    n = random.randint(1,9)
    item_log = sample(list(items.keys()),n)
    labels = []
    for qid in item_log:
        alpha = items[qid][0]
        beta = items[qid][1]
        pred = alpha * theta + beta
        if pred>0.5:
            label=1
        else:
            label = 0
        labels.append(label)
    # logs  = list(zip(item_log,labels))
    
    model = deepcopy(vanilla_model)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)
    sids = [sid]*n
    sids = torch.LongTensor(sids).to(device)
    qids = torch.LongTensor(item_log).to(device)
    labels = torch.LongTensor(labels).to(device)
    # theta0 = model.model.theta(torch.tensor(sid).to(device))
    # print(sid, theta0)
    optimizer.zero_grad()
    preds = model.model(sids, qids).view(-1)
    loss = model._loss_function(preds, labels)
    loss.backward()
    optimizer.step()
    
    # theta1 = model.model.theta(torch.tensor(sid).to(device))
    # print(sid, theta1)
    res_arr = []
    
    for qid,(alpha,beta) in items.items():
        for name, param in model.model.named_parameters():
            if 'theta' not in name:
                param.requires_grad = False

        original_weights = model.model.theta.weight.data.clone()
        # print(original_weights)

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        correct = torch.LongTensor([1]).to(device).float()
        wrong = torch.LongTensor([0]).to(device).float()
        optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)
        for ep in range(epochs):
            optimizer.zero_grad()
            pred = model.model(student_id, question_id)
            loss = model._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = model.model.theta.weight.data.clone()
        model.model.theta.weight.data.copy_(original_weights)
        optimizer = torch.optim.Adam(model.model.parameters(), lr=lr)
        for ep in range(epochs):
            optimizer.zero_grad()
            pred = model.model(student_id, question_id)
            loss = model._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = model.model.theta.weight.data.clone()
        # model.model.theta.weight.data.copy_(original_weights)

        for param in model.model.parameters():
            param.requires_grad = True
        
        if type(alpha) == float:
            alpha = np.array([alpha])
        if type(theta) == float:
            theta = np.array([theta])
        pred = np.matmul(alpha.T, theta) + beta
        pred = 1 / (1 + np.exp(-pred))
        # result = pred * torch.norm(pos_weights - original_weights).item() + \
        #     (1 - pred) * torch.norm(neg_weights - original_weights).item()
        result = pred * (pos_weights - original_weights).sum().tolist()+ \
            (1 - pred) * (neg_weights - original_weights).sum().tolist()            
        res_arr.append((result,qid))
        model.model.theta.weight.data.copy_(original_weights)
    res_arr_sorted = sorted(res_arr, reverse=True)
    # return [i[1] for i in res_arr_sorted[:k]],[i[0]for i in res_arr],logs

    # print(sid, original_weights[sid].tolist())
    # print(pos_weights - original_weights,neg_weights - original_weights)
    # print(pos_weights[sid],neg_weights[sid],original_weights[sid])

    return [i[1] for i in res_arr_sorted[:k]],[i[0]for i in res_arr],original_weights.tolist()[sid]

def get_ncd_k_emc(k,sid,theta,items,vanilla_model,data,pred_all):
    # epochs = vanilla_model.config['num_epochs']
    # lr = vanilla_model.config['learning_rate']
    # device = vanilla_model.config['device']
    model =deepcopy(vanilla_model)
    original_weights = model.model.student_emb.weight.data.clone()
    theta = original_weights[sid].tolist()
    emc_arr = [(model.expected_model_change(sid, qid, data, pred_all),qid) for qid in items.keys()]
    # list(zip(emc_arr,items.keys()))
    # emc_arr_sorted = sorted(list(zip(emc_arr,items.keys())), reverse=True)
    emc_arr_sorted = sorted(emc_arr, reverse=True)
    return [i[1] for i in emc_arr_sorted[:k]],[i[0]for i in emc_arr],theta
    
    
    
def get_k_kli(k, theta, items):
    items_n=len(items.keys())
    # ns = [random.randint(1,19) for i in range(items_n)]
    n = random.randint(1,9)
    dim = 1
    res_arr = []
    for qid,(alpha, beta) in items.items():
    # for (qid,(alpha, beta)),n in zip(items.items(),ns):
        if type(alpha) == float:
            alpha = np.array([alpha])
        if type(theta) == float:
            theta = np.array([theta])
        pred_estimate = np.matmul(alpha.T, theta) + beta
        pred_estimate = 1 / (1 + np.exp(-pred_estimate))
        def kli(x):
            if type(x) == float:
                x = np.array([x])
            pred = np.matmul(alpha.T, x) + beta
            pred = 1 / (1 + np.exp(-pred))
            q_estimate = 1 - pred
            q = 1 - pred
            
            if pred==0.0 or pred==1.0:
                # threshold = 1e-5
                # pred = min(pred,1-threshold)
                # pred = max(pred,threshold)
                # q = min(q,1-threshold)
                # q = max(q,threshold)
                # pred_estimate = min(pred_estimate,1-threshold)
                # pred_estimate = max(pred_estimate,threshold)
                return 0
            # print(pred_estimate * np.log(pred_estimate / pred) + \
                # q_estimate * np.log((q_estimate / q)) )
            return pred_estimate * np.log(pred_estimate / pred) + \
                q_estimate * np.log((q_estimate / q)) 
        c = 3
        boundaries = [
            [theta[i] - c / np.sqrt(n), theta[i] + c / np.sqrt(n)] for i in range(dim)]
        # if theta[0]==4.2123823165893555 and qid==192:
        #         print(1)
        if len(boundaries) == 1:
            # KLI
            v, err = integrate.quad(kli, boundaries[0][0], boundaries[0][1])
            res_arr.append((v,qid))
        else:
            # MKLI
            integ = vegas.Integrator(boundaries)
            result = integ(kli, nitn=10, neval=1000)
            res_arr.append((result.mean,qid))
    res_arr_sorted = sorted(res_arr, reverse=True)
    for idx,(info,_) in enumerate(res_arr):
        if info!=info:
            print(idx,info)
    return [i[1] for i in res_arr_sorted[:k]],[i[0]for i in res_arr],[c / np.sqrt(n)]*items_n
    # return [i[1] for i in res_arr_sorted[:k]],[i[0]for i in res_arr],ns
    # return [i[1] for i in res_arr_sorted[:k]],[i[0]for i in res_arr],[c / np.sqrt(n) for n in ns]
