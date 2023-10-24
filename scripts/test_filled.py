import sys
sys.path.append('..')
import CAT
import json
import torch
import logging
import datetime
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
from CAT.distillation.MFI.model import distillModel 
from CAT.mips.ball_tree import BallTree,search_metric_tree,search_metric_tree_k
import heapq

def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def main(dataset = "eedi", cdm = "irt", stg = ['Random'], test_length = 10, ctx="cuda:1", lr=0.2, num_epoch=1, efficient=False,  postfix='11', dissimilarity_partition=False, last_leaf=False, threshold=0.95):
    search_k=False
    if dataset =='eedi':
        leaves_threshold=50
    # lr=0.05 if dataset=='assistment' else 0.2
    setuplogger()
    logging.info('start loading file')
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(0)
    lr_config={
        "assistment":{
            "Random":0.03,
            "MFI":0.03,
            "KLI":0.04,
            'MAAT':0.08
        },
        "ifytek":{
            "MFI":0.2,
            "KLI":0.2,
            "Random":0.2,
            'MAAT':0.15
        },
       "junyi":{
            "Random":0.08,
            "MFI":0.08,
            "KLI":0.1,
            'MAAT':0.149
       },
       "eedi": {
            "Random": 0.01,
            "MFI": 0.01,
            "KLI": 0.01,
            # 'MAAT':0.15
            'MAAT': 0.01,
        } 
    }
    path_prefix = os.path.abspath('.')
    metadata = json.load(open(f'{path_prefix}/data/{dataset}/metadata.json', 'r'))
    users = json.load(open(f'{path_prefix}/data/{dataset}/users.json', 'r'))
    ckpt_path = f'{path_prefix}/ckpt/{dataset}/{cdm}.pt'
    # read datasets
    test_triplets = pd.read_csv(f'{path_prefix}/data/{dataset}/test_{cdm}_filled_triplets.csv', encoding='utf-8').to_records(index=False)
    # test_triplets = pd.read_csv(f'{path_prefix}/data/{dataset}/test_triplets.csv', encoding='utf-8').to_records(index=False)
    # valid_triplets = pd.read_csv(f'{path_prefix}/data/{dataset}/test_triples.csv', encoding='utf-8').to_records(index=False)
    map_name = 'concept_map'
    concept_map = json.load(open(f'{path_prefix}/data/{dataset}/{map_name}.json', 'r'))
    concept_map = {int(k):v for k,v in concept_map.items()}
    item_n = len(concept_map.keys())
    test_data = CAT.dataset.AdapTestDataset(test_triplets, concept_map,
                                            metadata['num_test_students'], 
                                            metadata['num_questions'], 
                                            metadata['num_concepts'])
    # valid_data = CAT.dataset.AdapTestDataset(valid_triplets, concept_map,
    #                                         metadata['num_test_students'], 
    #                                         metadata['num_questions'], 
    #                                         metadata['num_concepts'])
    strategy_dict = {
        'Random' : CAT.strategy.RandomStrategy(),
        'MFI' : CAT.strategy.MFIStrategy(),
        'KLI' : CAT.strategy.KLIStrategy(),
        'MAAT' : CAT.strategy.MAATStrategy(),
    } 
   
    strategies = [strategy_dict[i] for i in stg]
    df = pd.DataFrame() 
    df1 = pd.DataFrame()
    for i, strategy in enumerate(strategies):
        config = {
            'learning_rate': lr_config[dataset][stg[i]],
            'batch_size': 2048,
            'num_epochs': num_epoch,
            'num_dim': 1, # for IRT or MIRT
            'device': ctx,
            # for NeuralCD
            # 'prednet_len1': 128,
            # 'prednet_len2': 64,
            'prednet_len1': 32,
            'prednet_len2': 16,
        }
        if cdm == 'irt':
            model = CAT.model.IRTModel(**config)
        elif cdm =='ncd':
            model = CAT.model.NCDModel(**config)
        model.init_model(test_data)
        model.adaptest_load(ckpt_path)
        test_data.reset()
        if efficient:
            # trait = json.load(open(f'{path_prefix}/data/{dataset}/{stg[i]}/trait.json', 'r'))
            ball_trait = json.load(open(f'{path_prefix}/data/{dataset}/{stg[i]}/ball_trait{postfix}.json', 'r'))
            ball_trait = {int(k):v for k,v in ball_trait.items()}
            # item_label = json.load(open(f"{path_prefix}/data/{dataset}/{stg[i]}/item_label.json", 'r'))
            distill_k=50
            embedding_dim=15
            if stg[i]=='KLI':
                # tested_info= trait['tested_info']
                # user_dim=np.array(tested_info).shape[-1]+1
                user_dim=2
            else:
                user_dim=1
            dmodel = distillModel(distill_k,embedding_dim,user_dim,device=ctx)
            dmodel.load(f'{path_prefix}/ckpt/eedi/{cdm}_{stg[i]}_ip{postfix}.pt')
        logging.info('-----------')
        logging.info(f'start adaptive testing with {strategy.name} strategy')
        logging.info('lr: ' + str(config['learning_rate']))
        logging.info(f'Iteration 0')
        res=[]
        time=0
        # starttime = datetime.datetime.now()
        logs=[]
        t_queries=[]
        for sid in tqdm(test_data.data.keys(),'testing '):
            log=[]
            q_log=[]
            if efficient:
                # selected_ball_trait = {}
                # for k,v in enumerate(zip(item_label,ball_trait)):
                    # selected_ball_trait[k]=v
                # T = BallTree(selected_ball_trait,dissimilarity_partition,threshold=leaves_threshold)
                T = BallTree(ball_trait,dissimilarity_partition,threshold=leaves_threshold)
                
            tmp_model= deepcopy(model)
            
            results={}
            results['mse'] = abs(tmp_model.get_theta(torch.LongTensor([sid]).to(ctx))[0]-users[str(sid)])
            results['time']=0
            results['count']=0
            tmp =[list(results.values())]
            time = datetime.timedelta(microseconds=0)
            res_q={'qid':-1,'quantity':-1,'leaves':{}}
            # theta qids
            candidates={}
            # leaves_candidates=set()
            queries=[]
            qlog=[]
            for it in range(1, test_length + 1):
                # print(tmp_model.model.theta(torch.tensor(sid).to(ctx)))
                # print(results)
                starttime = datetime.datetime.now()
                if efficient:
                    theta = tmp_model.model.theta(torch.tensor(sid).to(ctx))
                    # print(theta)
                    if user_dim==1:
                        u_emb = dmodel.model.utn(theta).tolist()
                    else:
                        if stg[i]=='KLI':
                            u_emb = dmodel.model.utn(torch.cat((theta,torch.Tensor([3/np.sqrt(it)]).to(ctx)),0)).tolist()
                        # elif stg[i]=='MFI':
                        #     u_emb = dmodel.model.utn(torch.cat((theta,torch.Tensor(avg_tested_emb).to(ctx)),0)).tolist()
                        # elif stg[i] == 'MAAT':
                        #     weight =tmp_model.model.theta.weight.data.clone()[sid]
                        #     print(theta, weight)
                        #     u_emb = dmodel.model.utn(torch.cat((theta,weight),0)).tolist()
                    queries.append(u_emb)
                    if not search_k:
                        tested_set = set(test_data.tested[sid])
                        # threshold=0.9
                        # max_cos = -1
                        # s_query = None
                        leaves_candidates=set()
                        count=0
                        for key , qs in candidates.items():
                            a = np.array(key)
                            b = np.array(u_emb)
                            tmp_cos = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
                            if tmp_cos>threshold:
                                count+=1
                                # max_cos = tmp_cos
                                # s_query = key
                                # count=len(qs)
                                leaves_candidates.update(qs)
                        leaves_candidates_inl = list(leaves_candidates-tested_set)
                        
                        if last_leaf and len(leaves_candidates_inl)>0:
                            tmp_ip = [np.dot(np.array(u_emb),np.array(ball_trait[q][1])) for q in leaves_candidates_inl]
                            qip = heapq.nlargest(1, tmp_ip)
                            qid = leaves_candidates_inl[list(map(tmp_ip.index, qip))[0]]
                        else:
                            res_q={'qid':-1,'quantity':-1,'leaves':{}}
                            count = search_metric_tree(res_q, tested_set,np.array(u_emb),T)
                            qid = res_q['qid']
                            
                            candidates[tuple(u_emb)]=set(res_q['leaves'].keys())
                    else:
                        tmp_candidates=dict(zip(list(range(metadata['num_questions'],metadata['num_questions']+it)),[0]*it))
                        count  = search_metric_tree_k(tmp_candidates,np.array(u_emb),T)
                        untested_qids = set(tmp_candidates.keys())-set(test_data.tested[sid])
                        if len(untested_qids) == 1:
                            max_score = 0 
                            for k,v in tmp_candidates.items():
                                if k in untested_qids:
                                    if v>max_score:
                                        qid=k
                                        max_score=v
                        else:
                            qid = strategy.adaptest_select(tmp_model, sid, test_data,item_candidates=untested_qids) 
                else:
                     qid = strategy.adaptest_select(tmp_model, sid, test_data)
                log.append([float(tmp_model.get_theta(torch.LongTensor([sid]).to(ctx))),float(tmp_model.get_alpha(torch.LongTensor([qid]).to(ctx))[0]),float(tmp_model.get_beta(torch.LongTensor([qid]).to(ctx))[0])])
                # print(log)
                test_data.apply_selection(sid, qid)
                # tmp_model.adaptest_update(sid, qid, test_data)
                tmp_model.adaptest_update1(sid, qid, test_data)
                time += (datetime.datetime.now() - starttime)
                # results = tmp_model.evaluate(sid, valid_data)
                # print(tmp_model.get_theta(torch.LongTensor([sid]).to(ctx))[0])
                q_log.append(qid)
                # print(abs(tmp_model.get_theta(torch.LongTensor([sid]).to(ctx))[0]-users[str(sid)]))
                # print('======================')
                print(tmp_model.get_theta(torch.LongTensor([sid]).to(ctx))[0],users[str(sid)])
                results['mse'] = abs(tmp_model.get_theta(torch.LongTensor([sid]).to(ctx))[0]-users[str(sid)])
                # del results['cov']
                results['time']=time.seconds+time.microseconds*1e-6
                if efficient:
                    results['count']=count
                else:
                    results['count']=0
                tmp.append(list(results.values()))
                # print(tmp)
            # print(q_log)
            t_queries.append(queries)
            res.append(tmp)
            logs.append(log)
        with open(f'{path_prefix}/data/{dataset}/{stg[i]}/log.json', "w", encoding="utf-8") as f:
            f.write(json.dumps(logs, ensure_ascii=False))
        if efficient:
            with open(f'{path_prefix}/data/{dataset}/{stg[i]}/query.json', "w", encoding="utf-8") as f:
                f.write(json.dumps(t_queries, ensure_ascii=False))

        # time +=  (datetime.datetime.now() - starttime).seconds
        res = torch.mean(torch.Tensor(res).permute(2,1,0), dim=-1).tolist()
        exp_info={
            f"{stg[i]}": ['mse']+res[0],
            # f"{config['learning_rate']}": ['auc']+res[1],
            f"{config['learning_rate']}": ['time']+res[1],
            "  ": ['count']+res[2]
            }    
        exp_info = pd.DataFrame(exp_info)
        idx= ['']
        idx.extend(range(0,test_length+1))
        exp_info.index=idx
        
        selected_num = [3,5,10]
        # selected_num = [5,10,20]
        short_mse = [mse for i,mse in enumerate(res[0]) if i in selected_num]
        # short_auc = [auc for i,auc in enumerate(res[1]) if i in selected_num]
        short_time = [auc for i,auc in enumerate(res[1]) if i in selected_num]
        short_cnt = [auc for i,auc in enumerate(res[2]) if i in selected_num]
        short_exp_info={
            f"{stg[i]}": ['mse']+short_mse,
            # f"{config['learning_rate']}": ['auc']+short_auc,
            f"{config['learning_rate']}": ['time']+short_time,
            "   ": ['cnt']+short_cnt,
        }
        short_exp_info = pd.DataFrame(short_exp_info)
        idx= ['']
        idx.extend(selected_num)
        short_exp_info.index=idx

        print(exp_info.transpose())
        print(short_exp_info.transpose())
        
        df1 = df1.append(short_exp_info.transpose())
        df = df.append(exp_info.transpose())
    df1.to_csv(
        f"{path_prefix}/data/{dataset}/model/{cdm}/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}_short_{'_'.join(stg)}.csv")
    df.to_csv(f"{path_prefix}/data/{dataset}/model/{cdm}/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}_{'_'.join(stg)}.csv")  

if __name__ == '__main__':
    import fire

    fire.Fire(main)
