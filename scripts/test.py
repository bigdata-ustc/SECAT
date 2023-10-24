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
from CAT.mips.ball_tree import BallTree, search_metric_tree
import heapq
import os


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def main(dataset="eedi",
         cdm="irt",
         stg=['Random','MAAT'],
         test_length=10,
         ctx="cpu",
         inc=False,
         num_epoch=4,
         efficient=False,
         postfix='11',
         dissimilarity_partition=False,
         last_leaf=False,
         threshold=0.95):
    leaves_threshold = 20
    # setuplogger()
    logging.info('start loading file')
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(20)
    path_prefix = os.path.abspath('.')
    if 'SECAT' not in path_prefix:
        path_prefix += '/SECAT'
    
    lr_config = {
        
        "ifytek": {
            "Random": 0.03,
            "MFI": 0.03,
            "KLI": 0.04,
            'MAAT': 0.13
        },
        "eedi": {
            "Random": 0.01,
            "MFI": 0.01,
            "KLI": 0.01,
            # 'MAAT':0.15
            'MAAT': 0.01
        }
    }

    metadata = json.load(
        open(f'{path_prefix}/data/{dataset}/metadata.json', 'r'))
    ckpt_path = f'{path_prefix}/ckpt/{dataset}/{cdm}.pt'
    test_triplets = pd.read_csv(
        f'{path_prefix}/data/{dataset}/test_triplets.csv',
        encoding='utf-8').to_records(index=False)
    map_name = 'concept_map'
    concept_map = json.load(
        open(f'{path_prefix}/data/{dataset}/{map_name}.json', 'r'))
    concept_map = {int(k): v for k, v in concept_map.items()}
    test_data = CAT.dataset.AdapTestDataset(test_triplets, concept_map,
                                            metadata['num_test_students'],
                                            metadata['num_questions'],
                                            metadata['num_concepts'])
    strategy_dict = {
        'Random': CAT.strategy.RandomStrategy(),
        'MFI': CAT.strategy.MFIStrategy(),
        'KLI': CAT.strategy.KLIStrategy(),
        'MAAT': CAT.strategy.MAATStrategy(),
    }

    strategies = [strategy_dict[i] for i in stg]
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    for i, strategy in enumerate(strategies):
        config = {
            'learning_rate': lr_config[dataset][stg[i]],
            'batch_size': 2048,
            'num_epochs': num_epoch,
            'num_dim': 1,  # for IRT or MIRT
            'device': ctx,
            # for NeuralCD
            'prednet_len1': 128,
            'prednet_len2': 64,
        }
        if cdm == 'irt':
            model = CAT.model.IRTModel(**config)
        elif cdm == 'ncd':
            model = CAT.model.NCDModel(**config)
        model.init_model(test_data)
        model.adaptest_load(ckpt_path)
        test_data.reset()
        if efficient:
            trait = json.load(
                open(f'{path_prefix}/data/{dataset}/{stg[i]}/trait.json', 'r'))
            ball_trait = json.load(
                open(
                    f'{path_prefix}/data/{dataset}/{stg[i]}/ball_trait{postfix}.json',
                    'r'))
            ball_trait = {int(k):v for k,v in ball_trait.items()}
            distill_k = 50
            embedding_dim = 15
            if stg[i] == 'KLI':
                user_dim = 2
            else:
                user_dim = 1
            dmodel = distillModel(distill_k,
                                  embedding_dim,
                                  user_dim,
                                  device=ctx)
            dmodel.load(
                f'{path_prefix}/ckpt/eedi/{cdm}_{stg[i]}_ip{postfix}.pt')
        logging.info('-----------')
        logging.info(f'start adaptive testing with {strategy.name} strategy')
        logging.info('lr: ' + str(config['learning_rate']))
        logging.info(f'Iteration 0')
        res = []
        time = 0
        logs = []
        for sid in tqdm(list(test_data.data.keys()), 'testing '):
            log = []
            if efficient:
                selected_ball_trait = {}
                qids = test_data.untested[sid]
                for k, v in ball_trait.items():
                    if k in qids:
                        selected_ball_trait[k] = v
                T = BallTree(selected_ball_trait,
                             dissimilarity_partition,
                             threshold=leaves_threshold)
            tmp_model = deepcopy(model)
            results = tmp_model.evaluate(sid, test_data)
            # print(results)
            results['count'] = 0
            tmp = [list(results.values())]
            time = datetime.timedelta(microseconds=0)
            res_q = {'qid': -1, 'quantity': -1, 'leaves': {}}
            # theta qids
            candidates = {}
            # leaves_candidates=set()
            for it in range(1, test_length + 1):
                starttime = datetime.datetime.now()
                if efficient:
                    theta = tmp_model.model.theta(torch.tensor(sid).to(ctx))
                    if user_dim == 1:
                        u_emb = dmodel.model.utn(theta).tolist()
                    else:
                        if stg[i] == 'KLI':
                            u_emb = dmodel.model.utn(
                                torch.cat(
                                    (theta, torch.Tensor([3 / np.sqrt(it)
                                                          ]).to(ctx)),
                                    0)).tolist()
                        elif stg[i] == 'MFI':
                            if len(test_data.tested[sid]) == 0:
                                avg_tested_emb = np.array([0, 0]).tolist()
                            else:
                                avg_tested_emb = np.array([
                                    trait['item'][str(qid)]
                                    for qid in test_data.tested[sid]
                                ]).mean(axis=0).tolist()
                            avg_tested_emb.extend([it])
                            u_emb = dmodel.model.utn(
                                torch.cat(
                                    (theta,
                                     torch.Tensor(avg_tested_emb).to(ctx)),
                                    0)).tolist()
                    tested_set = set(test_data.tested[sid])
                    leaves_candidates = set()
                    count = 0
                    for key, qs in candidates.items():
                        a = np.array(key)
                        b = np.array(u_emb)
                        tmp_cos = a.dot(b) / (np.linalg.norm(a) *
                                              np.linalg.norm(b))
                        if tmp_cos > threshold:
                            count += 1
                            leaves_candidates.update(qs)
                    leaves_candidates_inl = list(leaves_candidates -
                                                 tested_set)

                    if last_leaf and len(leaves_candidates_inl) > 0:
                        tmp_ip = [
                            np.dot(np.array(u_emb), np.array(ball_trait[q]))
                            for q in leaves_candidates_inl
                        ]
                        qip = heapq.nlargest(1, tmp_ip)
                        qid = leaves_candidates_inl[list(map(
                            tmp_ip.index, qip))[0]]
                    else:
                        res_q = {'qid': -1, 'quantity': -1, 'leaves': {}}
                        count = search_metric_tree(res_q, tested_set,
                                                   np.array(u_emb), T)
                        qid = res_q['qid']

                        candidates[tuple(u_emb)] = set(res_q['leaves'].keys())
                else:
                    qid = strategy.adaptest_select(tmp_model, sid, test_data)
                test_data.apply_selection(sid, qid)
                if inc:
                    tmp_model.adaptest_update(sid, qid, test_data)
                else:
                    # tmp_model = deepcopy(model)
                    tmp_model.adaptest_update1(sid, qid, test_data)
                time += (datetime.datetime.now() - starttime)
                results = tmp_model.evaluate(sid, test_data)
                del results['cov']
                results['time'] = time.seconds + time.microseconds * 1e-6
                if efficient:
                    results['count'] = count
                else:
                    results['count'] = 0
                tmp.append(list(results.values()))
            res.append(tmp)
            logs.append(log)
        with open(f'{path_prefix}/data/{dataset}/{stg[i]}/log.json',
                  "w",
                  encoding="utf-8") as f:
            f.write(json.dumps(logs, ensure_ascii=False))

        # time +=  (datetime.datetime.now() - starttime).seconds
        res = torch.mean(torch.Tensor(res).permute(2, 1, 0), dim=-1).tolist()
        exp_info = {
            f"{stg[i]}": ['acc'] + res[0],
            f"{config['learning_rate']}": ['auc'] + res[1],
            "": ['time'] + res[2],
            "  ": ['count'] + res[3]
        }
        exp_info = pd.DataFrame(exp_info)
        idx = ['']
        idx.extend(range(0, test_length + 1))
        exp_info.index = idx
        if test_length == 10:
            selected_num = [3, 5, 10]
        else:
            selected_num = [5, 10, 20]
        short_acc = [acc for i, acc in enumerate(res[0]) if i in selected_num]
        short_auc = [auc for i, auc in enumerate(res[1]) if i in selected_num]
        short_time = [auc for i, auc in enumerate(res[2]) if i in selected_num]
        short_cnt = [auc for i, auc in enumerate(res[3]) if i in selected_num]
        short_exp_info = {
            f"{stg[i]}": ['acc'] + short_acc,
            f"{config['learning_rate']}": ['auc'] + short_auc,
            "  ": ['time'] + short_time,
            "   ": ['cnt'] + short_cnt,
        }
        short_exp_info = pd.DataFrame(short_exp_info)
        idx = ['']
        idx.extend(selected_num)
        short_exp_info.index = idx

        print(exp_info.transpose())
        print(short_exp_info.transpose())

        df1 = df1.append(short_exp_info.transpose())
        df = df.append(exp_info.transpose())
    df1.to_csv(
        f"{path_prefix}/data/{dataset}/model/{cdm}/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}_short_{'_'.join(stg)}.csv"
    )
    df.to_csv(
        f"{path_prefix}/data/{dataset}/model/{cdm}/{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')}_{'_'.join(stg)}.csv"
    )


if __name__ == '__main__':
    import fire

    fire.Fire(main)
