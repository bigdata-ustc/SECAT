import CAT
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import os


def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def run(cdm, model, dataset, *args, **kwargs):
    path_prefix = os.path.abspath('.')
    if 'SECAT' not in path_prefix:
        path_prefix+='/SECAT'
    train_triplets = pd.read_csv(
        f'{path_prefix}/data/{dataset}/train_triplets.csv',
        encoding='utf-8').to_records(index=False)
    valid_triplets = pd.read_csv(
        f'{path_prefix}/data/{dataset}/valid_triplets.csv',
        encoding='utf-8').to_records(index=False)
    concept_map = json.load(
        open(f'{path_prefix}/data/{dataset}/concept_map.json', 'r'))
    concept_map = {int(k): v for k, v in concept_map.items()}
    metadata = json.load(
        open(f'{path_prefix}/data/{dataset}/metadata.json', 'r'))
    train_data = CAT.dataset.TrainDataset(train_triplets, concept_map,
                                          metadata['num_train_students'],
                                          metadata['num_questions'],
                                          metadata['num_concepts'])
    valid_data = CAT.dataset.TrainDataset(valid_triplets, concept_map,
                                          metadata['num_train_students'],
                                          metadata['num_questions'],
                                          metadata['num_concepts'])

    model.init_model(valid_data)
    model.train(train_data, test_data=None)
    # model.train(train_data, test_data=valid_data)
    model.adaptest_save(f'{path_prefix}/ckpt/{dataset}/{cdm}.pt')
    model.adaptest_save(f'{path_prefix}/ckpt/{dataset}/{cdm}_with_theta.pt',
                        save_theta=True)


def main(dataset="eedi", cdm="ncd", ctx="cpu", num_dim=1, lr=0.025):
    setuplogger()
    seed = 0
    torch.manual_seed(seed)
    epoch_config = {
        'assistment': {
            'irt': 4,
            'ncd': 20
        },
        'ifytek': {
            'irt': 1,
            'ncd': 1
        },
        'junyi': {
            'irt': 10,
            'ncd': 3
        },
        'eedi': {
            'irt': 1,
            'ncd': 1
        },
    }
    config = {
        'learning_rate': lr,
        'batch_size': 2048,
        'num_epochs': epoch_config[dataset][cdm],
        'num_dim': num_dim,  # for IRT or MIRT
        'device': ctx,
        # for NeuralCD
        'prednet_len1': 128,
        'prednet_len2': 64,
    }
    if cdm == 'irt':
        model = CAT.model.IRTModel(**config)
    elif cdm == 'ncd':
        model = CAT.model.NCDModel(**config)

    run(cdm=cdm, model=model, dataset=dataset, **config)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
