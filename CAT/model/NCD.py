from urllib.parse import scheme_chars
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

from CAT.model.abstract_model import AbstractModel
from CAT.dataset import AdapTestDataset, TrainDataset, Dataset
import random


class NCD(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self,
                 student_n,
                 exer_n,
                 knowledge_n,
                 prednet_len1=128,
                 prednet_len2=64):
        self.knowledge_dim = 4
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = prednet_len1, prednet_len2  # changeable
        # self.prednet_len2 = knowledge_n
        super(NCD, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = nn.Linear(self.prednet_input_len,
                                       self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        torch.manual_seed(0)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb=None):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10

        # input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = e_discrimination * (stu_emb - k_difficulty)
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))
        # output = torch.sigmoid(input_x)
        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        # stat_emb = self.student_emb(stu_id)
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # k_difficulty = self.k_difficulty(exer_id)
        # e_discrimination = self.e_discrimination(exer_id)
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class NCDModel(AbstractModel):
    def __init__(self, precision=6, **config):
        super().__init__()
        self.config = config
        self.model = None
        self._epsilon = float("1e-" + str(precision))

    @property
    def name(self):
        return 'Neural Cognitive Diagnosis'

    def init_model(self, data: Dataset):
        # self.model = NCD(data.num_students, data.num_questions, 3,
        self.model = NCD(data.num_students, data.num_questions,
                         data.num_concepts, self.config['prednet_len1'],
                         self.config['prednet_len2'])

    def train(self, train_data: TrainDataset, test_data=None):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        self.model.to(device)
        logging.info('train on {}'.format(device))

        train_loader = data.DataLoader(train_data,
                                       batch_size=batch_size,
                                       shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for ep in range(1, epochs + 1):
            loss = []
            log_step = 20
            for student_ids, question_ids, concepts_emb, labels in tqdm(
                    train_loader, f'Epoch {ep}'):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                concepts_emb = concepts_emb.to(device)
                labels = labels.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                # loss += bz_loss.data.float()
                loss.append(bz_loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (ep, float(np.mean(loss))))
            if test_data is not None:
                test_loader = data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=True)
                self.eval(test_loader, device)

                # if cnt % log_step == 0:
                # logging.info('Epoch [{}] Batch [{}]: loss={:.5f}'.format(ep, cnt, loss / cnt))

    def eval(self, adaptest_data: AdapTestDataset, device):
        # data = adaptest_data.data
        self.model.to(device)

        with torch.no_grad():
            self.model.eval()
            y_pred = []
            y_true = []
            y_label = []
            for student_ids, question_ids, concepts_emb, labels in tqdm(
                    adaptest_data, "evaluating"):
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_emb = torch.Tensor(concepts_emb).to(device)
                pred: torch.Tensor = self.model(student_ids, question_ids,
                                                concepts_emb).view(-1)
                y_pred.extend(pred.detach().cpu().tolist())
                y_true.extend(labels.tolist())
                y_label.extend([0 if p < 0.5 else 1 for p in pred])
            self.model.train()

        # y_true = np.array(y_true)
        # y_pred = np.array(y_pred)
        acc = accuracy_score(y_true, y_label)
        auc = roc_auc_score(y_true, y_pred)
        print(classification_report(y_true, y_label, digits=4))
        print('auc:', auc)

        return {
            'acc': acc,
            'auc': auc,
        }

    def fill(self, sid, qids, adaptest_data=None, MonteCarlo=True):
        device = self.config['device']
        self.model.to(device)
        tmp_res = []

        concepts_embs = []
        for qid in qids:
            concepts_emb = [0.] * adaptest_data.num_concepts
            for concept in adaptest_data.concept_map[qid]:
                concepts_emb[concept] = 1.0
            concepts_embs.append(concepts_emb)
        sid_t = torch.LongTensor([sid] * len(qids)).to(device)
        qid_t = torch.LongTensor(list(qids)).to(device)
        concepts_emb_t = torch.Tensor(concepts_embs).to(device)
        pred_t = self.model(sid_t, qid_t, concepts_emb_t).view(-1).tolist()
        if MonteCarlo:
            pred_t = [1 if random.random() < p else 0 for p in pred_t]
        else:
            pred_t = [0 if p < 0.5 else 1 for p in pred_t]
        res = list(zip([sid] * len(qids), list(qids), pred_t))
        # res.append([sid, qid, pred])
        return res

    def _loss_function(self, pred, real):
        pred_0 = torch.ones(pred.size()).to(self.config['device']) - pred
        output = torch.cat((pred_0, pred), 1)
        criteria = nn.NLLLoss()
        return criteria(torch.log(output), real)

    # def _loss_function1(self, pred, real):
    #     pred_0 = torch.ones(pred.size()).to(self.config['device']) - pred
    #     output = torch.cat((pred_0, pred), 1)
    #     criteria = nn.NLLLoss()
    #     return criteria(torch.log(output), real)

    def adaptest_save(self, path, save_theta=False):
        """
        Save the model. Do not save the parameters for students.
        """
        model_dict = self.model.state_dict()
        if save_theta == False:
            model_dict = {
                k: v
                for k, v in model_dict.items() if 'student' not in k
            }
        torch.save(model_dict, path)

    def adaptest_load(self, path):
        """
        Reload the saved model
        """
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.config['device'])

    def adaptest_update(self,
                        sid,
                        qid,
                        adaptest_data: AdapTestDataset,
                        update_lr=None,
                        optimizer=None,
                        scheduler=None):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.student_emb.parameters(),
                                         lr=lr)

        label = adaptest_data.data[sid][qid]
        # print('label:', label)
        concepts_emb = [0.] * adaptest_data.num_concepts
        for concept in adaptest_data.concept_map[qid]:
            concepts_emb[concept] = 1.0
        sid = torch.LongTensor([sid]).to(device)
        qid = torch.LongTensor([qid]).to(device)
        label = torch.LongTensor([int(label)]).to(device)
        concepts_emb = torch.Tensor(concepts_emb).to(device)
        pred: torch.Tensor = self.model(sid, qid, concepts_emb)
        bz_loss = self._loss_function(pred, label)
        optimizer.zero_grad()
        bz_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            # print('lr:', scheduler.get_last_lr())
        self.model.apply_clipper()
        # print('concept:', concepts_emb)
        # print('difficulty:', self.model.get_exer_params(qid)[0]*concepts_emb)
        # print('disc:', self.model.get_exer_params(qid)[1]*concepts_emb)
        # print('stu_emb: ', self.model.get_knowledge_status(sid))

    def adaptest_update1(self,
                         sid,
                         qid,
                         adaptest_data: AdapTestDataset,
                         update_lr=None,
                         optimizer=None,
                         scheduler=None):
        lr = self.config['learning_rate']
        device = self.config['device']
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.student_emb.parameters(),
                                         lr=lr)
        #
        qid = adaptest_data.tested[sid]
        label = [adaptest_data.data[sid][q] for q in qid]
        tested_len = len(adaptest_data.tested[sid])
        sid = torch.LongTensor([sid] * tested_len).to(device)

        #
        # label = adaptest_data.data[sid][qid]
        concepts_embs = []
        for q in qid:
            concepts = adaptest_data.concept_map[q]
            concepts_emb = [0.] * adaptest_data.num_concepts
            for concept in concepts:
                concepts_emb[concept] = 1.0
            concepts_embs.append(concepts_emb)
        qid = torch.LongTensor(qid).to(device)
        label = torch.LongTensor(label).to(device)
        concepts_emb = torch.Tensor(concepts_embs).to(device)
        l_loss = 10.0
        epoch = 0
        # original_weights = self.model.student_emb.weight.data.clone()
        while True:
            pred: torch.Tensor = self.model(sid, qid, concepts_emb)
            bz_loss = self._loss_function(pred, label)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()
            self.model.apply_clipper()
            epoch += 1
            # print(bz_loss.item(), '-', l_loss, '=',
            #       abs(bz_loss.item() - l_loss))
            if abs(bz_loss.item() -
                   l_loss) < self._epsilon and bz_loss.item() < l_loss:
                break
            l_loss = bz_loss.item()
        # new_weights = self.model.student_emb.weight.data.clone()
        # print(new_weights-original_weights)
            # print(l_loss)
        # print(epoch)
        # pass

    def evaluate(self, sid, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        with torch.no_grad():
            self.model.eval()
            # for sid in data:
            student_ids = [sid] * len(data[sid])
            question_ids = list(data[sid].keys())
            concepts_embs = []
            for qid in question_ids:
                concepts = concept_map[qid]
                concepts_emb = [0.] * adaptest_data.num_concepts
                for concept in concepts:
                    concepts_emb[concept] = 1.0
                concepts_embs.append(concepts_emb)
            real = [data[sid][qid] for qid in question_ids]
            student_ids = torch.LongTensor(student_ids).to(device)
            question_ids = torch.LongTensor(question_ids).to(device)
            concepts_embs = torch.Tensor(concepts_embs).to(device)
            # print(student_ids,question_ids)
            output = self.model(student_ids, question_ids,
                                concepts_embs).view(-1)
            pred = output.tolist()
            # print('test:', self.model.student_emb.weight[sid].sum())
            self.model.train()

        # coverages = []
        # for sid in data:
        #     all_concepts = set()
        #     tested_concepts = set()
        #     for qid in data[sid]:
        #         all_concepts.update(set(concept_map[qid]))
        #     for qid in adaptest_data.tested[sid]:
        #         tested_concepts.update(set(concept_map[qid]))
        #     coverage = len(tested_concepts) / len(all_concepts)
        #     coverages.append(coverage)
        # cov = sum(coverages) / len(coverages)

        real = np.array(real)
        pred = np.array(pred)
        pred_label = [0 if p < 0.5 else 1 for p in pred]
        acc = accuracy_score(real, pred_label)
        auc = roc_auc_score(real, pred)
        # print('acc:', acc,'auc:', auc)
        # print('\n')

        return {
            'acc': acc,
            'auc': auc,
            'cov': 0,
        }

    def get_pred(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        pred_all = {}
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids,
                                    concepts_embs).view(-1).tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.model.train()
        return pred_all

    def get_pred_all(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        pred_all = {}
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                pred_all[sid] = {}
                qids = list(adaptest_data.concept_map.keys())
                student_ids = [sid] * len(qids)
                concepts_embs = []
                for qid in qids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(qids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids,
                                    concepts_embs).view(-1).tolist()
                for i, qid in enumerate(qids):
                    pred_all[sid][qid] = output[i]
            self.model.train()
        return pred_all
    
    def expected_model_change1(self, sid: int, qid: int,
                              adaptest_data: AdapTestDataset, pred_all: dict):
        lr = self.config['learning_rate']
        device = self.config['device']
        
        for name, param in self.model.named_parameters():
            if 'student' not in name:
                param.requires_grad = False

        original_weights = self.model.student_emb.weight.data.clone()
        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        concepts = adaptest_data.concept_map[qid]
        concepts_emb = [0.] * adaptest_data.num_concepts
        for concept in concepts:
            concepts_emb[concept] = 1.0
        concepts_emb = torch.Tensor([concepts_emb]).to(device)
        correct = torch.LongTensor([1]).to(device)
        wrong = torch.LongTensor([0]).to(device)
        
        qid = adaptest_data.tested[sid]
        label = [adaptest_data.data[sid][q] for q in qid]
        concepts_embs = []
        for q in qid:
            concepts = adaptest_data.concept_map[q]
            tmp = [0.] * adaptest_data.num_concepts
            for concept in concepts:
                tmp[concept] = 1.0
            concepts_embs.append(tmp)
        concepts_embs = torch.Tensor(concepts_embs).to(device)
        label = [adaptest_data.data[sid][q] for q in qid]
        qid = torch.LongTensor(qid).to(device)
        tested_len = len(adaptest_data.tested[sid])
        sids = torch.LongTensor([sid] * (tested_len+1)).to(device)
        label = torch.LongTensor(label).to(device)
        # sids = torch.cat((sid, student_id))
        qids = torch.cat((qid, question_id))
        cpts = torch.cat((concepts_embs, concepts_emb))
        #
        
        pos_label = torch.cat((label, correct))
        pred = self.model(sids, qids, cpts)
        bz_loss = self._loss_function(pred, pos_label)
        pos_grads = torch.autograd.grad(bz_loss, self.model.student_emb.parameters(),create_graph=False)
        pos_grad = torch.norm(pos_grads[0][sid]).item()
        # print(grads[0][0])
        del pos_grads
        
        self.model.student_emb.weight.data.copy_(original_weights)
        neg_label = torch.cat((label, wrong))
        pred = self.model(sids, qids, cpts)
        bz_loss = self._loss_function(pred, neg_label)
        neg_grads = torch.autograd.grad(bz_loss, self.model.student_emb.parameters(),create_graph=False)
        neg_grad = torch.norm(neg_grads[0][sid]).item()
        del neg_grads

        self.model.student_emb.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        pred = self.model(student_id, question_id, concepts_emb).item()
        # return pred * (pos_weights - original_weights).sum().tolist()+ \
        #     (1 - pred) * (neg_weights - original_weights).sum().tolist()
        res = pred * pos_grad + \
            (1 - pred) * neg_grad
        # print(pred, res)
        return res

    
    def expected_model_change(self, sid: int, qid: int,
                              adaptest_data: AdapTestDataset, pred_all: dict):
        """ get expected model change
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            float, expected model change
        """
        lr = self.config['learning_rate']
        device = self.config['device']
        
        for name, param in self.model.named_parameters():
            if 'student' not in name:
                param.requires_grad = False

        original_weights = self.model.student_emb.weight.data.clone()
        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        concepts = adaptest_data.concept_map[qid]
        concepts_emb = [0.] * adaptest_data.num_concepts
        for concept in concepts:
            concepts_emb[concept] = 1.0
        concepts_emb = torch.Tensor([concepts_emb]).to(device)
        correct = torch.LongTensor([1]).to(device)
        wrong = torch.LongTensor([0]).to(device)
        
        qid = adaptest_data.tested[sid]
        label = [adaptest_data.data[sid][q] for q in qid]
        concepts_embs = []
        for q in qid:
            concepts = adaptest_data.concept_map[q]
            tmp = [0.] * adaptest_data.num_concepts
            for concept in concepts:
                tmp[concept] = 1.0
            concepts_embs.append(tmp)
        concepts_embs = torch.Tensor(concepts_embs).to(device)
        label = [adaptest_data.data[sid][q] for q in qid]
        qid = torch.LongTensor(qid).to(device)
        tested_len = len(adaptest_data.tested[sid])
        sid = torch.LongTensor([sid] * tested_len).to(device)
        label = torch.LongTensor(label).to(device)
        sids = torch.cat((sid, student_id))
        qids = torch.cat((qid, question_id))
        cpts = torch.cat((concepts_embs, concepts_emb))
        #
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        l_loss = 10.0
        epoch = 0
        pos_label = torch.cat((label, correct))
        while True:
            pred = self.model(sids, qids, cpts)
            bz_loss = self._loss_function(pred, pos_label)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()
            self.model.apply_clipper()
            epoch += 1
            # print(bz_loss.item(), '-', l_loss, '=',
            #       abs(bz_loss.item() - l_loss))
            if abs(bz_loss.item() -
                   l_loss) < self._epsilon and bz_loss.item() < l_loss:
                break
            l_loss = bz_loss.item()
            # break
        
        pos_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        l_loss = 10.0
        epoch = 0
        neg_label = torch.cat((label, wrong))
        while True:
            pred = self.model(sids, qids, cpts)
            bz_loss = self._loss_function(pred, neg_label)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()
            self.model.apply_clipper()
            epoch += 1
            # print(bz_loss.item(), '-', l_loss, '=',
            #       abs(bz_loss.item() - l_loss))
            if abs(bz_loss.item() -
                   l_loss) < self._epsilon and bz_loss.item() < l_loss:
                break
            l_loss = bz_loss.item()
            # break

        neg_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        pred = self.model(student_id, question_id, concepts_emb).item()
        # return pred * (pos_weights - original_weights).sum().tolist()+ \
        #     (1 - pred) * (neg_weights - original_weights).sum().tolist()
        res = pred * torch.norm(pos_weights - original_weights).item() + \
            (1 - pred) * torch.norm(neg_weights - original_weights).item()
        # print(pred, res)
        return res
