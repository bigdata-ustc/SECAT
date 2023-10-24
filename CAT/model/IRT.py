import os
import time
import copy
from tkinter.messagebox import NO
import vegas
import logging
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from math import exp as exp
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from scipy import integrate
import torch.nn.functional as F
import random
from copy import deepcopy

from CAT.model.abstract_model import AbstractModel
from CAT.dataset import AdapTestDataset, TrainDataset, Dataset


class IRT(nn.Module):
    def __init__(self, num_students, num_questions, num_dim, a_range=None):
        # num_dim: IRT if num_dim == 1 else MIRT
        super().__init__()
        self.num_dim = num_dim
        self.num_students = num_students
        self.num_questions = num_questions
        self.theta = nn.Embedding(self.num_students, self.num_dim)
        self.alpha = nn.Embedding(self.num_questions, self.num_dim)
        self.beta = nn.Embedding(self.num_questions, 1)
        self.a_range = a_range
        torch.manual_seed(50)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_ids, question_ids):
        theta = self.theta(student_ids)
        alpha = self.alpha(question_ids)
        beta = self.beta(question_ids)
        # pred = (alpha * theta).sum(dim=1, keepdim=True) + beta
        # if self.a_range is not None:
        #     alpha = self.a_range * torch.sigmoid(alpha)
        # else:
        #     alpha = F.softplus(alpha)
        # pred = (alpha * theta).sum(dim=1, keepdim=True) + beta
        pred = 1.702 * (alpha * (theta - beta))
        pred = torch.sigmoid(pred)
        return pred


class IRTModel(AbstractModel):
    def __init__(self, precision=6, **config):
        super().__init__()
        self.config = config
        self.model = None
        self._epsilon = float("1e-" + str(precision))

    @property
    def name(self):
        return 'Item Response Theory'

    def init_model(self, data: Dataset):
        self.model = IRT(data.num_students, data.num_questions,
                         self.config['num_dim'])

    def train(self, train_data: TrainDataset, test_data=None, log_step=20):
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
            for student_ids, question_ids, _, labels in tqdm(
                    train_loader, f'Epoch {ep}'):
                # for cnt, (student_ids, question_ids, _, labels) in enumerate(train_loader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device).float()
                pred = self.model(student_ids, question_ids).view(-1)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                # loss += bz_loss.data.float()
                loss.append(bz_loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (ep, float(np.mean(loss))))
            # if cnt % log_step == 0:
            #     logging.info('Epoch [{}] Batch [{}]: loss={:.5f}'.format(ep, cnt, loss / cnt))
            if test_data is not None:
                test_loader = data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=True)
                self.eval(test_loader, device=self.config['device'])

    def eval(self, adaptest_data: AdapTestDataset, device='cuda:5'):
        # data = adaptest_data.data
        self.model.to(device)

        with torch.no_grad():
            self.model.eval()
            y_pred = []
            y_true = []
            y_label = []
            for student_ids, question_ids, _, labels in tqdm(
                    adaptest_data, "evaluating"):
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                pred: torch.Tensor = self.model(student_ids,
                                                question_ids).view(-1)
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
        # for qid in qids:
        sid_t = torch.LongTensor([sid] * len(qids)).to(device)
        qid_t = torch.LongTensor(list(qids)).to(device)
        pred_t = self.model(sid_t, qid_t).view(-1).tolist()
        # if random.random()<1:
        if MonteCarlo:
            pred_t = [1 if random.random() < p else 0 for p in pred_t]
        else:
            pred_t = [0 if p < 0.5 else 1 for p in pred_t]
        res = list(zip([sid] * len(qids), list(qids), pred_t))
        # res.append([sid, qid, pred])
        return res

    def adaptest_save(self, path, save_theta=False):
        """
        Save the model. Only save the parameters of questions(alpha, beta)
        """
        model_dict = self.model.state_dict()
        if save_theta == False:
            model_dict = {
                k: v
                for k, v in model_dict.items() if 'theta' not in k
            }
        # else:
        #     model_dict = {k: v for k, v in model_dict.items()
        #                 if 'alpha' in k or 'beta' in k or 'theta' in k}
        torch.save(model_dict, path)

    def adaptest_load(self, path):
        """
        Reload the saved model
        """
        self.model.to(self.config['device'])
        self.model.load_state_dict(
            torch.load(path, map_location=torch.device(self.config['device'])),
            strict=False,
        )

    def adaptest_update(self,
                        sid,
                        qid,
                        adaptest_data: AdapTestDataset,
                        update_lr=None,
                        optimizer=None,
                        scheduler=None):
        """
        Update CDM with tested data
        """
        # self.model.eval()
        # for name, param in self.model.named_parameters():
        #     print(name,param.requires_grad)
        if update_lr is None:
            lr = self.config['learning_rate']
        else:
            lr = update_lr
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)
        label = adaptest_data.data[sid][qid]
        sid = torch.LongTensor([sid]).to(device)
        qid = torch.LongTensor([qid]).to(device)
        label = torch.LongTensor([label]).to(device).float()
        for _ in range(epochs):
            pred = self.model(sid, qid).view(-1)
            bz_loss = self._loss_function(pred, label)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()
        # print('theta',self.get_theta(sid))
        # print('a:',self.get_alpha(qid))
        # print('b:',self.get_beta(qid))
        # print(label.tolist(),' ', pred.tolist())
        # print('================================')

    def adaptest_update1(self,
                         sid,
                         qid,
                         adaptest_data: AdapTestDataset,
                         update_lr=None,
                         optimizer=None,
                         scheduler=None):
        """
        Update CDM with tested data
        """
        # self.model.eval()
        # for name, param in self.model.named_parameters():
        #     print(name,param.requires_grad)
        if update_lr is None:
            lr = self.config['learning_rate']
        else:
            lr = update_lr
        device = self.config['device']
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)

        qid = adaptest_data.tested[sid]
        label = [adaptest_data.data[sid][q] for q in qid]
        qid = torch.LongTensor(qid).to(device)
        tested_len = len(adaptest_data.tested[sid])
        sid = torch.LongTensor([sid] * tested_len).to(device)
        label = torch.LongTensor(label).to(device).float()
        l_loss = -10.0
        epoch = 0
        # if len(label)==10:
        #     print(label)
        while True:
            pred = self.model(sid, qid).view(-1)
            bz_loss = self._loss_function(pred, label)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()
            epoch += 1
            if abs(bz_loss.item() - l_loss) < self._epsilon:
                # print(bz_loss.item(),'-',l_loss,'=',abs(bz_loss.item()-l_loss))
                break
            l_loss = bz_loss.item()
        # print(bz_loss.item(), bz_loss)
        # print('theta', self.get_theta(sid[0]))
        # print(epoch)
        pass
        # print('a:', self.get_alpha(qid[-1]))
        # print('b:', self.get_beta(qid[-1]))
        # print('theta', self.get_theta(sid[-1]))
        # print(label[-1].tolist(), ' ', pred[-1].tolist())
        # print('================================')

    def evaluate(self, sid, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        device = self.config['device']
        untested = adaptest_data.untested[sid]
        # untested = adaptest_data.data[sid].keys()

        real = []
        pred = []
        with torch.no_grad():
            self.model.eval()
            # for sid in data:
            student_ids = [sid] * len(untested)
            # student_ids = [sid] * len(data[sid])
            # question_ids = list(data[sid].keys())
            question_ids = list(untested)
            real += [data[sid][qid] for qid in question_ids]
            student_ids = torch.LongTensor(student_ids).to(device)
            question_ids = torch.LongTensor(question_ids).to(device)
            output = self.model(student_ids, question_ids).view(-1)
            pred += output.tolist()
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
        if len(np.unique(real)) == 1:  # bug in roc_auc_score
            auc = accuracy_score(real, pred_label)
        else:
            auc = roc_auc_score(real, pred)
        # print(classification_report(real, pred_label, digits=4))
        # return classification_report(real, pred_label, digits=4)
        return {
            'acc': acc,
            'auc': auc,
            'cov': 0,
            # 'cov': cov,
        }

    def get_pred(self, adaptest_data: AdapTestDataset):
        """
        Returns:
            predictions, dict[sid][qid]
        """
        data = adaptest_data.data
        device = self.config['device']

        pred_all = {}

        with torch.no_grad():
            self.model.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                output = self.model(student_ids,
                                    question_ids).view(-1).tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.model.train()

        return pred_all

    def _loss_function(self, pred, real):
        return -(real * torch.log(0.0001 + pred) +
                 (1 - real) * torch.log(1.0001 - pred)).mean()

    def get_alpha(self, question_id):
        """ get alpha of one question
        Args:
            question_id: int, question id
        Returns:
            alpha of the given question, shape (num_dim, )
        """
        device = self.config['device']
        qid = torch.LongTensor([question_id]).to(device)
        alpha = self.model.alpha(qid)
        # if self.model.a_range is not None:
        #     alpha = self.model.a_range * torch.sigmoid(alpha)
        # else:
        #     alpha = F.softplus(alpha)
        # return alpha.clone().detach().cpu()[0]
        return np.array(alpha.clone().detach().cpu()[0])
        # return self.model.alpha.weight.data.cpu().numpy()[question_id]

    def get_beta(self, question_id):
        """ get beta of one question
        Args:
            question_id: int, question id
        Returns:
            beta of the given question, shape (1, )
        """
        return self.model.beta.weight.data.cpu().numpy()[question_id]

    def get_theta(self, student_id):
        """ get theta of one student
        Args:
            student_id: int, student id
        Returns:
            theta of the given student, shape (num_dim, )
        """
        return self.model.theta.weight.data.cpu().numpy()[student_id]

    def get_kli(self, student_id, question_id, n, pred_all):
        """ get KL information
        Args:
            student_id: int, student id
            question_id: int, question id
            n: int, the number of iteration
        Returns:
            v: float, KL information
        """
        if n == 0:
            return np.inf
        device = self.config['device']
        dim = self.model.num_dim
        sid = torch.LongTensor([student_id]).to(device)
        qid = torch.LongTensor([question_id]).to(device)
        theta = self.get_theta(sid)  # (num_dim, )
        alpha = self.get_alpha(qid)  # (num_dim, )
        beta = self.get_beta(qid)  # float value
        pred_estimate = pred_all[student_id][question_id]

        def kli(x):
            """ The formula of KL information. Used for integral.
            Args:
                x: theta of student sid
            """
            if type(x) == float:
                x = np.array([x])
            pred = 1.702 * (alpha * (x - beta))
            # pred = np.matmul(alpha.T, x) + beta
            pred = 1 / (1 + np.exp(-pred))[0]
            q_estimate = 1 - pred_estimate
            q = 1 - pred
            # if pred == 0. or q == 0.:
            #     return 0
            # else:
            return pred_estimate * np.log(pred_estimate / pred) + \
                q_estimate * np.log(q_estimate / q)

        c = 1
        boundaries = [[theta[i] - c / np.sqrt(n), theta[i] + c / np.sqrt(n)]
                      for i in range(dim)]
        # KLI
        v, err = integrate.quad(kli, boundaries[0][0], boundaries[0][1])
        return v

    def get_fisher(self, student_id, question_id, pred_all):
        """ get Fisher information
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            fisher_info: matrix(num_dim * num_dim), Fisher information
        """
        device = self.config['device']
        qid = torch.LongTensor([question_id]).to(device)
        alpha = self.get_alpha(qid)
        pred = pred_all[student_id][question_id]
        q = 1 - pred
        fisher_info = (q * pred * (alpha * alpha.T))[0]
        return fisher_info

    def expected_model_change1(self, sid: int, qid: int,
                              adaptest_data: AdapTestDataset, pred_all: dict):
        device = self.config['device']

        for name, param in self.model.named_parameters():
            if 'theta' not in name:
                param.requires_grad = False
        original_weights = self.model.theta.weight.data.clone()
        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        correct = torch.LongTensor([1]).to(device).float()
        wrong = torch.LongTensor([0]).to(device).float()

        qid = adaptest_data.tested[sid]
        label = [adaptest_data.data[sid][q] for q in qid]
        qid = torch.LongTensor(qid).to(device)
        tested_len = len(adaptest_data.tested[sid])
        sids = torch.LongTensor([sid] * (tested_len+1)).to(device)
        label = torch.LongTensor(label).to(device).float()
        qids = torch.cat((qid, question_id))

        pos_label = torch.cat((label, correct))
        pos_pred = self.model(sids, qids).view(-1)
        bz_loss = self._loss_function(pos_pred, pos_label)
        pos_grads = torch.autograd.grad(bz_loss, self.model.theta.parameters(),create_graph=False)
        pos_grad = abs(pos_grads[0][sid].item())
        # print(grads[0][0])
        del pos_grads

        self.model.theta.weight.data.copy_(original_weights)
        neg_label = torch.cat((label, wrong))
        neg_pred = self.model(sids, qids).view(-1)
        bz_loss = self._loss_function(neg_pred, neg_label)
        neg_grads = torch.autograd.grad(bz_loss, self.model.theta.parameters(),create_graph=False)
        # print(grads[0][0])
        neg_grad = abs(neg_grads[0][sid].item())
        del neg_grads

        self.model.theta.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        pred = self.model(student_id, question_id).item()
        # print(pos_grad,neg_grad,(pos_grad-neg_grad)>0)
        # if pred>0.5:
        #     return pred * pos_grad
        # else:
        #     return (1 - pred) * neg_grad
        res = pred * pos_grad + \
            (1 - pred) * neg_grad
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
            if 'theta' not in name:
                param.requires_grad = False
        original_weights = self.model.theta.weight.data.clone()
        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        correct = torch.LongTensor([1]).to(device).float()
        wrong = torch.LongTensor([0]).to(device).float()

        #
        qid = adaptest_data.tested[sid]
        label = [adaptest_data.data[sid][q] for q in qid]
        qid = torch.LongTensor(qid).to(device)
        tested_len = len(adaptest_data.tested[sid])
        sids = torch.LongTensor([sid] * (tested_len+1)).to(device)
        label = torch.LongTensor(label).to(device).float()
        qids = torch.cat((qid, question_id))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        l_loss = -10.0
        epoch = 0
        pos_label = torch.cat((label, correct))
        while True:
            pos_pred = self.model(sids, qids).view(-1)
            bz_loss = self._loss_function(pos_pred, pos_label)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()
            epoch += 1
            if abs(bz_loss.item() - l_loss) < self._epsilon:
                # print(bz_loss.item(),'-',l_loss,'=',abs(bz_loss.item()-l_loss))
                break
            l_loss = bz_loss.item()
            # break
        # print('=========')
        pos_weights = self.model.theta.weight.data.clone()
        self.model.theta.weight.data.copy_(original_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        l_loss = -10.0
        epoch = 0
        neg_label = torch.cat((label, wrong))
        while True:
            optimizer.zero_grad()
            neg_pred = self.model(sids, qids).view(-1)
            bz_loss = self._loss_function(neg_pred, neg_label)
            # print(bz_loss)
            # original_loss = bz_loss
            bz_loss.backward()
            # print(bz_loss-original_loss)
            optimizer.step()
            epoch += 1
            if abs(bz_loss.item() - l_loss) < self._epsilon:
                break
            l_loss = bz_loss.item()
            # break
        # print('neg_loss:', epoch, ' ', l_loss)
        neg_weights = self.model.theta.weight.data.clone()
        self.model.theta.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        pred = self.model(student_id, question_id).item()
        # if pred>0.5:
        #     return pred * torch.norm(pos_weights - original_weights).item()
        # else:
        #     return (1 - pred) * torch.norm(neg_weights - original_weights).item()
        res = pred * torch.norm(pos_weights - original_weights).item() + \
            (1 - pred) * torch.norm(neg_weights - original_weights).item()
        # print(abs(pred-0.5), res)
        # model.model.get_knowledge_status(sid)
        # model.model.get_exer_params(adaptest_data.untested[sid][0])
        return res