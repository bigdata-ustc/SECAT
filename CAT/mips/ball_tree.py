import numpy as np
import random
import heapq

def search_metric_tree_k(candidates, q, T):
    min_candidates = min(candidates.values())
    if min_candidates >= np.dot(q, T.center)+T.radius*(((q**2).sum())**0.5):
        return 0

    if T.is_leaf():
        for key,val in T.dict.items():
            if np.dot(val[1],q)<=min_candidates:
                continue
            for k1,v1 in candidates.items():
                if v1 == min_candidates:
                    del candidates[k1]
                    candidates[key] = np.dot(val[1],q)
                    min_candidates = min(candidates.values())
                    break
        return 1
    else:
        I_l =np.dot(T.left.center,q)
        I_r =np.dot(T.right.center,q)
        if I_l<=I_r:
            cnt_l = search_metric_tree_k(candidates, q, T.right)
            cnt_r =search_metric_tree_k(candidates, q, T.left)
        else:
            cnt_l = search_metric_tree_k(candidates, q, T.left)
            cnt_r = search_metric_tree_k(candidates, q, T.right)
        return cnt_l+cnt_r

def search_metric_tree_c(res, tested_set, q, T):
    if res['quantity'] >= np.dot(q, T.center)+T.radius*(((q**2).sum())**0.5):
        return 0
    if T.is_leaf():
        for key, val in T.dict.items():
            if (key in tested_set) or (np.dot(np.array(val[1]),q)<=res['quantity']):
                continue
            res['qid'] = key
            res['quantity'] = np.dot(np.array(val[1]),q)
            res['leaves']  = T.dict
            # res={'qid':key,'quantity':np.dot(np.array(val[1]),q)}
            # for k1,v1 in candidates.items():
            #     if v1 == min_candidates:
            #         del candidates[k1]
            #         candidates[key] = np.dot(val[1],q)
            #         min_candidates = min(candidates.values())
            #         break
        return len((T.dict.keys()))
    else:
        I_l =np.dot(T.left.center,q)
        I_r =np.dot(T.right.center,q)
        if I_l<=I_r:
            cnt_l = search_metric_tree_c(res, tested_set, q, T.right)
            cnt_r = search_metric_tree_c(res, tested_set, q, T.left)
        else:
            cnt_l = search_metric_tree_c(res, tested_set, q, T.left)
            cnt_r = search_metric_tree_c(res, tested_set, q, T.right)
        return cnt_l + cnt_r

def search_metric_tree(res, tested_set, q, T):
    if res['quantity'] >= np.dot(q, T.center)+T.radius*(((q**2).sum())**0.5):
        return 0
    if T.is_leaf():
        for key, val in T.dict.items():
            if (key in tested_set) or (np.dot(np.array(val[1]),q)<=res['quantity']):
                continue
            res['qid'] = key
            res['quantity'] = np.dot(np.array(val[1]),q)
            res['leaves']  = T.dict
            # res={'qid':key,'quantity':np.dot(np.array(val[1]),q)}
            # for k1,v1 in candidates.items():
            #     if v1 == min_candidates:
            #         del candidates[k1]
            #         candidates[key] = np.dot(val[1],q)
            #         min_candidates = min(candidates.values())
            #         break
        return 1
    else:
        I_l =np.dot(T.left.center,q)
        I_r =np.dot(T.right.center,q)
        if I_l<=I_r:
            cnt_l = search_metric_tree(res, tested_set, q, T.right)
            cnt_r = search_metric_tree(res, tested_set, q, T.left)
        else:
            cnt_l = search_metric_tree(res, tested_set, q, T.left)
            cnt_r = search_metric_tree(res, tested_set, q, T.right)
        return cnt_l + cnt_r

def get_leaves(T,res):
    if T.is_leaf():
        res.append(T.dict)
    else:
        get_leaves(T.left,res)
        get_leaves(T.right,res)

class BallTree(object):
    def __init__(self,item_pool,dissimilarity_partition,threshold=20):
        random.seed(1729) 
        self.left=None
        self.right=None
        self.radius=None
        self.threshold=threshold
        self.dissimilarity_partition = dissimilarity_partition
        # self.item_label=item_label
        self.make_ball(item_pool)
    
    def make_ball(self, item_pool):
        trait = [i[1] for i in item_pool.values()]
        self.data = np.array(trait)
        # self.data = np.array(list(item_pool.values()))
        self.dict = item_pool
        if len(self.data)==0:
            return
        self.center = self.get_center()
        self.radius = self.get_radius()
        if len(self.data)<=self.threshold:
            return 
        w,b = self.make_metric_tree_split(item_pool)
        items_left={}
        items_right={}
        for key,(theta,val) in self.dict.items():
            if np.dot(val, w) + b <= 0:
                items_left[key]=(theta,val)
                # self.item_label
                # label_left.append()
            else:
                items_right[key]=(theta,val)
        self.left=BallTree(items_left, self.dissimilarity_partition)
        self.right=BallTree(items_right, self.dissimilarity_partition)
    
    def is_leaf(self):
        return self.left==None  and self.right==None
     
    def get_center(self):
        return np.mean(self.data, axis=0)
    
    def get_l2_list(self,point):
        diff_matrix = self.data-np.expand_dims(point,0).repeat(len(self.data),axis=0)
        return (diff_matrix**2).sum(axis=1)

    def get_radius(self):
        return (self.get_l2_list(self.center).max())**0.5
        
    def make_metric_tree_split(self,item_pool): 
        thetas = [theta for q,(theta,trait) in item_pool.items()]
        max_threshold = min(heapq.nlargest(int(self.threshold/2), thetas))
        min_threshold = max(heapq.nsmallest(int(self.threshold/2),thetas))
        
        if max_threshold>min_threshold and self.dissimilarity_partition:
        # if max_threshold>min_threshold and False:
            As=[]
            Bs=[]
            for q,(theta,trait) in item_pool.items():
                if theta>=max_threshold:
                    As.append(trait)
                if theta<=min_threshold:
                    Bs.append(trait)
            As = np.array(As)
            A = As.mean(axis=0)
            Bs = np.array(Bs)
            B = Bs.mean(axis=0)
        else:
            idx = random.randint(0,len(self.data)-1)
            x = self.data[idx] 
            A =  self.data[np.argmax(self.get_l2_list(x))]
            B =  self.data[np.argmax(self.get_l2_list(A))]
        w = B-A
        b = -1/2*((B**2).sum()-(A**2).sum())
        return (w,b)
    
    