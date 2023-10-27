

#%%
import numpy as np
import ot
import torch
import torch.nn as nn
import torch.optim as optim
from utils1 import *

from msda.barycenters import sinkhorn_barycenter
from ot.da import SinkhornL1l2Transport
from msda.utils import bar_zeros_initializer
from ot.utils import unif
from functools import partial
from otdd.pytorch.distance import DatasetDistance, FeatureCost
import os
from utils.options import get_basic_parser, add_fl_parser
from utils.data_utils import get_fed_data, split_dict_users, get_fed_domain_data
from utils.log_tools import generate_log_dir
from utils.global_utils import set_random_seed, resolve_args

import sys
from models.model_utils import get_model
from utils.dataset import Emnist
from utils.constants import DEFAULT_DATA_DIR as root_path
from utils.client_sampling import one_label, noniid_dir, noniid_label, iid
from algs.alg_tools import check_client_class_distribution
from utils.global_utils import draw_bars
from utils.data_utils import get_fed_data, get_normal_data
from utils.global_utils import load_pkl
from torchvision.models import resnet18

class localOT:
    def __init__(self, n_supp,n_localepoch, t_val=None, verbose=False,
                 get_int_list=False,
                 metric='sqeuclidean',server = False ):
        self.n_supp = n_supp  # n_supp of the interpolating measure
        self.t_val = t_val
        self.verbose = verbose
        self.n_localepoch = n_localepoch
        self.get_int_list = get_int_list
        self.metric = metric
        self.random_val_init = 1
        if self.metric == 'sqeuclidean':
            self.p = 2
        elif self.metric == 'euclidean':
            self.p = 1
        self.server= server

    def fit(self, xs, xt, wint_m,ws=None, wt=None, approx_interp=True,
            learn_support=False):
        self.approx_interp = approx_interp
        self.learn_support = learn_support
        dim = xs.shape[1]
        cost_diff = 0
        istensor = False
        if type(xs) == torch.Tensor:
            xs_ = torch.clone(xs)
            xs = xs.detach().numpy()
            xt_ = torch.clone(xt)
            xt = xt.detach().numpy()
            istensor = True
            if ws is not None:
                ws = ws.numpy().astype(np.float64)
            else:
                ws = np.ones((xs.shape[0],), dtype=np.float64) / xs.shape[0]

            if wt is not None:
                wt = wt.numpy().astype(np.float64)
            else:
                wt = np.ones((xt.shape[0],), dtype=np.float64) / xt.shape[0]
        # creating object for interpolation

        interp_G = InterpMeas(metric=self.metric, t_val=self.t_val, approx_interp=approx_interp,
                              learn_support=self.learn_support,server= self.server)
        # interp_H = InterpMeas(metric=self.metric, t_val=self.t_val, approx_interp=approx_interp,
        #                       learn_support=self.learn_support)



        list_cost = []
        list_int_m = []
        list_int_G = []

        for i in range(self.n_localepoch):
            if self.verbose:
                print(i)
            if self.get_int_list:
                list_int_m.append(int_m)

            # on client S , int_m就是实验中的gamma
            interp_G.fit(xs,xt, a=ws, b=wt)
            G, weight_G, cost_g,plan_g,unnomorlized_plan = interp_G.int_m, interp_G.weights, interp_G.cost, interp_G.plan,interp_G.unnomorlized_plan
            interp_G.int_init = G  # G就是interpolation measure

            # on client T
            # interp_H.fit(int_m, xt, a=weight_int_m, b=wt)
            # H, weight_H, cost_h, tplan= interp_H.int_m, interp_H.weights, interp_H.cost , interp_H.plan
            # interp_H.int_init = H
            # send costs, G and H to the server
            # on server
            # list_cost.append(cost_g + cost_h)
            # interp_m = interp_m.fit(H, G, a=weight_H, b=weight_G)  # 对公式9的计算
            # int_m, weight_int_m = interp_m.int_m, interp_m.weights
            # interp_m.int_init = int_m.copy()
            if self.get_int_list:
                list_int_G.append(G)
                # list_int_H.append(H)
        # preparing output for differentiable cost
        if istensor:
            eps = 1e-6

            Ms = euclidean_dist_torch(xs_.double(), torch.from_numpy(int_m).double()).pow(self.p)
            Mt = euclidean_dist_torch(torch.from_numpy(int_m).double(), xt_.double()).pow(self.p)

            with torch.no_grad():
                ns, nt = xs_.shape[0], xt_.shape[0]
                nm = int_m.shape[0]
                c = weight_int_m
                Ms_aux = Ms.detach().data.numpy()
                Mt_aux = Mt.detach().data.numpy()
                normMs = np.max(Ms_aux) if np.max(Ms_aux) > 1 else 1
                normMt = np.max(Mt_aux) if np.max(Mt_aux) > 1 else 1
                # print(np.sum(a),np.sum(b),np.sum(c),)
                gamma_s = ot.emd(ws, c, Ms_aux / normMs)
                planS = torch.from_numpy(gamma_s)
                gamma_t = ot.emd(c, wt, Mt_aux / normMt)
                planT = torch.from_numpy(gamma_t)
            cost = (torch.sum(Ms * planS) + eps) ** (1 / self.p) + \
                   (torch.sum(Mt * planT) + eps) ** (1 / self.p)
        else:
            nt = xt.shape[0]
            interp_G.fit(xs, xt)
            G, weight_G, cost_g, planS = interp_G.int_m, interp_G.weights, interp_G.cost, interp_G.plan
            # interp_H.fit(int_m, xt)
            # H, weight_G, cost_h, planT = interp_H.int_m, interp_H.weights, interp_H.cost, interp_H.plan
            # cost = cost_g + cost_h

        self.int_meas = int_m
        self.weights = weight_int_m
        self.list_cost = list_cost
        self.cost = cost_g
        self.plan = plan_g
        self.unnomorlized_plan = unnomorlized_plan
        # self.planS, self.planT = planS, planT
        # self.plan = planS @ planT * nt
        # self.list_int_meas = list_int_m
        # self.list_int_G = list_int_G
        # self.list_int_H = list_int_H
        self.eta = G
        self.localweight = weight_G
        if self.server == True:
            self.sourceplan = interp_G.plan
            self.inter_support = interp_G.int_m
        return self


def euclidean_dist_torch(x1, x2):
    x1p = x1.pow(2).sum(1).unsqueeze(1)
    x2p = x2.pow(2).sum(1).unsqueeze(1)
    prod_x1x2 = torch.mm(x1, x2.t())
    distance = x1p.expand_as(prod_x1x2) + \
        x2p.t().expand_as(prod_x1x2) - 2*prod_x1x2
    return torch.sqrt(distance)  # /x1.size(0)/x2.size(0)


def get_interp_measure(xs,xt,G0,t,thresh=1e-5):
    """ Get an exact interpolating measure between xs, xt 
    given the transport plan G0 and $t$.

    Args:
        xs (array): _description_
        xt (array): _description_
        G0 (_type_): _description_
        t (scalar real): _description_

    Returns:
        _type_: _description_
    """
    n_s, dim = xs.shape
    n_t = xt.shape[0]
    xsp = np.zeros((n_s+n_t+1, dim))
    xtp = np.zeros((n_s+n_t+1, dim))
    weights = np.zeros((n_s+n_t+1,))
    k = 0
    for i in range(xs.shape[0]):
        ind = np.where(G0[i, :]>thresh)[0]
        for j in range(len(ind)):
            xsp[k,:] = xs[i, :]
            xtp[k,:] = xt[ind[j], :]
            weights[k] = G0[i,ind[j]]
            k += 1
    # if k > n_s:
    #     print(k, n_s)
    #     pass
    xsp = xsp[:k, :]
    xtp = xtp[:k, :]
    xz = (1-t)*xsp + t*xtp
    weights = weights[:k]/np.sum(weights[:k])
    #print(xz.shape, weights.shape)
    
    return xz, weights

def interp_meas(X,Y,t_val=None,metric='sqeuclidean',approx_interp=True,
                a = None,b = None):
    """
    compute an the OT plan, cost and an interpolating measure
    works for squared euclidean distance
    everything is done on numpy
    
    return 
        * the interpolating measure
        * the OT cost between X and Y
        * the transport plan
    """
    nx, ny  = X.shape[0], Y.shape[0]
    p = 2 if metric=='sqeuclidean' else 1  
    if a is None:  
        a = np.ones((nx,),dtype=np.float64) / nx
    if b is None:
        b = np.ones((ny,),dtype=np.float64) / ny  
    # loss matrix
    M = ot.dist(X,Y,metric=metric) # squared euclidean distance 'default' # 计算两个分布的distance matrix
    # compute EMD
    norm = np.max(M) if np.max(M)>1 else 1
    G0 = ot.emd(a, b, M/norm)
    unnomorlized_GO = ot.emd(a, b, M)
    
    t = np.random.rand(1) if t_val==None else t_val
    #print('t',t)
    if approx_interp:
        Z = (1-t)*X + t*(G0*nx)@Y #对应公式(10) GO就是transportation plan( OT matrix )
        weights =  np.ones((nx,),dtype=np.float64) / nx
    else:
        Z, weights = get_interp_measure(X,Y,G0,t)
    cost = np.sum(G0*M)**(1/p)
    # 计算source到interpolating measure 的plan , 这里权重我用的都是uniform的
    # M_s = ot.dist(X, Z, metric=metric)
    # norm = np.max(M_s) if np.max(M_s) > 1 else 1
    G0_s =None

    return Z, weights, cost, G0,G0_s , unnomorlized_GO



    
def learn_interp_meas_support(xs,xt,n_supp=100,n_epoch=100,
                                t_val = None, lr= 0.01,p=2,
                                z_init=None, verbose=False,
                                a = None, b = None):
    """
    xs and xt are supposed to be numpy arrays
    
    p = 2 squared euclidean distance
    p = 1 euclidean distance
    
    output are numpy arrays 
    """
    if t_val is None:
        t_val = np.random.rand(1)[0] if t_val==None else t_val
    # TODO: add numpy transformation of xs and xt
    
    dim = xs.shape[1]
    c = np.ones(n_supp)/n_supp
    z = nn.Embedding(n_supp, dim)
    if z_init is not None:
        z.weight.data = torch.from_numpy(z_init)
    else:
        z.weight.data = torch.ones(n_supp, dim)
    z_init = z.weight.detach().clone()
    ns = xs.shape[0]
    nt = xt.shape[0]
    if a is None:  
        a = np.ones((ns,),dtype=np.float64) / ns
    if b is None:
        b = np.ones((nt,),dtype=np.float64) / nt 
    optimizer = optim.Adam(z.parameters(), lr=lr)
    s_list = []
    #print('learn',t_val)
    for i in range(n_epoch):
        # computing distance matrices 
        # between samples and interpolating measure

        Ms = euclidean_dist_torch(torch.from_numpy(xs).double(), z.weight.double()).pow(p)
        Mt = euclidean_dist_torch( z.weight.double(), torch.from_numpy(xt).double()).pow(p)
        with torch.no_grad():
            Ms_aux =  Ms.detach().data.numpy()
            Mt_aux =  Mt.detach().data.numpy()
            normMs = np.max(Ms_aux) if np.max(Ms_aux)>1 else 1
            normMt = np.max(Mt_aux) if np.max(Mt_aux)>1 else 1

            gamma_s = ot.emd(a, c, Ms_aux/normMs)
            gamma_s = torch.from_numpy(gamma_s)
            gamma_t = ot.emd(c,b, Mt_aux/normMt)
            gamma_t = torch.from_numpy(gamma_t)
        S = (1-t_val)*(torch.sum(Ms*gamma_s)).pow(1/p) + t_val*(torch.sum(Mt*gamma_t)).pow(1/p)
        z.zero_grad()
        S.backward()
        s_list.append(S.item())
        optimizer.step()
    cost = (torch.sum(Ms*gamma_s)).pow(1/p) + (torch.sum(Mt*gamma_t)).pow(1/p)
    z = z.weight.detach().numpy()

    # TODO: change plan to the full plan from X to Y
    return z, cost.detach().item(), [gamma_s,gamma_t], s_list



class InterpMeas:
    def __init__(self,metric='sqeuclidean',t_val=None,approx_interp=True,
                 learn_support=False,server = False ):
        self.metric = metric
        self.t_val = t_val
        self.n_supp = 100
        self.approx_interp = approx_interp

        #-- useful for learning support
        self.lr = 0.01
        self.n_epoch = 100
        self.int_init = None
        self.learn_support = learn_support
        self.server = server
        if self.server:
            self.learn_support = True
    def fit(self,X,Y, a=None, b=None):
        """_summary_

        Args:
            X (np_array): size nx x dim
            Y (np_array): _description_
            a (np_array, optional): _weights of the empirical distribution X . Defaults to None with equal weights.
            b (np_array, optional): _weights of the empirical distribution X . Defaults to None with equal weights.
            
        Returns:
            An InterpMeas object with the following attributes:
            int_m (np_array): size n_supp x dim
            weights (np_array): size n_supp x 1
            plan (np_array): size nx x nt
            loss_learn (list): list of the loss function during the learning of the support
            cost (float): cost of the optimal transport plan
        """
        t = np.random.rand(1)[0] if self.t_val==None else self.t_val
        if not self.learn_support:
            Z, weights, cost, G0, GOs,unnomorlized_GO= interp_meas(X,Y,t_val=t,metric=self.metric,
                                               a=a,b=b,approx_interp=self.approx_interp)
            self.t = t
            self.int_m = Z 
            self.cost = cost
            self.plan = G0
            self.weights = weights
            self.sourceplan = None
            self.unnomorlized_plan = unnomorlized_GO
        elif self.learn_support and self.server == True:
            t = np.random.rand(1)[0] if self.t_val==None else self.t_val
            p = 2 if self.metric=='sqeuclidean' else 1    
            Z, cost, gamma, s_list = learn_interp_meas_support(X,Y,n_supp=self.n_supp,n_epoch=self.n_epoch,
                                t_val = t, lr= self.lr, p=p,
                                z_init= self.int_init,
                                a=a, b = b)
            self.int_m = Z
            self.weights = np.ones((Z.shape[0],),dtype=np.float64) / Z.shape[0]
            self.cost =  cost
            # TODO: change plan to the full plan from X to Y
            self.plan = gamma
            self.loss_learn = s_list
        
        
        
        return self



if __name__ == '__main__':

    parser = add_fl_parser(get_basic_parser())
    args = parser.parse_args(['--exp_name', 'exp0_ot', '--dataset', 'mnist', '--split', 'iid'])

    args, setting_name, method_name = resolve_args(root='.', args=args)
    if not args.save_clients:
        save_dir = os.path.join(args.save_root, args.project_name, args.exp_name, setting_name, method_name)
    else:
        save_dir = os.path.join(args.save_root, args.project_name, args.exp_name, setting_name, '_grad', method_name)
    args.save_dir = generate_log_dir(path=save_dir, is_use_tb=False, has_timestamp=args.timestamp)
    setting_dir = os.path.join(args.save_root, args.project_name, args.exp_name, setting_name)
    # # ---------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # set random seed
    set_random_seed(seed=args.seed)

    if args.dataset in ["digits"]:
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_fed_domain_data(args.dataset,
                                                                                             args.bs_train,
                                                                                             args.data_frac)
    else:
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_fed_data(setting_dir, args.dataset,
                                                                                      args.split, args.num_users,
                                                                                      args.dir_alpha,
                                                                                      args.clsnum_peruser,
                                                                                      args.imb_alpha, args.data_frac)


    sys.path.append("/Users/muz1lee/Desktop/代码/fedselect/")

    path = "/Users/muz1lee/Desktop/代码/fedselect/exp0_ot/cifar10_dFr1.0_nUs10_iid_f1.0_e100_lEp5_s1/dict_users.pkl"
    dict_users = load_pkl(path)
    dataset = get_normal_data('mnist')
    dataset_train, dataset_test = dataset['train'], dataset['test']
    results = check_client_class_distribution(dataset_train.targets, dict_users[0])
    results = results.iloc[:, 4:]
    draw_bars(results.to_numpy(), title="mnist")
    df = check_client_class_distribution(dataset_train.targets, dict_users[0])
    print(df)

    # Embed using a pretrained (+frozen) resnet
    embedder = resnet18(pretrained=True).eval()
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False

    # Here we use same embedder for both datasets
    feature_cost = FeatureCost(src_embedding=embedder,
                               src_dim=(3, 32, 32),
                               tgt_embedding=embedder,
                               tgt_dim=(3, 32, 32),
                               p=2,
                               device='-1')

    ot_results = []
    for client_idx in range(10):
        # for client_idx in [3]:
        local_train = DatasetSplit(dataset_train, dict_users_train[client_idx])
        trainloader = torch.utils.data.DataLoader(local_train, batch_size=32, shuffle=False)
        dist = DatasetDistance(trainloader, dataset_test,
                               inner_ot_method='exact',
                               debiased_loss=True,
                               feature_cost=feature_cost,
                               sqrt_method='spectral',
                               sqrt_niters=10,
                               precision='single',
                               p=2, entreg=1e-1,
                               device='-1')

        a = dist._get_label_stats()