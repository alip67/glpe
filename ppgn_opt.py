import os
import PPGN.models
import torch
import torch.nn.functional as F
from PPGN.models.base_model import BaseModel
import torch
import torch.nn as nn
import PPGN.layers.layers as layers
import PPGN.layers.modules as modules

import wandb

import os.path as osp
import torch
# from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
# from gnn import GNN
from torch import Tensor

from tqdm import tqdm
import argparse
import time
import numpy as np
# import geotorch
import geoopt

import scipy.io


### importing OGB
# from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.utils import to_networkx, to_dense_adj
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import Evaluator

import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
from torch_geometric.loader import DataLoader

import torch
import argparse
from timeit import default_timer as timer
import torch
import torch.nn.functional as F
import torch.nn as nn
from numba import jit

from typing import Any, Optional
from torch_geometric.data import Data

import networkx as nx

def get_orthonormal_eigvec(eigval, eigvec):
   #We transform our eigenvectors into an orthonormalbasis (next 4 cells) such that it is in the Stiefel manifold

    eps = 2.220446049250313e-6
    i = 0
    k= 0 
    liste = []
    for j in range(eigval.size):
        if not liste:
            liste.append(i)
        elif round(eigval[j-1],4)==round(eigval[j],4):
            #liste.append(i)
            k = k+1
        else: 
            i = i+1
            k =k+1
            liste.append(k)
    liste.append(eigvec.shape[1])

    ll = []
    siz = 0
    for i in range(1,len(liste)):
        #print(i, liste[i-1], liste[i], eigvec[:,liste[i-1]: liste[i]].shape)
        siz = siz + eigvec[:,liste[i-1]: liste[i]].shape[1]
        ll.append(eigvec[:,liste[i-1]: liste[i]])
    
    lll = []
    for i in ll:
        if np.linalg.norm(np.matmul(np.transpose(i), i)) > eps:
            lll.append(scipy.linalg.orth(i))
        else: lll.append(i)

    hi = lll[0]
    for i in range(0,len(lll)-1):
        hi = np.concatenate((hi, lll[i+1]), axis=1)

    return hi

def my_loss(F, XX, pp):
    """
    X: (N x N)-tensor with adjacency matrix
    F: (N x F)-tensor with eigenvectors
    """
    #mat = F.squeeze()
    #matt = torch.diagonal(mat, dim1=-2, dim2=-1)
    #mattt = matt.transpose(-1,-2)   
    #mattt,_ = torch.qr(mattt)
    #f = mattt
    f = F
    n = f.shape[0]
    FF = f.repeat(1,n)
    FF = FF.reshape(n,n,f.shape[1])
    FFF = torch.norm(f, pp,dim=0)
    FFF = torch.pow(FFF,pp)
    FF = FF.transpose(2,0)
    GG = FF.transpose(1,2)
    A = XX.unsqueeze(dim=1)
    KK = FF - GG
    KKK = KK.unsqueeze(dim=-1)
    KKK = torch.pow(torch.abs(KKK),pp)
    KKK = KKK.type(torch.float64)
    A = A.type(torch.float64)
    LL = torch.matmul(A, KKK)
    FFF = torch.pow(FFF,-1)
    FFF.unsqueeze_(-1)
    FFF.unsqueeze_(-1)
    FFF.unsqueeze_(-1)
    FFF = FFF.repeat(1,n,1,1)
    b = torch.matmul(LL.float(),FFF)
    b = torch.sum(b)
    return b

def make_2d_graph(m, n, periodic=False, return_pos=False):
    network = nx.grid_2d_graph(m, n, periodic=False, create_using=None)
    matrix = nx.linalg.graphmatrix.adjacency_matrix(network).todense()
    matrix = np.array(matrix).astype(float)
    return matrix

def get_graph_props(A, normalize_L='none', shift_to_zero_diag=False):
    ran = range(A.shape[0])

    D = np.zeros_like(A)
    D[ran, ran] = np.abs(np.sum(A, axis=1) - A[ran, ran])
    L = D - A

    if (normalize_L is None) or (normalize_L=='none') or (normalize_L == False):
        pass
    elif (normalize_L == 'inv'):
        Dinv = np.linalg.inv(D)
        L = np.matmul(Dinv, L)  # Normalized laplacian
    elif (normalize_L == 'sym'):
        Dinv = np.sqrt(np.linalg.inv(D))
        L = np.matmul(np.matmul(Dinv, L), Dinv)
    elif (normalize_L == 'abs'):
        L = np.abs(L)
    else:
        raise ValueError('unsupported normalization option')

    eigval, eigvec = np.linalg.eigh(L)
    eigval =  np.real(eigval)
    # eigidx = np.argsort(eigval)[::-1]
    eigidx = np.argsort(eigval)
    eigval = eigval[eigidx]
    eigvec = eigvec[:, eigidx]


    L_inv = np.linalg.pinv(L)

    if shift_to_zero_diag:
        L_inv_diag = L_inv[np.eye(L.shape[0])>0]
        L_inv = (L_inv - L_inv_diag[:, np.newaxis])

    return D, L, L_inv, eigval, eigvec

class BaseModel(nn.Module):
    def __init__(self, K, depth,hidden_dim):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()

        self.use_new_suffix = True
        block_features = []
        # List of number of features in each regular block

        for i in range(depth):
            block_features.append(hidden_dim)
        block_features.append(K)
        
        original_features_num = K + 1  # Number of features of the input
        print(K)

        # First part - sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = modules.RegularBlock(last_layer_features, next_layer_features)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

        # Second part
        self.fc_layers = nn.ModuleList()

    def forward(self, input):
        x = input
        scores = torch.tensor(0, device=input.device, dtype=x.dtype)

        for i, block in enumerate(self.reg_blocks):

            x = block(x)

        mat = x.squeeze()
        matt = torch.diagonal(mat, dim1=-2, dim2=-1)
        mattt = matt.transpose(-1,-2)   
        mattt,_ = torch.linalg.qr(mattt)
        
        return mattt

def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))

    
def training_loop1(model, optimizer, sched, W, epochs, X, p):
    "Training loop for torch model."
    wandb.watch(model, my_loss, log="all", log_freq=1)

    losses = []
    values = []
    for i in range(epochs):
        preds = model(W)
        loss = my_loss(preds, X, p)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)
        values.append(preds)
        # Where the magic happens
        wandb.log({"epoch": i, "loss": loss})
        #print(f"Loss after " + str(i) + f": {loss:.3f}")
    return losses, values
    
wandb.init(project='gpt3')

def main(cmd_opt):
    opt = cmd_opt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_eigs = opt['num_eigs']   #gives the dimension of the embedding or/ the number of eigenvectors we calculate
    p = opt['p_laplacian']
    epochs = opt['epochs']

    grid_sizes = [(16,4)]
    
    A = make_2d_graph(grid_sizes[0][0],grid_sizes[0][1], periodic=False) 
    print(A.shape)
    D, L, L_inv, eigval,eigvec = get_graph_props(A,normalize_L='none')

    init_eigs = get_orthonormal_eigvec(eigval,eigvec)
    
    
    #Preprocessing
   
    A_ = torch.tensor(A)
    F_ = torch.tensor(init_eigs)[:,:num_eigs]
    F_ = F_.transpose(-1,0)
    
    F=torch.diag(F_[0,:])
    F=F.unsqueeze(dim=0)
    for i in range(1,F_.shape[0]):
        FF = torch.diag(F_[i,:])
        FF=FF.unsqueeze(dim=0)
        F=torch.cat((F, FF), 0)
    AA = A_.unsqueeze(dim=0)
    inp = torch.cat((AA,F),0)
    inp = inp.unsqueeze(dim=0)
    
    # instantiate input
    X=torch.tensor(A).to(device)

    # instantiate model
    num_layers = opt['num_layers']
    hidden_channels = opt['hidden_channels']

    bm = BaseModel(num_eigs, num_layers, hidden_channels).to(device)
    
    # instantiate optimizer
    opt_name = opt['optimizer']
    lr = opt['lr']
    weight_decay = opt['decay']


    optimizer = get_optimizer(opt_name, bm.parameters(), lr=lr, weight_decay=weight_decay)
    
    #optimizer = torch.optim.SGD(bm.parameters(), lr=lr)
    my_lr_scheduler = None #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    # 1. Start a W&B run
    wandb.init(project='ppgn_opt')

    # 2. Save model inputs and hyperparameters
    config = opt
    config = wandb.config

    # Model training here
    
    #training
    start = timer()
    bm.train()
    losses, values = training_loop1(bm, optimizer,my_lr_scheduler,inp.float().to(device), epochs, X, p) 
    end_loss=[losses[-1]]
    end = timer()
    print(end - start, " second")
    print("Final Loss: ", end_loss)
    # 3. Log metrics over time to visualize performance
    wandb.log({"loss": end_loss})

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    #Setting args
    parser.add_argument('--p_laplacian', type=float, default=1,
                        help='the value for p-Laplacian (default: 1)')
    parser.add_argument('--num_eigs', type=int, default=5,
                        help='number of eigenvectors (default: 5)')

    #PPGN args
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=16)

    
    #Optimizer args
    parser.add_argument('--epochs', type=int, default=10, 
                        help='number of epochs (default: 50)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')


    args = parser.parse_args()

    opt = vars(args)

    main(opt)

