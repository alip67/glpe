import os.path as osp
import torch
# from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
# from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np
import geotorch
import geoopt

### importing OGB
# from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

# from torch_geometric.datasets.zinc import ZINC

import torch
import argparse
from timeit import default_timer as timer
import torch
import torch.nn.functional as F
import torch.nn as nn
# from numba import jit

#import torch_geometric.transforms as T
# from torch_geometric.nn import GCNConv

import numpy as np
import networkx as nx
import os
import scipy.io
import shutil
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from copy import copy
import seaborn as sns
sns.set_style("whitegrid")

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

class Model_RGD(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, D, p, n, K, ball):
        
        super().__init__()
        # initialize weights with eigenvectors
        self.initeigv = D.clone() #normally calculate here from the adjacency matrix, just testing now
        self.p = p
        self.n= n
        self.K = K

        self.ball = ball
        self.plane_shape = geoopt.utils.size2shape(n)
        self.num_planes = K

        # Create manifold parameters
        self.weight = geoopt.ManifoldParameter(
            torch.empty(n, K), manifold=self.ball
        )
        #self.points = nn.Parameter(self.initeigv.clone())
        #geotorch.grassmannian(self, "weight") 
        #Stiefel = self.parametrizations.weight[0]
        #self.weight = Stiefel.sample()
        #self.linear = nn.Linear(n, K)
        #self.linear.weight = nn.Parameter(D) 
        #geotorch.orthogonal(self.linear, "weight")     
        #self.linear.weight =  D.transpose(1,0)
        #self.linear.weight =  torch.eye(K,n)
        #geotorch.orthogonal(self.weights)
        #geotorch.Stiefel(self.linear, "weight") 
        #geotorch.Stiefel(self.weights)
        self.reset_parameters()

    def reset_parameters(self):
        # Every manifold has a convenience sample method, but you can use your own initializer
        #Stiefel = nn.Parameter(self.initeigv.clone())#self.initeigv#.type(torch.float64).requires_grad_()
        #Stiefel = nn.Parameter(self.initeigv.clone())#self.initeigv#.type(torch.float64).requires_grad_()
        self.weight = nn.Parameter(self.initeigv.clone())
        pass

    def forward(self, X):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-k * X) + b),
        """
        #p=2
        f = self.weight
        FF = f.repeat(1,self.n)
        FF = FF.reshape(self.n,self.n,self.K)
        #FFF = torch.sum(torch.pow(torch.abs(f), 1/p))
        FFF = torch.norm(self.weight, self.p,dim=0)
        FFF = torch.pow(FFF,self.p)
        FF = FF.transpose(2,0)
        GG =FF.transpose(1,2)
        A = X.unsqueeze(dim=1)
        #WW = A.unsqueeze(dim=-1)
        #Ww = WW.expand(-1,-1,-1,3)
        KK = FF - GG #this must be changed, since the values must be taken in norm and so on
        KKK = KK.unsqueeze(dim=-1)
        KKK = torch.pow(torch.abs(KKK),self.p)
        #print(A.size(), KKK.size())
        KKK = KKK.type(torch.float64)
        A = A.type(torch.float64)
        LL = torch.matmul(A, KKK)
        FFF = torch.pow(FFF,-1)
        FFF.unsqueeze_(-1)
        FFF.unsqueeze_(-1)
        FFF.unsqueeze_(-1)
        FFF = FFF.repeat(1,self.n,1,1)
        b = torch.matmul(LL,FFF)
        b = torch.sum(b)
        return b

def training_loop1(model, optimizer, sched,W, epochs=100):
    "Training loop for torch model."
    losses = []
    for i in range(epochs):
        preds = model(W)
        loss = preds
        #print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #sched.step(loss)
        losses.append(loss)  
    return losses



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=75,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="output",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ## automatic dataloading and splitting
    # dataset = PygGraphPropPredDataset(name = args.dataset)

    grid_sizes = [(64,24)]
    num_eigs = 5 #gives the dimension of the embedding or: num_eigs - 1 is the number of eigenvectors we calculate
    plot_labels = False
    add_side_plots = True

    A = make_2d_graph(grid_sizes[0][0],grid_sizes[0][1], periodic=False) 
    print(A.shape)
    D, L, L_inv, eigval,eigvec = get_graph_props(A,normalize_L='none')

    D, L, L_inv, eigval, init_eigvec = get_graph_props(A,normalize_L='none')

    p = 1
    steps = 930
    alpha = 0.01

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

    W = torch.tensor(A)

    n= eigval.shape[0]
    K = num_eigs
    epochs = 1000
    # n = 64
    # K = 5

    # instantiate model
    W = torch.tensor(A).to(device)
    F_ = torch.tensor(hi[:, 0:num_eigs]).to(device) #We can use previous outputs weight
    m = Model_RGD(F_, 1, n, K, ball = geoopt.CanonicalStiefel()).to(device) #I think we should not use F_ at initizialization, rather as a forward input so that we can start different init, or just use the reset parameters differently

    # Instantiate optimizer
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    #opt = torch.optim.Adam(params=m.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = geoopt.optim.RiemannianAdam(m.parameters(), lr=1e-2)
    #optimizer = geoopt.optim.RiemannianSGD(m.parameters(), lr=1e-2, momentum=0.9)

    decayRate = 0.99
    my_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=opt)#, gamma=decayRate)

    scheduler = None #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    start = timer()
    #for i in range(0,10):
    #  losses = training_loop1(m, opt,scheduler)
    #end = timer()
    losses = training_loop1(m, optimizer,my_lr_scheduler,W, epochs) 
    end = timer()
    print(end - start, " Sekunden")

    # m.to('cpu')

    print('p = ', p, ' step size = ', alpha)


    m.to('cpu')
    for p in range(1,2):
        p = 1
        alpha = 0.01
        #F = init_eigvec[:, 0:num_eigs]
        print('p = ', p, ' step size = ', alpha)

        
        #for i in range(0,steps):
            #should give us the full approximation matrix of p-eigenfunctions
            #k'th row is the embedding of node k
        #  grad, ss = calc_grad(A, F, p)
        #  F = F - alpha*ss*grad

        for grid_size in grid_sizes:

            # Initialize figure sizes
            size_factor = 5 / grid_size[1]
            node_size = 800 * size_factor
            figsize = [g * size_factor for g in grid_size]

            # Initialize adjacency
            A = make_2d_graph(*grid_size, periodic=False) 
            # print(A)

            # Get graph, laplacian, spacial positions
            # D, L, L_inv, eigval,eigvec = get_graph_props(A)
            p_eigvec = np.array(m.weight.detach().numpy())
            fig = plt.figure()
            # plt.scatter(np.arange(A.shape[0]), L_inv[0, :])
            graph = nx.from_numpy_array(A)
            pos = [(ii, jj) for ii in range(grid_size[0]) for jj in range(grid_size[1])]

            # Plot all the eigenvectors
            #for ii in range(num_eigs-1):
            for ii in range(num_eigs):

                im_dir = f'images_out/Eig grid side plots/grid-size-{grid_size}/'

                # Prepare the figure and subplots
                if add_side_plots:
                    grid_factor = grid_size[1]/grid_size[0]
                    new_fig_factor = np.sqrt((5*grid_factor + 2) / 7)
                    new_figsize = [figsize[0], figsize[1]/new_fig_factor]
                    f, axes = plt.subplots(3, 3, figsize=new_figsize,
                        gridspec_kw={'width_ratios': [1, 5, 1], 'height_ratios': [1, 5*grid_factor, 1]})
                    f.suptitle(f'$\phi_{ii}$ - grid_size {grid_size}')
                    axes[0, 0].axis('off')
                    axes[0, 2].axis('off')
                    axes[2, 0].axis('off')
                    axes[2, 2].axis('off')
                    plt.sca(axes[1, 1])
                    new_node_size = node_size * 0.5
                    
                else:
                    plt.figure(figsize=figsize)
                    new_node_size = node_size
                    im_dir = f'images_out/Eig grid/grid-size-{grid_size}/'
                
                # Plot the colored graph with eigenvectors
                node_vals = np.real(p_eigvec[:, ii])
                node_vals /= np.max(np.abs(node_vals)) + 1e-6
                labels = {ii: '{:.3f}'.format(node_vals[ii]) for ii in range(len(node_vals))} if plot_labels else {}
                plt.gca().set_aspect('equal')
                nx.draw(graph, pos=pos, node_color=node_vals, vmin=-1, vmax=1, cmap='PiYG', 
                        labels=labels, node_size=new_node_size, ax=plt.gca())
                

                if add_side_plots:
                    # Plot the eigenvectors on left
                    y = np.arange(grid_size[1])
                    axes[1, 0].plot(node_vals[:grid_size[1]], y, marker='o')
                    axes[1, 0].set_xlim(-1, 1)
                    axes[1, 0].set_xticks([-1, 0, 1])
                    axes[1, 0].set_yticks([])
                    axes[1, 0].axvline(0, linestyle=':')
                    axes[1, 0].set_title('left column')

                    # Plot the eigenvectors on the right
                    axes[1, 2].plot(node_vals[-grid_size[1]:], y, marker='o')
                    axes[1, 2].set_xlim(-1, 1)
                    axes[1, 2].set_xticks([-1, 0, 1])
                    axes[1, 2].set_yticks([])
                    axes[1, 2].axvline(0, linestyle=':')
                    axes[1, 2].set_title('right column')

                    # Plot the eigenvectors on bottom
                    x = np.arange(grid_size[0])
                    axes[2, 1].plot(x, node_vals[::grid_size[1]], marker='o')
                    axes[2, 1].set_ylim(-1, 1)
                    axes[2, 1].set_xticks([])
                    axes[2, 1].set_yticks([-1, 0, 1])
                    axes[2, 1].axhline(0, linestyle=':')
                    axes[2, 1].set_title('bottom row')

                    # Plot the eigenvectors on top
                    axes[0, 1].plot(x, node_vals[grid_size[1]-1::grid_size[1]], marker='o')
                    axes[0, 1].set_ylim(-1, 1)
                    axes[0, 1].set_xticks([])
                    axes[0, 1].set_yticks([-1, 0, 1])
                    axes[0, 1].axhline(0, linestyle=':')
                    axes[0, 1].set_title('top row')

                    # fig.savefig(f'./phi_{ii}.png')
                    plt.show()
                    # print(ii, node_vals)


if __name__ == "__main__":
    main()
