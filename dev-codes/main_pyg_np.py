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

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from torch_geometric.datasets.zinc import ZINC

import torch
import argparse
from timeit import default_timer as timer
import torch
import torch.nn.functional as F
import torch.nn as nn
from numba import jit

#import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

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

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

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

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def phi(x, p):
  return np.abs(x)**(p-1)*np.sign(x)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def lp_norm(x, p):
  #gives the p-th power of the lp-norm
  y = np.sum(np.power(np.abs(x),p))
  return y

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def jacobian_grad(A, f, p):
#A is the weight matrix and f denotes the current embedding of the nodes 
#(row is the node and column are the dimension of the embedding)
#implements equation (22)
  B = np.zeros((f.shape[0], f.shape[1]))
  for i in range(0, f.shape[0]):
    for k in range(0, f.shape[1]):
      helper = [ A[i,j]*phi(f[i,k] - f[j,k],p)  for j in range(0, f.shape[0]) ]
      grad = np.sum(np.array(helper))
      grad = grad - phi(f[i,k],p)/lp_norm(f[:,k], p)
      grad = 1/lp_norm(f[:,k], p) * grad
      B[i,k] = grad
  return B

def calc_grad(A, f,p):
#implements G in Algorithm 1
  grad = jacobian_grad(A, f, p)
  G = grad - np.matmul(np.matmul(f,np.transpose(grad)),f)
  step_size = lp_norm(f, 1)/lp_norm(G,1) #for the adaptive stepsize
  return G, step_size

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

    grid_sizes = [(132,24)]
    num_eigs = 5 #gives the dimension of the embedding or: num_eigs - 1 is the number of eigenvectors we calculate
    plot_labels = False
    add_side_plots = True

    A = make_2d_graph(grid_sizes[0][0],grid_sizes[0][1], periodic=False) 
    print(A.shape)
    D, L, L_inv, eigval,eigvec = get_graph_props(A,normalize_L='none')

    D, L, L_inv, eigval, init_eigvec = get_graph_props(A,normalize_L='none')


    start = timer()

    """
    for i in range(0,10):
    p = 1.5
    steps = 1000
    alpha = 0.01
    FF = init_eigvec[:, 0:num_eigs]
    print('p = ', p, ' step size = ', alpha)


    for i in range(0,steps):
        #should give us the full approximation matrix of p-eigenfunctions
        #k'th row is the embedding of node k
        grad, ss = calc_grad(A, FF, p)
        FF = FF - alpha*ss*grad
    """

    p = 1
    steps = 10
    alpha = 0.01
    FF = init_eigvec[:, 0:num_eigs]
    print('p = ', p, ' step size = ', alpha)


    for i in tqdm(range(0,steps)):
        #should give us the full approximation matrix of p-eigenfunctions
        #k'th row is the embedding of node k
        grad, ss = calc_grad(A, FF, p)
        FF = FF - alpha*ss*grad

    end = timer()


    print((end - start), " Sekunden")


    for i in range(1,2):
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
            p_eigvec = FF
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

                    fig.savefig(f'./phi_{ii}.png')
                    plt.show()
                    print(ii, node_vals)



if __name__ == "__main__":
    main()
