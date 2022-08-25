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
# from numba import jit

from typing import Any, Optional
from torch_geometric.data import Data

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
import json

from utils_1 import get_graph_props, make_2d_graph
from gnn import GCN,SAGE
from logger import Logger
from data import get_dataset, set_fixed_train_val_test_split, update_dataset
from best_params import best_params_dict
from utils import ROOT_DIR

sns.set_style("whitegrid")

import wandb
from pytorch_lightning.loggers import WandbLogger


gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv,
    "ChebNet": geom_nn.ChebConv
}



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


def mask_to_index(mask: Tensor) -> Tensor:
    r"""Converts a mask to an index representation.

    Args:
        mask (Tensor): The mask.
    """
    return mask.nonzero(as_tuple=False).view(-1)

def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)
    y_true = data.y.unsqueeze(1)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc




def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)



class GNNModel(nn.Module):
    
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels, 
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels, 
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class NodeLevelGNN(pl.LightningModule):
    
    def __init__(self, model_name, opt_name, lr, weight_decay, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        
        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GNNModel(**model_kwargs)
        self.optimizer = get_optimizer(opt_name, self.parameters(), lr=lr, weight_decay=weight_decay)

        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        print(data)
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)
        
        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"
        
        loss = self.loss_module(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = self.optimizer
        #optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    """
    def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = "model_final.onnx"
        self.to_onnx(model_filename, dummy_input, export_params=True)
        wandb.save(model_filename)
    """

    def test_epoch_end(self, outputs):
        print(outputs)
        final_value = 0
        for dataloader_outputs in outputs:
            for test_step_out in dataloader_outputs:
                # do something
                final_value += test_step_out

        self.log("final_metric", final_value)

    # def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
    #     dummy_input = torch.zeros(self.hparams["c_in"], device=self.device)
    #     model_filename = "model_final.onnx"
    #     self.to_onnx(model_filename, dummy_input, export_params=True)
    #     wandb.save(model_filename)

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)


def train_node_classifier(model_name, dataset, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = DataLoader(dataset, batch_size=1)
    
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=200)
    
    pl.seed_everything()
    if dataset == cora_dataset:
      model = NodeLevelGNN(model_name=model_name, c_in=cora_dataset.num_node_features, c_out=cora_dataset.num_classes, **model_kwargs)
    else:
      model = NodeLevelGNN(model_name=model_name, c_in=cora_dataset.num_node_features + num_eigs, c_out=cora_dataset.num_classes, **model_kwargs)
    trainer.fit(model, node_data_loader, node_data_loader)
    model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)


    # Test best model on the test set
    #test_result = trainer.test(model, node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    _, test_acc = model.forward(batch, mode='test')
    result = {"train": train_acc,
              "val": val_acc,
              "test": test_acc
              }
              #"test": test_result[0]['test_acc']}
    return model, result


def train_node_classifier_1(device,num_eigs,CHECKPOINT_PATH,opt_name, lr, weight_decay,dataset_type,model_name, dataset, max_epochs,**model_kwargs):
    pl.seed_everything(42)
    node_data_loader = geom_data.DataLoader(dataset, batch_size=1)
    
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         logger = WandbLogger(offline=True), 
                         max_epochs=max_epochs)

    
    pl.seed_everything()

    if dataset_type == "original":
      model = NodeLevelGNN(model_name,opt_name, lr, weight_decay, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs)
    else:
      model = NodeLevelGNN(model_name,opt_name, lr, weight_decay, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs)

    model = NodeLevelGNN(model_name,opt_name, lr, weight_decay, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs)
    trainer.fit(model, node_data_loader, node_data_loader)
    model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)


    # Test best model on the test set
    test_result = trainer.test(model, node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    _, test_acc = model.forward(batch, mode='test')
    result = {"train": train_acc,
              "val": val_acc,
              "test": test_result[0]['test_acc']}
    return model, result

# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print(f"Train accuracy: {(100.0*result_dict['train']):4.2f}%")
    if "val" in result_dict:
        print(f"Val accuracy:   {(100.0*result_dict['val']):4.2f}%")
    print(f"Test accuracy:  {(100.0*result_dict['test']):4.2f}%")

def get_orthonromal_eigvec(eigval, eigvec):
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

def plot_fig(grid_sizes, p_eigvec,num_eigs,add_side_plots=True,plot_labels=False):

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

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = nn.Parameter(self.initeigv.clone())
        pass

    def forward(self, X):
        f = self.weight
        FF = f.repeat(1,self.n)
        FF = FF.reshape(self.n,self.n,self.K)
        FFF = torch.norm(self.weight, self.p,dim=0)
        FFF = torch.pow(FFF,self.p)
        FF = FF.transpose(2,0)
        GG =FF.transpose(1,2)
        A = X.unsqueeze(dim=1)
        KK = FF - GG
        KKK = KK.unsqueeze(dim=-1)
        KKK = torch.pow(torch.abs(KKK),self.p)
        KKK = KKK.type(torch.float64)
        A = A.type(torch.float64)
        LL = torch.matmul(A, KKK)
        FFF = torch.pow(FFF,-1)
        FFF.unsqueeze_(-1)
        FFF.unsqueeze_(-1)
        FFF.unsqueeze_(-1)
        FFF = FFF.repeat(1,self.n,1,1)
        b = torch.matmul(LL.float(),FFF)
        b = torch.sum(b)
        return b

def training_loop1(model, optimizer, sched,W, epochs=100):
    "Training loop for torch model."
    losses = []
    for i in range(epochs):
        preds = model(W)
        loss = preds
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)
    return losses


class GCNLayer(nn.Module):
    
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections. 
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats


def main(cmd_opt):
    opt = cmd_opt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path to the folder where the datasets are/should be downloaded 
    DATASET_PATH = "../data"
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "../saved_models/node_level"

    # Setting the seed
    pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_org = get_dataset(opt, f'{ROOT_DIR}/data', opt['not_lcc'])


    cora_adj = to_dense_adj(dataset_org[0].edge_index)
    cora_adj.squeeze_()

    num_eigs = opt['num_eigs']   #gives the dimension of the embedding or/ the number of eigenvectors we calculate

    A = cora_adj.numpy()
    D, L, L_inv, eigval,eigvec = get_graph_props(A,normalize_L='none')

    p = opt['p_laplacian']
    alpha = 0.01

    hi = get_orthonromal_eigvec(eigval,eigvec)


    W = torch.tensor(A)

    n= eigval.shape[0]
    K = num_eigs
    epochs = opt['epochs_manifold']

    # instantiate model
    W = torch.tensor(A).float().to(device)
    F_ = torch.tensor(hi[:, 0:num_eigs]).float().to(device) #We can use previous outputs weight

    if opt['manifold'] == "Can Stiefel":
        m = Model_RGD(F_, p, n, K, ball=geoopt.CanonicalStiefel()).to(device)
    if opt['manifold'] == "Euc Exact Stiefel":
        m = Model_RGD(F_, p, n, K, ball=geoopt.EuclideanStiefelExact()).to(device)
    if opt['manifold'] == "Euc Stiefel":
        m = Model_RGD(F_, p, n, K, ball=geoopt.EuclideanExact()).to(device)


    # Instantiate optimizer
    lr_m = opt['lr_manifold']

    #optim = torch.optim.SGD(m.parameters(), lr=0.01)
    #opt = torch.optim.Adam(params=m.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = geoopt.optim.RiemannianAdam(m.parameters(), lr=lr)
    #optimizer = geoopt.optim.RiemannianSGD(m.parameters(), lr=1e-2, momentum=0.9)
    if opt['optimizer_manifold'] == "adam":
        optimizer = geoopt.optim.RiemannianAdam(m.parameters(), lr=lr_m)
    if opt['optimizer_manifold'] == "sgd":
        optimizer = geoopt.optim.RiemannianSGD(m.parameters(), lr=lr_m, momentum=0.9)


    decayRate = 0.99
    my_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optim)#, gamma=decayRate)

    scheduler = None #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    start = timer()
    #for i in range(0,10):
    #  losses = training_loop1(m, opt,scheduler)
    #end = timer()
    losses = training_loop1(m, optimizer,my_lr_scheduler,W, epochs) 
    end = timer()
    print(end - start, " second")
    

    loss = [x.to('cpu').detach().numpy() for x in losses]
    x = np.arange(0,epochs)
    y = np.array(loss)
    
    # Plotting the Graph
    m.to('cpu')

    """
    plt.plot(x, y)
    plt.title("Loss per epoch in optimization for p-eigenvectors")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plt.savefig('loss.png')

    # cora_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name="Cora")

    m.to('cpu')
    """
    # xx = torch.cat((cora_dataset[0].x, m.weight),1)

    # datal = [Data(xx,cora_dataset[0].edge_index)]
         
    # datal[0].train_mask = cora_dataset[0].train_mask
    # datal[0].val_mask = cora_dataset[0].val_mask

    # datal[0].test_mask = cora_dataset[0].test_mask
    # datal[0].y  = cora_dataset[0].y

    # loader = DataLoader(datal, batch_size=32)

    if args.use_lp:
        dataset = update_dataset(dataset_org, m.weight)
    opt = cmd_opt
    dataset = dataset_org.copy()
    dataset_enriched = update_dataset(dataset_org, m.weight)

    #load model parameters

    use_baseline = opt['use_baseline']
    opt_name = opt['optimizer']
    lr = opt['lr']
    weight_decay = opt['decay']



    # Standard dataset
    if not use_baseline:
        node_gnn_model, node_gnn_result = train_node_classifier_1(device,
                                                                num_eigs,
                                                                CHECKPOINT_PATH,
                                                                opt_name=opt_name,
                                                                lr=lr,
                                                                weight_decay=weight_decay,
                                                                dataset_type="original",
                                                                model_name="GCN",
                                                                layer_name="GCN",
                                                                dataset=dataset,
                                                                max_epochs = opt['max_epochs'],
                                                                c_hidden=opt['hidden_channels'],
                                                                num_layers=opt['num_layers'],
                                                                dp_rate=opt['dropout']
                                                                  )
        print_results(node_gnn_result)


        #Pretransformed with p-LPE
    node_gnn_model, node_gnn_result = train_node_classifier_1(device,
                                                            num_eigs,
                                                            CHECKPOINT_PATH,
                                                            dataset_type="lp",
                                                            model_name="GCN",
                                                            layer_name="GCN",
                                                            dataset=dataset_enriched, 
                                                            max_epochs = opt['max_epochs'],
                                                            c_hidden=opt['hidden_channels'],
                                                            num_layers=opt['num_layers'],
                                                            dp_rate=opt['dropout'])
    print_results(node_gnn_result)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cora_defaults', action='store_true',
                        help='Whether to run with best params for cora. Overrides the choice of dataset')
    # data args
    parser.add_argument('--dataset', type=str, default='texas',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true',
                        help='use the 10 fixed splits from '
                            'https://arxiv.org/abs/2002.05287')
    parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                        help='the number of splits to repeat the results on')
    parser.add_argument('--label_rate', type=float, default=0.5,
                        help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--planetoid_split', action='store_false',
                        help='use planetoid splits for Cora/Citeseer/Pubmed')
    parser.add_argument('--not_lcc', action='store_true',
                        help='use largest connected component')
    parser.add_argument("--global_random_seed", type=int, default=2021,
                            help="Random seed (for reproducibility).")
    parser.add_argument("--outputpath", type=str, default="empty.json",
                        help="outputh file path to save the result")
    parser.add_argument("--train_ratio", type=float, default=1.,
                        help="the start value of the train ratio (inclusive).")


    # Preprocessing args
    parser.add_argument('--use_lp', action='store_true',
                        help='use LPE eigenfunctions')
    parser.add_argument('--use_baseline', type=bool, default=False,
                        help='Train baseline model without positional encoding')
    parser.add_argument('--manifold', type=str, default="Euc Exact Stiefel",
                        help='Choice of Stiefel manifold (default: Euc Exact Stiefel). Choices: Euc Stiefel, Can Stiefel')
    parser.add_argument('--lr_manifold', type=float, default=0.01,
                        help='Choice of Stiefel manifold (default: Euc Exact Stiefel). Choices: Euc Stiefel, Can Stiefel')
    parser.add_argument('--optimizer_manifold', type=str, default="sgd",
                        help='Choice of manifold optimizer (default: SGD). Choices: Adam')
    parser.add_argument('--epochs_manifold', type=int, default=250, help='number of epochs to train for p-LP ev calculations (default: 250)')



    # Training settings
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="output",
                        help='filename to output result (default: )')
    parser.add_argument('--max_epochs', type=int, default="200",
                        help='number of epochs to train the GNN (default: 200)')

    # GNN args
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--p_laplacian', type=float, default=1,
                        help='the value for p-laplcian (default: 1)')
    parser.add_argument('--num_eigs', type=int, default=7,
                        help='number of eigenvectors (default: 5)')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    
    
    # parser.add_argument("--splits", type=int, default=5,
    #                     help="The number of re-shuffling & splitting for each train ratio.")


    # parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
    # parser.add_argument('--fc_out', dest='fc_out', action='store_true',
    #                     help='Add a fully connected layer to the decoder.')
    # parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    # parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    # parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    # parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    # parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
    # parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    # parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    # parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
    #                     help='apply sigmoid before multiplying by alpha')
    # parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    # parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
    # parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT')
    # parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
    #                     help='Add a fully connected layer to the encoder.')
    # parser.add_argument('--add_source', dest='add_source', action='store_true',
    #                     help='If try get rid of alpha param and the beta*x0 source term')
    # parser.add_argument('--cgnn', dest='cgnn', action='store_true', help='Run the baseline CGNN model from ICML20')


    args = parser.parse_args()

    opt = vars(args)

    main(opt)

