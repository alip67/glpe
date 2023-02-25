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

#import wandb

from pytorch_lightning.loggers import WandbLogger
##wandb_logger = WandbLogger(project="Node Classification Cora")







def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
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



def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)

import torch
from torch_geometric.nn import MessagePassing




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
        #self.log('train_loss', loss)
        #self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)


    


def train_node_classifier_1(device,num_eigs,CHECKPOINT_PATH,opt_name, lr, weight_decay,dataset_type,model_name, dataset, max_epochs,**model_kwargs):
    pl.seed_everything(42)
    node_data_loader = geom_data.DataLoader(dataset, batch_size=1)
    
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         ##logger = wandb_logger, 
                         max_epochs=max_epochs)

    
    pl.seed_everything()

    model = NodeLevelGNN(model_name,opt_name, lr, weight_decay, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs)

    trainer.fit(model, node_data_loader, node_data_loader)
    model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    ##wandb_logger.watch(model, log="all")

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

gnn_layer_by_name = {
    "GCN": GCN,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv,
    "ChebNet": geom_nn.ChebConv
    }



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

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    
    #dataset_org = get_dataset(opt, f'{ROOT_DIR}/data', opt['not_lcc'])
    dataset_org = get_dataset(opt, f'{ROOT_DIR}/data', True)
    
    cora_adj = to_dense_adj(dataset_org[0].edge_index)
    cora_adj.squeeze_()

    class GCN(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super(GCN, self).__init__(aggr='add')
            self.in_channels = in_channels
            self.out_channels =out_channels
            self.register_parameter('alpha', torch.nn.Parameter(torch.ones(1)))
            self.register_parameter('W', torch.nn.Parameter(torch.randn(in_channels, out_channels)))
            # Construct D^(-1/2)AD^(-1/2)
            D = cora_adj.sum(dim=1)
            D_inv = 1/D
            D_inv_sqrt = torch.sqrt(D_inv)
            D_inv_sqrt = torch.diag(D_inv_sqrt)
            self.L = D_inv_sqrt @ cora_adj @ D_inv_sqrt
            # Compute eigendecomposition of L
            e, U = torch.eig(self.L, eigenvectors=True)
            e = e.real
            U = U.real
            self.e, self.U = e, U

        def forward(self, x, edge_index):
            return self.propagate(edge_index, x=x)

        def message(self, x_j, edge_index, size_i):
            return x_j

        def update(self, aggr_out, x):
            # Construct L^alpha
            e_alpha = torch.pow(self.e, self.alpha)
            e_alpha = torch.diag(e_alpha)
            L_alpha = self.U @ e_alpha @ self.U.T
            # Compute L^alpha x W
            return L_alpha @ x @ self.W

    opt_name = opt['optimizer']
    lr = opt['lr']
    weight_decay = opt['decay']
     
    node_gnn_model, node_gnn_result = train_node_classifier_1(device,
                                                            1,
                                                            CHECKPOINT_PATH,
                                                            opt_name=opt_name,
                                                            lr=lr,
                                                            weight_decay=weight_decay,
                                                            dataset_type="original",
                                                            model_name="GCN",
                                                            layer_name="GCN",
                                                            dataset=dataset_org,
                                                            max_epochs = opt['max_epochs'],
                                                            c_hidden=opt['hidden_channels'],
                                                            num_layers=opt['num_layers'],
                                                            dp_rate=opt['dropout']
                                                                )
    print_results(node_gnn_result)


    
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cora_defaults', action='store_true',
                        help='Whether to run with best params for cora. Overrides the choice of dataset')
    # data args
    parser.add_argument('--use_cache', type=bool, default=True, help='Is their a pretrained version?')
    parser.add_argument('--dataset', type=str, default='Cora',
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
    parser.add_argument('--max_epochs', type=int, default="50",
                        help='number of epochs to train the GNN (default: 500)')

    # GNN args
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
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

