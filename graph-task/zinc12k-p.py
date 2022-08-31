
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import (GINConv,global_add_pool,GATConv,ChebConv,GCNConv)
from torch_geometric.datasets import ZINC
from torch_geometric.utils import to_networkx, to_dense_adj
import geoopt
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from torch_geometric.data import InMemoryDataset

from libs.spect_conv import SpectConv,ML3Layer
from libs.utils import Zinc12KDataset,SpectralDesign,get_n_params
from utils_1 import get_graph_props, make_2d_graph
from torch_geometric.data import Data
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data.collate import collate

from tqdm import tqdm
import argparse


import os
os.environ["WANDB_MODE"]="offline"

import wandb


 
def print_statistics(dataset,type):

    print()
    print(f'Dataset: {dataset}_{type}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')


    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


# transform = SpectralDesign(nmax=37,recfield=2,dv=2,nfreq=7) 

# dataset = Zinc12KDataset(root="graph-task/dataset/ZINC/",pre_transform=transform)

# train_data = ZINC(root="dataset/ZINC/", subset=True)

# trid=list(range(0,10000))
# vlid=list(range(10000,11000))
# tsid=list(range(11000,12000))

# train_loader = DataLoader(dataset[trid], batch_size=64, shuffle=True)
# val_loader = DataLoader(dataset[vlid], batch_size=64, shuffle=False)
# test_loader = DataLoader(dataset[tsid], batch_size=64, shuffle=False)

parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--p_laplacian', type=int, default=1,
                    help='the value for p-laplcian (default: 1)')
parser.add_argument('--num_eigs', type=int, default=5,
                    help='number of eigenvectors (default: 5)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers (default: 0)')
parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                    help='dataset name (default: ogbg-molhiv)')
parser.add_argument('--batch-size', type=int, default=128)

parser.add_argument('--feature', type=str, default="full",
                    help='full feature or simple feature')
parser.add_argument('--filename', type=str, default="output",
                    help='filename to output result (default: )')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_data = ZINC(root="data/ZINC/", subset=True, split='train')
# val_data = ZINC(root="data/ZINC/", subset=True, split='val' )
# test_data = ZINC(root="data/ZINC/", subset=True, split='test')


# print_statistics(train_data,type="train")
# print_statistics(val_data,type="valid")
# print_statistics(test_data,type="test")

# num_eigs = args.num_eigs#gives the dimension of the embedding or/ the number of eigenvectors we calculate
# p = args.p_laplacian
# epochs = args.epochs

# train_dataset = preprocess_dataset(train_data,num_eigs,epochs,p,device)
# val_dataset = preprocess_dataset(val_data,num_eigs,epochs,p,device)
# test_dataset = preprocess_dataset(test_data,num_eigs,epochs,p,device)

# batch_size = args.batch_size


# train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


transform = SpectralDesign(nmax=37,recfield=2,dv=2,nfreq=7) 

# dataset = Zinc12KDataset(root="graph-task/dataset/ZINC/",pre_transform=transform)
#dataset = Zinc12KDataset(root="graph-task/dataset/ZINC/",pre_transform=transform) #For Ali

########## commented this:
#dataset = Zinc12KDataset(root="dataset/ZINC/",pre_transform=transform) #For Sohir


# dataset = Zinc12KDataset(root="graph-task/dataset/ZINC/")

########## commented this:
#print_statistics(dataset,type="train") 

num_eigs = args.num_eigs#gives the dimension of the embedding or/ the number of eigenvectors we calculate
p = args.p_laplacian
epochs = args.epochs

######### commented this
#dataset_dup= dataset.copy()

# train_dataset = dataset.post_process(dataset_dup,num_eigs,epochs,p,device)
# torch.save(train_dataset, f'dataset_zinc_p{p}.pt')
train_dataset = torch.load(f'../dataset_zinc_p{p}.pt')


trid=list(range(0,10000))
vlid=list(range(10000,11000))
tsid=list(range(11000,12000))

train_loader = DataLoader(train_dataset[trid], batch_size=64, shuffle=True)
val_loader = DataLoader(train_dataset[vlid], batch_size=64, shuffle=False)
test_loader = DataLoader(train_dataset[tsid], batch_size=64, shuffle=False)

# train_loader = DataLoader([train_dataset[i] for i in np.asarray(trid)], batch_size=64, shuffle=True)
# val_loader = DataLoader([train_dataset[i] for i in np.asarray(vlid)], batch_size=64, shuffle=False)
# test_loader = DataLoader([train_dataset[i] for i in np.asarray(tsid)], batch_size=64, shuffle=False)


class PPGN(nn.Module):
    def __init__(self,nmax=37,nneuron=32):
        super(PPGN, self).__init__()

        self.nmax=nmax        
        self.nneuron=nneuron
        ninp=dataset.data.X2.shape[1]
        
        bias=False
        self.mlp1_1 = torch.nn.Conv2d(ninp,nneuron,1,bias=bias) 
        self.mlp1_2 = torch.nn.Conv2d(ninp,nneuron,1,bias=bias) 
        self.mlp1_3 = torch.nn.Conv2d(nneuron+ninp, nneuron,1,bias=bias) 

        self.mlp2_1 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp2_2 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias)
        self.mlp2_3 = torch.nn.Conv2d(2*nneuron,nneuron,1,bias=bias) 

        self.mlp3_1 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp3_2 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp3_3 = torch.nn.Conv2d(2*nneuron,nneuron,1,bias=bias)       

        self.mlp4_1 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp4_2 = torch.nn.Conv2d(nneuron,nneuron,1,bias=bias) 
        self.mlp4_3 = torch.nn.Conv2d(2*nneuron,nneuron,1,bias=bias)          
        
        self.h1 = torch.nn.Linear(2*4*nneuron, 64) 
        self.h2 = torch.nn.Linear(64, 1)       
        

    def forward(self,data):
        x=data.X2 
        M=torch.sum(data.M,(1),True) 

        x1=F.relu(self.mlp1_1(x)*M) 
        x2=F.relu(self.mlp1_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp1_3(torch.cat([x1x2,x],1))*M) 

        # sum layer readout
        xo1=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3)),torch.sum(x*data.M[:,1:2,:,:],(2,3))],1)
        

        x1=F.relu(self.mlp2_1(x)*M) 
        x2=F.relu(self.mlp2_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp2_3(torch.cat([x1x2,x],1))*M) 

        # sum layer readout       
        xo2=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3)),torch.sum(x*data.M[:,1:2,:,:],(2,3))],1)
        
        x1=F.relu(self.mlp3_1(x)*M) 
        x2=F.relu(self.mlp3_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp3_3(torch.cat([x1x2,x],1))*M) 

        # sum  layer readout
        xo3=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3)),torch.sum(x*data.M[:,1:2,:,:],(2,3))],1)


        x1=F.relu(self.mlp4_1(x)*M) 
        x2=F.relu(self.mlp4_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp4_3(torch.cat([x1x2,x],1))*M) 

        # sum  layer readout
        xo4=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3)),torch.sum(x*data.M[:,1:2,:,:],(2,3))],1)
        

        x=torch.cat([xo1,xo2,xo3,xo4],1) 
        x=F.relu(self.h1(x))
        return self.h2(x)

class GinNet(nn.Module):
    def __init__(self):
        super(GinNet, self).__init__()

        nn1 = Sequential(Linear(train_dataset.num_features, 64), ReLU(), Linear(64, 64))
        self.conv1 = GINConv(nn1,train_eps=True)
        self.bn1 = torch.nn.BatchNorm1d(64)

        nn2 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv2 = GINConv(nn2,train_eps=True)
        self.bn2 = torch.nn.BatchNorm1d(64)

        nn3 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv3 = GINConv(nn3,train_eps=True)
        self.bn3 = torch.nn.BatchNorm1d(64)

        nn4 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv4 = GINConv(nn4,train_eps=True)
        self.bn4 = torch.nn.BatchNorm1d(64)

        
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 1) 

    def forward(self, data):

        x=data.x        
            
        edge_index=data.edge_index

        
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x) 

        
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)    

        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)       

        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x) 

class GcnNet(nn.Module):
    def __init__(self):
        super(GcnNet, self).__init__()
        neuron=96
        self.conv1 = GCNConv(train_dataset.num_features, neuron, cached=False)
        self.conv2 = GCNConv(neuron, neuron, cached=False)
        self.conv3 = GCNConv(neuron, neuron, cached=False)
        self.conv4 = GCNConv(neuron, neuron, cached=False)       
        
        self.fc1 = torch.nn.Linear(neuron, 32)
        self.fc2 = torch.nn.Linear(32, 1) 

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))            
        x = F.relu(self.conv2(x, edge_index))        
        x = F.relu(self.conv3(x, edge_index)) 
        x = F.relu(self.conv4(x, edge_index))

        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x) 

class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()

        self.conv1 = torch.nn.Linear(dataset.num_features, 32)
        self.conv2 = torch.nn.Linear(32, 64)
        self.conv3 = torch.nn.Linear(64, 64)       
        
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 1) 

    def forward(self, data):

        x=data.x
        edge_index=data.edge_index
        x = F.relu(self.conv1(x))                
        x = F.relu(self.conv2(x))        
        x = F.relu(self.conv3(x)) 
        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x) 

class ChebNet(nn.Module):
    def __init__(self,S=5):
        super(ChebNet, self).__init__()

        S=2
        nn=64
        self.conv1 = ChebConv(dataset.num_features, nn,S)
        self.conv2 = ChebConv(nn, nn, S)
        self.conv3 = ChebConv(nn, nn, S)
        self.conv4 = ChebConv(nn, nn, S)
        
        self.fc1 = torch.nn.Linear(nn, 32)
        self.fc2 = torch.nn.Linear(32, 1) 

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index,lambda_max=data.lmax,batch=data.batch))              
        x = F.relu(self.conv2(x, edge_index,lambda_max=data.lmax,batch=data.batch))        
        x = F.relu(self.conv3(x, edge_index,lambda_max=data.lmax,batch=data.batch))
        x = F.relu(self.conv4(x, edge_index,lambda_max=data.lmax,batch=data.batch))

        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x) 

class GatNet(nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()

        '''number of param (in+3)*head*out
        '''
        self.conv1 = GATConv(dataset.num_features, 8, heads=8,concat=True, dropout=0.0)        
        self.conv2 = GATConv(64, 12, heads=8, concat=True, dropout=0.0)
        self.conv3 = GATConv(96, 12, heads=8, concat=True, dropout=0.0)
        self.conv4 = GATConv(96, 12, heads=8, concat=True, dropout=0.0)

        self.fc1 = torch.nn.Linear(96, 64)
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):
        x=data.x
        
        x = F.elu(self.conv1(x, data.edge_index))        
        x = F.elu(self.conv2(x, data.edge_index))        
        x = F.elu(self.conv3(x, data.edge_index))
        x = F.elu(self.conv4(x, data.edge_index)) 

        x = global_add_pool(x, data.batch)        
        x = F.relu(self.fc1(x))        
        return self.fc2(x) 



def train(epoch):
    
    model.train()
    
    L=0
    correct=0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        pre=model(data)
        #lss= torch.square(pre- data.y.unsqueeze(-1)).sum() 
        lss= torch.nn.functional.l1_loss(pre, data.y.unsqueeze(-1),reduction='sum')
        
        lss.backward()
        optimizer.step()  
        L+=lss.item()

    return L/len(trid)

def test():
    model.eval()
    
    L=0
    for data in test_loader:
        data = data.to(device)

        pre=model(data)
        #lss= torch.square(pre- data.y.unsqueeze(-1)).sum() 
        lss= torch.nn.functional.l1_loss(pre, data.y.unsqueeze(-1),reduction='sum') 
    
        L+=lss.item()
    
    Lv=0
    for data in val_loader:
        data = data.to(device)
        pre=model(data)
        #lss= torch.square(pre- data.y.unsqueeze(-1)).sum() 
        lss= torch.nn.functional.l1_loss(pre, data.y.unsqueeze(-1),reduction='sum')
        Lv+=lss.item()    
    return L/len(tsid), Lv/len(vlid)

bval=1000
btest=0
tr_loss= []
ts_loss=[]
valid_loss=[]

# 1. Start a W&B run
wandb.init(project='Graph Regression Zinc12K', config=args)


# 1.1 Make model
model = GcnNet().to(device)   # GatNet  ChebNet  GcnNet  GinNet  MlpNet  PPGN 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
print(get_n_params(model))

# 2. Save model inputs and hyperparameters
config = args
config = wandb.config

for epoch in range(1, 401):
    wandb.watch(model, tr_loss, log="all", log_freq=1)
    trloss=train(epoch)
    test_loss,val_loss = test()
    wandb.log({"epoch": epoch, "val_loss": val_loss})
    wandb.log({"epoch": epoch, "tr_loss": trloss})
    wandb.log({"epoch": epoch, "test_loss": test_loss})
    ts_loss.append(test_loss)
    tr_loss.append(trloss)
    valid_loss.append(val_loss) 
    if bval>val_loss:
        bval=val_loss
        btest=test_loss
        
    #print('Epoch: {:02d}, trloss: {:.4f},  Val: {:.4f}, Test: {:.4f}'.format(epoch,trloss,val_acc, test_acc))
    print('Epoch: {:02d}, trloss: {:.4f},  Valloss: {:.4f}, Testloss: {:.4f}, best test loss: {:.4f}'.format(epoch,trloss,val_loss,test_loss,btest))

wandb.log({"best_val_loss": bval})
wandb.log({"best_test_loss": btest})


x = np.arange(0,400)
y = np.array(ts_loss)
np.save(f'ts_loss_gin_p{p}.npy', ts_loss) 
# Plotting the Graph
plt.plot(x, y)
plt.title("Curve plotted using the given points")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
plt.savefig('loss_ZINC_GCN_with_p_laplcian.png')

