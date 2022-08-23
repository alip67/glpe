
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
import scipy.io

from libs.spect_conv import SpectConv,ML3Layer
from libs.utils import Zinc12KDataset,SpectralDesign,get_n_params
from utils_1 import get_graph_props, make_2d_graph
from torch_geometric.data import Data

from tqdm import tqdm
import argparse



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
        b = torch.matmul(LL.float(),FFF)
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

def preprocess_dataset(dataset,num_eigs,epochs,p,device): 
    datal = []
    for data in dataset:
        #Preprocessing

        cora_adj = to_dense_adj(data.edge_index)
        cora_adj.squeeze_()

        A = cora_adj.numpy()
        D, L, L_inv, eigval,eigvec = get_graph_props(A,normalize_L='none')

        #We transform our eigenvectors into an orthonormalbasis such that it is in the Stiefel manifold

        hi = get_orthonromal_eigvec(eigval,eigvec)

        n= eigval.shape[0]
        K = num_eigs
        epochs = epochs

        # instantiate model
        W = torch.tensor(A).float().to(device)
        F_ = torch.tensor(hi[:, 0:num_eigs]).float().to(device) #We can use previous outputs weight
        m = Model_RGD(F_, p, n, K, ball = geoopt.EuclideanStiefelExact()).to(device) #I think we should not use F_ at initizialization, rather as a forward input so that we can start different init, or just use the reset parameters differently

        # Instantiate optimizer
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        #opt = torch.optim.Adam(params=m.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        optimizer = geoopt.optim.RiemannianSGD(m.parameters(), lr=1e-2)
        #optimizer = geoopt.optim.RiemannianSGD(m.parameters(), lr=1e-2, momentum=0.9)

        decayRate = 0.99
        my_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=opt)#, gamma=decayRate)

        scheduler = None #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        #Learn the 1-eigenvector. It is then given by m.weight
        start = timer()
        losses = training_loop1(m, optimizer,my_lr_scheduler,W, epochs)  
        end = timer()
        print(end - start, " Second")

        m.to('cpu')
        xx = torch.cat((data.x, m.weight[:,1:3]),1)

        #xx = torch.cat((data.x, torch.tensor(eigvec[:,:7])),1)

        #Didnt know how to pretransform the features of CORA; This is my workaround
        datal.append(Data(xx,data.edge_index, y=data.y, edge_attr=data.edge_attr, batch = data.batch))
    return datal



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
parser.add_argument('--p_laplacian', type=int, default=1.2,
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
# dataset = Zinc12KDataset(root="graph-task/dataset/ZINC/",pre_transform=transform)
dataset = Zinc12KDataset(root="graph-task/dataset/ZINC/")

print_statistics(dataset,type="train")

num_eigs = args.num_eigs#gives the dimension of the embedding or/ the number of eigenvectors we calculate
p = args.p_laplacian
epochs = args.epochs

train_dataset = preprocess_dataset(dataset,num_eigs,epochs,p,device)

trid=list(range(0,10000))
vlid=list(range(10000,11000))
tsid=list(range(11000,12000))

train_loader = DataLoader(dataset[trid], batch_size=64, shuffle=True)
val_loader = DataLoader(dataset[vlid], batch_size=64, shuffle=False)
test_loader = DataLoader(dataset[tsid], batch_size=64, shuffle=False)


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

        nn1 = Sequential(Linear(dataset.num_features, 64), ReLU(), Linear(64, 64))
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
        self.conv1 = GCNConv(dataset.num_features, neuron, cached=False)
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


class GNNML1(nn.Module):
    def __init__(self):
        super(GNNML1, self).__init__()
        
        # number of neuron
        nout=16        
        # three part concatenate or sum?
        self.concat=True

        if self.concat:
            nin=3*nout
        else:
            nin=nout
        self.conv11 = SpectConv(dataset.num_features, nout,selfconn=False)
        self.conv21 = SpectConv(nin, nout, selfconn=False)
        self.conv31 = SpectConv(nin, nout, selfconn=False)
        self.conv41 = SpectConv(nin, nout, selfconn=False)
        
        
        self.fc11 = torch.nn.Linear(dataset.num_features, nout)
        self.fc21 = torch.nn.Linear(nin, nout)
        self.fc31 = torch.nn.Linear(nin, nout)
        self.fc41 = torch.nn.Linear(nin, nout)
        
        self.fc12 = torch.nn.Linear(dataset.num_features, nout)
        self.fc22 = torch.nn.Linear(nin, nout)
        self.fc32 = torch.nn.Linear(nin, nout)
        self.fc42 = torch.nn.Linear(nin, nout)

        self.fc13 = torch.nn.Linear(dataset.num_features, nout)
        self.fc23 = torch.nn.Linear(nin, nout)
        self.fc33 = torch.nn.Linear(nin, nout)
        self.fc43 = torch.nn.Linear(nin, nout)
        
 
        self.fc1 = torch.nn.Linear(nin, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        

    def forward(self, data):
        x=data.x  
              
        edge_index=data.edge_index
        edge_attr=torch.ones(edge_index.shape[1],1).to('cuda')
        
        if self.concat:            
            x = torch.cat([F.relu(self.fc11(x)), F.relu(self.conv11(x, edge_index,edge_attr)),F.relu(self.fc12(x)*self.fc13(x))],1)
            x = torch.cat([F.relu(self.fc21(x)), F.relu(self.conv21(x, edge_index,edge_attr)),F.relu(self.fc22(x)*self.fc23(x))],1)
            x = torch.cat([F.relu(self.fc31(x)), F.relu(self.conv31(x, edge_index,edge_attr)),F.relu(self.fc32(x)*self.fc33(x))],1)
            x = torch.cat([F.relu(self.fc41(x)), F.relu(self.conv41(x, edge_index,edge_attr)),F.relu(self.fc42(x)*self.fc43(x))],1)
        else: 
                      
            x = F.relu(self.fc11(x)+self.conv11(x, edge_index,edge_attr)+self.fc12(x)*self.fc13(x))
            x = F.relu(self.fc21(x)+self.conv21(x, edge_index,edge_attr)+self.fc22(x)*self.fc23(x))
            x = F.relu(self.fc31(x)+self.conv31(x, edge_index,edge_attr)+self.fc32(x)*self.fc33(x))
            x = F.relu(self.fc41(x)+self.conv41(x, edge_index,edge_attr)+self.fc42(x)*self.fc43(x))        

        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class GNNML3(nn.Module):
    def __init__(self):
        super(GNNML3, self).__init__()
        

        # number of neuron for for part1 and part2
        nout1=30
        nout2=2

        nin=nout1+nout2
        ne=dataset.data.edge_attr2.shape[1]
        ninp=dataset.num_features

        self.conv1=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=ninp,nout1=nout1,nout2=nout2)
        self.conv2=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)
        self.conv3=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)
        self.conv4=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)        

        self.fc1 = torch.nn.Linear(nin, 32)
        self.fc2 = torch.nn.Linear(32, 1)
        

    def forward(self, data):
        x=data.x
        edge_index=data.edge_index2
        edge_attr=data.edge_attr2


        x=(self.conv1(x, edge_index,edge_attr))
        x=(self.conv2(x, edge_index,edge_attr))
        x=(self.conv3(x, edge_index,edge_attr))
        x=(self.conv4(x, edge_index,edge_attr))

        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)



model = GcnNet().to(device)   # GatNet  ChebNet  GcnNet  GinNet  MlpNet  PPGN GNNML1 GNNML3 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(get_n_params(model))

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

    return L/len(train_loader)

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
    return L/len(test_loader), Lv/len(val_loader)

bval=1000
btest=0
for epoch in range(1, 401):
    trloss=train(epoch)
    test_loss,val_loss = test()
    if bval>val_loss:
        bval=val_loss
        btest=test_loss
        
    #print('Epoch: {:02d}, trloss: {:.4f},  Val: {:.4f}, Test: {:.4f}'.format(epoch,trloss,val_acc, test_acc))
    print('Epoch: {:02d}, trloss: {:.4f},  Valloss: {:.4f}, Testloss: {:.4f}, best test loss: {:.4f}'.format(epoch,trloss,val_loss,test_loss,btest))

