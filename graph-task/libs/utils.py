import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected
import numpy as np
import networkx as nx
import pickle
import os
import scipy.io as sio
from scipy.special import comb


from torch_geometric.utils import to_networkx, to_dense_adj
import geoopt
from timeit import default_timer as timer
import numpy as np
import scipy.io
from torch_geometric.data import InMemoryDataset
import torch.nn as nn
from utils_1 import get_graph_props, make_2d_graph


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp



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
            self.initeigv.clone(), manifold=self.ball
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
        #self.weight = nn.Parameter(self.initeigv.clone())
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

def postprocess_dataset(dataset,num_eigs,epochs,p,device): 
    datal = []
    for data in dataset:
        #Preprocessing

        cora_adj = to_dense_adj(data.edge_index)
        cora_adj.squeeze_()

        A = cora_adj.numpy()
        D, L, L_inv, eigval,eigvec = get_graph_props(A,normalize_L='none')
        
        #We transform our eigenvectors into an orthonormalbasis such that it is in the Stiefel manifold
        
        #Just removed for L_2 LPE
        
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
        xx = torch.cat((data.x, m.weight[:,1:num_eigs]),1)
        
        #xx = torch.cat((data.x, torch.tensor(eigvec)[:,1:3]),1)
        
        # xx = torch.cat((data.x, torch.tensor(eigvec[:,1:9])),1)

        #Didnt know how to pretransform the features of CORA; This is my workaround
        datal.append(Data(xx,data.edge_index, y=data.y, edge_attr=data.edge_attr, batch = data.batch))
        
    # data, slices = dataset.collate(datal)
    # torch.save((data, slices), f'dataset_zinc_p{p}.pt')
    return datal


class PtcDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        
        super(PtcDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["ptc.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]        
        F=a['F'][0]
        #Y=a['y'].astype(np.float32) #(a['y']+1)//2
        Y=a['Y'].astype(np.int)
        Y=Y[:,0]

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.tensor(F[i]).type(torch.float32)                 
            y=torch.tensor([Y[i]]) #.type(torch.float32)             
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class ProteinsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,contfeat=False):
        self.contfeat=contfeat
        super(ProteinsDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["proteins.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]        
        F=a['F'][0]
        #Y=a['y'].astype(np.float32) #(a['y']+1)//2
        Y=a['Y'].astype(np.int)
        Y=Y[:,0]

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            if self.contfeat:
                #ind=list(set(range(0,F[i].shape[1]))-set([3,4]))
                tmp=F[i]#[:,ind]
                x=torch.tensor(tmp).type(torch.float32)
            else:
                x=torch.tensor(F[i][:,0:3]).type(torch.float32) 
            y=torch.tensor([Y[i]]) #.type(torch.float32)             
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class EnzymesDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,contfeat=False):
        self.contfeat=contfeat
        super(EnzymesDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["enzymes.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]        
        F=a['F'][0]
        #Y=a['y'].astype(np.float32) #(a['y']+1)//2
        Y=a['Y'][0].astype(np.int)

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            if self.contfeat:
                #ind=list(set(range(0,F[i].shape[1]))-set([3,4]))
                tmp=F[i]#[:,ind]
                x=torch.tensor(tmp).type(torch.float32)
            else:
                x=torch.tensor(F[i][:,0:3]).type(torch.float32) 
            y=torch.tensor([Y[i]]) #.type(torch.float32)             
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class MutagDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MutagDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["mutag.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        TA=a['TA'][0]
        F=a['F'][0]
        #Y=a['y'].astype(np.float32) #(a['y']+1)//2
        Y=((a['y']+1)//2).astype(np.float32)

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.tensor(F[i]).type(torch.float32) 
            y=torch.tensor(Y[i]).type(torch.float32)             
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Zinc12KDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Zinc12KDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["Zinc.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) 
        # list of adjacency matrix
        F=a['F'][0]
        A=a['E'][0]
        Y=a['Y']
        nmax=37
        ntype=21
        maxdeg=4

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.zeros(A[i].shape[0],ntype+maxdeg)
            deg=(A[i]>0).sum(1)
            for j in range(F[i][0].shape[0]):
                # put atom code
                x[j,F[i][0][j]]=1
                # put degree code
                x[j,-int(deg[j])]=1
            y=torch.tensor(Y[i,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def post_process(self, dataset,num_eigs,epochs,p,device):
        datal = postprocess_dataset(dataset,num_eigs,epochs,p,device)
        data, slices = self.collate(datal)
        dataset.original_num_node_features = dataset.num_node_features
        dataset.data = data
        dataset.slices = slices
        dataset._data_list = datal
        return dataset

class BandClassDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(BandClassDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["bandclass.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A']
        F=a['F']
        Y=a['Y']
        F=np.expand_dims(F,2)

        data_list = []
        for i in range(len(A)):
            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.tensor(F[i,:,:]) 
            y=torch.tensor(Y[i,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=y))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class TwoDGrid30(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TwoDGrid30, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["TwoDGrid30.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A']
        # list of output
        F=a['F']
        F=F.astype(np.float32)

        data_list = []
        E=np.where(A>0)
        edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
        x=torch.tensor(F[:,0:1])
        y=torch.tensor(F[:,1:4])
        mask=torch.tensor(F[:,12:13])
        data_list.append(Data(edge_index=edge_index, x=x, y=y,mask=mask))
        x=torch.tensor(F[:,4:5])
        y=torch.tensor(F[:,5:8])
        data_list.append(Data(edge_index=edge_index, x=x, y=y,mask=mask)) 
        x=torch.tensor(F[:,8:9])
        y=torch.tensor(F[:,9:12])
        data_list.append(Data(edge_index=edge_index, x=x, y=y,mask=mask))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A'][0]
        # list of output
        Y=a['F']

        data_list = []
        for i in range(len(A)):
            a=A[i]
            A2=a.dot(a)
            A3=A2.dot(a)
            tri=np.trace(A3)/6
            tailed=((np.diag(A3)/2)*(a.sum(0)-2)).sum()
            cyc4=1/8*(np.trace(A3.dot(a))+np.trace(A2)-2*A2.sum())
            cus= a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()

            deg=a.sum(0)
            star=0
            for j in range(a.shape[0]):
                star+=comb(int(deg[j]),3)

            expy=torch.tensor([[tri,tailed,star,cyc4,cus]])

            E=np.where(A[i]>0)
            edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
            x=torch.ones(A[i].shape[0],1)
            #y=torch.tensor(Y[i:i+1,:])            
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/GRAPHSAT.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Grapg8cDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Grapg8cDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["graph8c.g6"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]   
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i,datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  #sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]   
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i,datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(),1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1,0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SpectralDesign(object):   

    def __init__(self,nmax=0,recfield=1,dv=5,nfreq=5,adddegree=False,laplacien=True,addadj=False,vmax=None):
        # receptive field. 0: adj, 1; adj+I, n: n-hop area 
        self.recfield=recfield  
        # b parameter
        self.dv=dv
        # number of sampled point of spectrum
        self.nfreq=  nfreq
        # if degree is added to node feature
        self.adddegree=adddegree
        # use laplacian or adjacency for spectrum
        self.laplacien=laplacien
        # add adjacecny as edge feature
        self.addadj=addadj
        # use given max eigenvalue
        self.vmax=vmax

        # max node for PPGN algorithm, set 0 if you do not use PPGN
        self.nmax=nmax    

    def __call__(self, data):

        n =data.x.shape[0]     
        nf=data.x.shape[1]  


        data.x=data.x.type(torch.float32)  
               
        nsup=self.nfreq+1
        if self.addadj:
            nsup+=1
            
        A=np.zeros((n,n),dtype=np.float32)
        SP=np.zeros((nsup,n,n),dtype=np.float32) 
        A[data.edge_index[0],data.edge_index[1]]=1
        
        if self.adddegree:
            data.x=torch.cat([data.x,torch.tensor(A.sum(0)).unsqueeze(-1)],1)

        # calculate receptive field. 0: adj, 1; adj+I, n: n-hop area
        if self.recfield==0:
            M=A
        else:
            M=(A+np.eye(n))
            for i in range(1,self.recfield):
                M=M.dot(M) 

        M=(M>0)

        
        d = A.sum(axis=0) 
        # normalized Laplacian matrix.
        dis=1/np.sqrt(d)
        dis[np.isinf(dis)]=0
        dis[np.isnan(dis)]=0
        D=np.diag(dis)
        nL=np.eye(D.shape[0])-(A.dot(D)).T.dot(D)
        V,U = np.linalg.eigh(nL) 
        V[V<0]=0
        # keep maximum eigenvalue for Chebnet if it is needed
        data.lmax=V.max().astype(np.float32)

        if not self.laplacien:        
            V,U = np.linalg.eigh(A)

        # design convolution supports
        vmax=self.vmax
        if vmax is None:
            vmax=V.max()

        freqcenter=np.linspace(V.min(),vmax,self.nfreq)
        
        # design convolution supports (aka edge features)         
        for i in range(0,len(freqcenter)): 
            SP[i,:,:]=M* (U.dot(np.diag(np.exp(-(self.dv*(V-freqcenter[i])**2))).dot(U.T))) 
        # add identity
        SP[len(freqcenter),:,:]=np.eye(n)
        # add adjacency if it is desired
        if self.addadj:
            SP[len(freqcenter)+1,:,:]=A
           
        # set convolution support weigths as an edge feature
        E=np.where(M>0)
        data.edge_index2=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
        data.edge_attr2 = torch.Tensor(SP[:,E[0],E[1]].T).type(torch.float32)  

        # set tensor for Maron's PPGN         
        if self.nmax>0:       
            H=torch.zeros(1,nf+2,self.nmax,self.nmax)
            H[0,0,data.edge_index[0],data.edge_index[1]]=1 
            H[0,1,0:n,0:n]=torch.diag(torch.ones(data.x.shape[0]))
            for j in range(0,nf):      
                H[0,j+2,0:n,0:n]=torch.diag(data.x[:,j])
            data.X2= H 
            M=torch.zeros(1,2,self.nmax,self.nmax)
            for i in range(0,n):
                M[0,0,i,i]=1
            M[0,1,0:n,0:n]=1-M[0,0,0:n,0:n]
            data.M= M        

        return data

class PPGNAddDegree(object):   

    def __init__(self,nmax=0,adddegree=True,):
        
        # if degree is added to node feature
        self.adddegree=adddegree       

        # max node for PPGN algorithm, set 0 if you do not use PPGN
        self.nmax=nmax    

    def __call__(self, data):

        n =data.x.shape[0]     
        nf=data.x.shape[1]  


        data.x=data.x.type(torch.float32)
        A=np.zeros((n,n),dtype=np.float32)        
        A[data.edge_index[0],data.edge_index[1]]=1
        
        if self.adddegree:
            data.x=torch.cat([data.x,torch.tensor(A.sum(0)).unsqueeze(-1)],1)
            
        # set tensor for Maron's PPGN         
        if self.nmax>0:       
            H=torch.zeros(1,nf+2,self.nmax,self.nmax)
            H[0,0,data.edge_index[0],data.edge_index[1]]=1 
            H[0,1,0:n,0:n]=torch.diag(torch.ones(data.x.shape[0]))
            for j in range(0,nf):      
                H[0,j+2,0:n,0:n]=torch.diag(data.x[:,j])
            data.X2= H 
            M=torch.zeros(1,2,self.nmax,self.nmax)
            for i in range(0,n):
                M[0,0,i,i]=1
            M[0,1,0:n,0:n]=1-M[0,0,0:n,0:n]
            data.M= M        

        return data
    
class DegreeMaxEigTransform(object):   

    def __init__(self,adddegree=True,maxdeg=40,addposition=False,addmaxeig=True):
        self.adddegree=adddegree
        self.maxdeg=maxdeg
        self.addposition=addposition
        self.addmaxeig=addmaxeig

    def __call__(self, data):

        n=data.x.shape[0] 
        A=np.zeros((n,n),dtype=np.float32)        
        A[data.edge_index[0],data.edge_index[1]]=1         
        if self.adddegree:
            data.x=torch.cat([data.x,torch.tensor(1/self.maxdeg*A.sum(0)).unsqueeze(-1)],1)
        if self.addposition:
            data.x=torch.cat([data.x,data.pos],1)

        if self.addmaxeig:
            d = A.sum(axis=0) 
            # normalized Laplacian matrix.
            dis=1/np.sqrt(d)
            dis[np.isinf(dis)]=0
            dis[np.isnan(dis)]=0
            D=np.diag(dis)
            nL=np.eye(D.shape[0])-(A.dot(D)).T.dot(D)
            V,U = np.linalg.eigh(nL)               
            vmax=np.abs(V).max()
            # keep maximum eigenvalue for Chebnet if it is needed
            data.lmax=vmax.astype(np.float32)        
        return data    
    
