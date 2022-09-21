import torch
import torch.nn.functional as F
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from scipy import sparse as sp
import numpy as np
import networkx as nx


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

# The dataset pickle and index files are in ./data/molecules/ dir
# [<split>.pickle and <split>.index; for split 'train', 'val' and 'test']




class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        
        with open(data_dir + "/%s.pickle" % self.split,"rb") as f:
            self.data = pickle.load(f)

        if self.num_graphs in [10000, 1000]:
            # loading the sampled indices from file ./zinc_molecules/<split>.index
            with open(data_dir + "/%s.index" % self.split,"r") as f:
                data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
                self.data = [ self.data[i] for i in data_idx[0] ]

            assert len(self.data)==num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
        
        """
        data is a list of Molecule dict objects with following attributes
        
          molecule = data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """
        
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.data:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]
    
    
class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Zinc'):
        t0 = time.time()
        self.name = name
        
        self.num_atom_type = 28 # known meta-info about the zinc dataset; can be calculated as well
        self.num_bond_type = 4 # known meta-info about the zinc dataset; can be calculated as well
        
        data_dir='./data/molecules'
        
        if self.name == 'ZINC-full':
            data_dir='./data/molecules/zinc_full'
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=220011)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=24445)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=5000)
        else:            
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=10000)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=1000)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=1000)
        print("Time taken: {:.4f}s".format(time.time()-t0))
        


def add_eig_vec(g, pos_enc_dim):
    """
     Graph positional encoding v/ Laplacian eigenvectors
     This func is for eigvec visualization, same code as positional_encoding() func,
     but stores value in a diff key 'eigvec'
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['eigvec'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    # zero padding to the end if n < pos_enc_dim
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['eigvec'] = F.pad(g.ndata['eigvec'], (0, pos_enc_dim - n + 1), value=float('0'))

    return g


def lap_positional_encoding(g, pos_enc_dim, tau=0):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    #A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    #N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    #L = sp.eye(g.number_of_nodes()) - N * A * N
    #L = L.toarray()

    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float).toarray() + tau / g.number_of_nodes()
    N = np.diag((dgl.backend.asnumpy(g.in_degrees()).clip(1) + tau) ** -0.5)
    L = np.eye(g.number_of_nodes()) - N @ A @ N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L)
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 

    # zero padding to the end if n < pos_enc_dim
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['pos_enc'] = F.pad(g.ndata['pos_enc'], (0, pos_enc_dim - n + 1), value=float('0'))

    g.ndata['eigvec'] = g.ndata['pos_enc']
    
    return g


def init_positional_encoding(g, pos_enc_dim, type_init):
    """
        Initializing positional encoding with RWPE
    """
    
    n = g.number_of_nodes()
    """
    if type_init == 'rand_walk':
        # Geometric diffusion features with Random Walk
        A = g.adjacency_matrix(scipy_fmt="csr")
        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
        RW = A * Dinv  
        M = RW
        
        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc-1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pos_enc'] = PE  
    """
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float).toarray()
    D = np.diag((dgl.backend.asnumpy(g.in_degrees()).clip(1)))
    L = D-A

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L)
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 

    # zero padding to the end if n < pos_enc_dim
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['pos_enc'] = F.pad(g.ndata['pos_enc'], (0, pos_enc_dim - n + 1), value=float('0'))

    g.ndata['eigvec'] = g.ndata['pos_enc']

    return g

def get_graph_props(A, normalize_L='none', shift_to_zero_diag=False, k=5):
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

    eigval, eigvec = scipy.sparse.linalg.eigs(L, k) #np.linalg.eigh(L)
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


    def set_parameters(self, init_eig):
        self.initeigv = init_eig
        self.weight = geoopt.ManifoldParameter(
            init_eig.clone(), manifold=self.ball
        )
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


def training_loop1(model, optimizer, sched,W, epochs=200):
    "Training loop for torch model."
    losses = []
    for i in range(epochs):
        preds = model(W)
        loss = preds
        #print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if sched is not None:
            sched.step()
        losses.append(loss)  
    return losses

def get_p_eigvals(X, F, p):
    n = X.shape[0]
    K = F.shape[1]
    f = F
    FFF = torch.norm(F, p,dim=0)
    FFF = torch.pow(FFF,-1)
    FFF.unsqueeze_(-1)
    FFF.unsqueeze_(-1)
    FFF.unsqueeze_(-1)
    FFF = FFF.repeat(1,n,1,1)
    FFF = torch.pow(FFF,p)

    FF = f.repeat(1,n)
    FF = FF.view(n,n,K)
    FF = FF.transpose(2,0)

    GG =FF.transpose(1,2)

    KK = FF - GG 
    KKK = KK.unsqueeze(dim=-1)
    KKK = torch.pow(torch.abs(KKK),p)

    X = X.unsqueeze(dim=1)

    KKK = KKK.type(torch.float64)
    X = X.type(torch.float64)

    LL = torch.matmul(X, KKK)
    b = torch.matmul(LL.float(),FFF.float())
    b = torch.sum(b, dim=1).squeeze_()
    return b

def p_lap_positional_encoding(g, pos_enc_dim, epochs,p,device):

    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        K = n
    else: 
        K = pos_enc_dim+1 
    

    """
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')     
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float).todense()
    A = np.asarray(A)

    D, L, L_inv, eigval,eigvec = get_graph_props(A,normalize_L='none')
    
    #We transform our eigenvectors into an orthonormalbasis such that it is in the Stiefel manifold
    
    #Just removed for L_2 LPE
    
    hi = get_orthonromal_eigvec(eigval,eigvec)
    
    """
    tau=0
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float).toarray() + tau / g.number_of_nodes()
    D = np.diag((dgl.backend.asnumpy(g.in_degrees()).clip(1) + tau))
    L = D-A

    # Eigenvectors with PYTORCH
    # Reason: pytorch has the fct torch.linalg.eigh which directly gives an ONB.
    eigval, eigvec = torch.linalg.eigh(torch.tensor(L))
    eigval = eigval.numpy()
    eigvec = eigvec.numpy()
    
    idx = eigval.argsort() # increasing order
    eigval, hi = eigval[idx], np.real(hi[:,idx])
    hi = eigvec
    
    start = timer()

    for i in range(1,5):
        n = eigval.shape[0]
        p = 2- (i/10)

        W = torch.tensor(A).float()#.to(device)
        if i == 1:
            F_ = torch.tensor(hi[:, :K]).float()#.to(device) #We can use previous outputs weight
        else: F_ = m.weight.clone()

        m = Model_RGD(F_, p, n, K, ball = geoopt.EuclideanStiefelExact())#.to(device)

        # Instantiate optimizer
        optimizer = geoopt.optim.RiemannianAdam(m.parameters(), lr=1e-2)


        decayRate = 0.99
        
        my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.25)


        #Learn the 1-eigenvector. It is then given by m.weight
        losses = training_loop1(m, optimizer,my_lr_scheduler,W, 200)  
        
    end = timer()
    print(end - start, " Second")
    
    m#.to('cpu')
    # xx = torch.cat((data.x, m.weight[:,1:pos_enc_dim]),1)
      
    p_eigs = m.weight[:,1:K]
    
    #Order the p-eigenvector ascending with repect to the eigenvalues
    p_eigvals = get_p_eigvals(W.to('cpu'), p_eigs, p)
    eigidx = torch.argsort(p_eigvals)
    p_eigvals = p_eigvals[eigidx]
    p_eigs = p_eigs[:, eigidx]

    g.ndata['pos_enc'] = p_eigs

    # zero padding to the end if n < pos_enc_dim
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['pos_enc'] = F.pad(g.ndata['pos_enc'], (0, pos_enc_dim - n + 1), value=float('0'))

    g.ndata['eigvec'] = g.ndata['pos_enc']
    
    return g




def make_full_graph(g, adaptive_weighting=None):

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    #Here we copy over the node feature data and laplace encodings
    full_g.ndata['feat'] = g.ndata['feat']
    
    try:
        full_g.ndata['pos_enc'] = g.ndata['pos_enc']
    except:
        pass
    
    try:
        full_g.ndata['eigvec'] = g.ndata['eigvec']
    except:
        pass
    
    #Populate edge features w/ 0s
    full_g.edata['feat']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    full_g.edata['real']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    
    #Copy real edge data over
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 

    
    # This code section only apply for GraphiT --------------------------------------------
    if adaptive_weighting is not None:
        p_steps, gamma = adaptive_weighting
    
        n = g.number_of_nodes()
        A = g.adjacency_matrix(scipy_fmt="csr")
        
        # Adaptive weighting k_ij for each edge
        if p_steps == "qtr_num_nodes":
            p_steps = int(0.25*n)
        elif p_steps == "half_num_nodes":
            p_steps = int(0.5*n)
        elif p_steps == "num_nodes":
            p_steps = int(n)
        elif p_steps == "twice_num_nodes":
            p_steps = int(2*n)

        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        I = sp.eye(n)
        L = I - N * A * N

        k_RW = I - gamma*L
        k_RW_power = k_RW
        for _ in range(p_steps - 1):
            k_RW_power = k_RW_power.dot(k_RW)

        k_RW_power = torch.from_numpy(k_RW_power.toarray())

        # Assigning edge features k_RW_eij for adaptive weighting during attention
        full_edge_u, full_edge_v = full_g.edges()
        num_edges = full_g.number_of_edges()

        k_RW_e_ij = []
        for edge in range(num_edges):
            k_RW_e_ij.append(k_RW_power[full_edge_u[edge], full_edge_v[edge]])

        full_g.edata['k_RW'] = torch.stack(k_RW_e_ij,dim=-1).unsqueeze(-1).float()
    # --------------------------------------------------------------------------------------
        
    return full_g


class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading ZINC datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        data_dir = f'{ROOT_DIR}/molecules/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(list(np.array(labels))).unsqueeze(1)
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        batched_graph = dgl.batch(graphs)       
        
        return batched_graph, labels, snorm_n
    

    def _add_lap_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [lap_positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [lap_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [lap_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]
    
    def _add_eig_vecs(self, pos_enc_dim):

        # This is used if we visualize the eigvecs
        self.train.graph_lists = [add_eig_vec(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [add_eig_vec(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [add_eig_vec(g, pos_enc_dim) for g in self.test.graph_lists]
    
    def _init_positional_encodings(self, pos_enc_dim, type_init):
        
        # Initializing positional encoding randomly with l2-norm 1
        self.train.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in self.train.graph_lists]
        self.val.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in self.val.graph_lists]
        self.test.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in self.test.graph_lists]

    def _add_p_positional_encodings(self, pos_enc_dim, epochs,p,use_cache,device):
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

        if use_cache:
            self.train.graph_lists = torch.load(f'{ROOT_DIR}/p-embeddings/zinc_train_emb{pos_enc_dim}_p{p}.pt')
            self.val.graph_lists = torch.load(f'{ROOT_DIR}/p-embeddings/zinc_valid_emb{pos_enc_dim}_p{p}.pt')
            self.test.graph_lists = torch.load(f'{ROOT_DIR}/p-embeddings/zinc_test_emb{pos_enc_dim}_p{p}.pt')
        else: 
            # Initializing p-positional encoding eith model RGD
            self.val.graph_lists = [p_lap_positional_encoding(g, pos_enc_dim, epochs,p,device) for g in self.val.graph_lists]
            torch.save(self.val.graph_lists, f'{ROOT_DIR}/p-embeddings/zinc_valid_emb{pos_enc_dim}_p{p}.pt')
            self.train.graph_lists = [p_lap_positional_encoding(g, pos_enc_dim, epochs,p,device) for g in self.train.graph_lists]         
            self.test.graph_lists = [p_lap_positional_encoding(g, pos_enc_dim, epochs,p,device) for g in self.test.graph_lists]
            torch.save(self.train.graph_lists, f'{ROOT_DIR}/p-embeddings/zinc_train_emb{pos_enc_dim}_p{p}.pt')            
            torch.save(self.test.graph_lists, f'{ROOT_DIR}/p-embeddings/zinc_test_emb{pos_enc_dim}_p{p}.pt')
        
    def _make_full_graph(self, adaptive_weighting=None):
        self.train.graph_lists = [make_full_graph(g, adaptive_weighting) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g, adaptive_weighting) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g, adaptive_weighting) for g in self.test.graph_lists]




