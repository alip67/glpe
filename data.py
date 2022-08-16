"""
Code partially copied from 'Diffusion Improves Graph Learning' repo https://github.com/klicperajo/gdc/blob/master/data.py
"""

import os

import numpy as np

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
# from graph_rewiring import get_two_hop, apply_gdc
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from sklearn.model_selection import ShuffleSplit
# from graph_rewiring import make_symmetric, apply_pos_dist_rewire
from heterophilic import WebKB, WikipediaNetwork, Actor
from torch_geometric.utils import to_scipy_sparse_matrix 
from utils import ROOT_DIR, generate_sparse_adj

DATA_PATH = f'{ROOT_DIR}/data'


# def rewire(data, opt, data_dir):
#   rw = opt['rewiring']
#   if rw == 'two_hop':
#     data = get_two_hop(data)
#   elif rw == 'gdc':
#     data = apply_gdc(data, opt)
#   elif rw == 'pos_enc_knn':
#     data = apply_pos_dist_rewire(data, opt, data_dir)
#   return data


def get_dataset(opt: dict, data_dir, use_lcc: bool = False) -> InMemoryDataset:
  
  #todo for Ali to add wikics dataset
  
  ds = opt['dataset']
  path = os.path.join(data_dir, ds)
  if ds in ['Cora', 'Citeseer', 'Pubmed']:
    dataset = Planetoid(path, ds)
  elif ds in ['Computers', 'Photo']:
    dataset = Amazon(path, ds)
  elif ds == 'CoauthorCS':
    dataset = Coauthor(path, 'CS')
  elif ds in ['cornell', 'texas', 'wisconsin']:
    dataset = WebKB(root=path, name=ds, transform=T.NormalizeFeatures())
  elif ds in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(root=path, name=ds, transform=T.NormalizeFeatures())
  elif ds == 'film':
    dataset = Actor(root=path, transform=T.NormalizeFeatures())
  elif ds == 'ogbn-arxiv':
    dataset = PygNodePropPredDataset(name=ds, root=path,
                                     transform=T.ToSparseTensor())
    use_lcc = False  # never need to calculate the lcc with ogb datasets
  else:
    raise Exception('Unknown dataset.')

  if use_lcc:
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
      x=x_new,
      edge_index=torch.LongTensor(edges),
      y=y_new,
      train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
      test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
      val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
    )
    dataset.data = data
  # if opt['rewiring'] is not None:
  #   dataset.data = rewire(dataset.data, opt, data_dir)
  train_mask_exists = True
  try:
    dataset.data.train_mask
  except AttributeError:
    train_mask_exists = False

  if ds == 'ogbn-arxiv':
    split_idx = dataset.get_idx_split()
    ei = to_undirected(dataset.data.edge_index)
    data = Data(
    x=dataset.data.x,
    edge_index=ei,
    y=dataset.data.y,
    train_mask=split_idx['train'],
    test_mask=split_idx['test'],
    val_mask=split_idx['valid'])
    dataset.data = data
    train_mask_exists = True


  # Add the sparse tensor represnattion of the Adjancency
  adj = to_scipy_sparse_matrix(dataset.data.edge_index)
  adj_t = generate_sparse_adj(adj)
  adj_t = adj_t.to_symmetric()
  dataset.data.adj_t = adj_t

  #todo this currently breaks with heterophilic datasets if you don't pass --geom_gcn_splits
  if (use_lcc or not train_mask_exists) and not opt['geom_gcn_splits']:
    dataset.data = set_fixed_train_val_test_split(
      12345,
      dataset.data,
      num_development=5000 if ds == "CoauthorCS" else 1500)

  return dataset




def update_dataset(dataset, new_features) -> InMemoryDataset:
  x_new = torch.cat((dataset.data.x, new_features),1)
  ei = to_undirected(dataset.data.edge_index)
  data = Data(
    x=x_new,
    edge_index=ei,
    y=dataset.data.y,
    train_mask=dataset.data.train_mask,
    test_mask=dataset.data.test_mask,
    val_mask=dataset.data.val_mask
  )
  dataset.data = data
  train_mask_exists = True

  adj = to_scipy_sparse_matrix(dataset.data.edge_index)
  adj_t = generate_sparse_adj(adj)
  adj_t = adj_t.to_symmetric()
  dataset.data.adj_t = adj_t

  # datal = [Data(xx,cora_dataset[0].edge_index)]
        
  # datal[0].train_mask = cora_dataset[0].train_mask
  # datal[0].val_mask = cora_dataset[0].val_mask

  # datal[0].test_mask = cora_dataset[0].test_mask
  # datal[0].y  = cora_dataset[0].y
  try:
    dataset.data.train_mask
  except AttributeError:
    train_mask_exists = False

  return dataset



def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = dataset.data.edge_index.numpy()
  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
  remaining_nodes = set(range(dataset.data.x.shape[0]))
  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(dataset, start)
    comps.append(comp)
    remaining_nodes = remaining_nodes.difference(comp)
  return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 1
  return mapper


def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]


def set_fixed_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
  rnd_state = np.random.RandomState(seed)
  num_nodes = data.y.shape[0]
  development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
  test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

  train_idx = []
  rnd_state = np.random.RandomState(seed)
  for c in range(data.y.max() + 1):
    class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
    train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

  val_idx = [i for i in development_idx if i not in train_idx]

  def get_mask(idx):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

  data.train_mask = get_mask(train_idx)
  data.val_mask = get_mask(val_idx)
  data.test_mask = get_mask(test_idx)

  return data
