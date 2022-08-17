import os
import sys
import torch
import numpy as np
from datetime import datetime

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

"""
How To:
Example for running from command line:
python <path_to>/ProvablyPowerfulGraphNetworks/main_scripts/main_10fold_experiment.py --config=configs/10fold_config.json --dataset_name=COLLAB
"""
# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

from data_loader.data_generator import DataGenerator
from models.model_wrapper import ModelWrapper
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils import doc_utils
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    parser = argparse.ArgumentParser(description='PGNN')
    parser.add_argument('--dataset_name', type=str, default='MUTAG',
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--config', type=str, default="configs/10fold_config.json",
                        help='filename to config path')
    args = parser.parse_args()
    try:
        # args = get_args()
        config = process_config(args.config, args.dataset_name)

    except Exception as e:
        print("missing or invalid arguments {}".format(e))
        exit(0)

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # TODO uncomment only for CUDA error debugging
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(100)
    np.random.seed(100)
    # torch.backends.cudnn.deterministic = True  # can impact performance
    # torch.backends.cudnn.benchmark = False  # can impact performance

    print("lr = {0}".format(config.hyperparams.learning_rate))
    print("decay = {0}".format(config.hyperparams.decay_rate))
    print(config.architecture)
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    doc_utils.doc_used_config(config)
    for exp in range(1, config.num_exp+1):
        for fold in range(1, 11):
            print("Experiment num = {0}\nFold num = {1}".format(exp, fold))
            # create your data generator
            config.num_fold = fold
            data = DataGenerator(config,device)
            # create an instance of the model you want
            model_wrapper = ModelWrapper(config, data,device)
            # create trainer and pass all the previous components to it
            trainer = Trainer(model_wrapper, data, config,device)
            # here you train your model
            trainer.train()

    doc_utils.summary_10fold_results(config.summary_dir)


if __name__ == '__main__':
    start = datetime.now()
    main()
    print('Runtime: {}'.format(datetime.now() - start))
