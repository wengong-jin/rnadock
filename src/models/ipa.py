import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.metrics import average_precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sru import SRUpp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from binary_label_metrics import BinaryLabelMetrics
from einops import repeat
from invariant_point_attention import InvariantPointAttention

sys.path.append("/home/dnori/rnadock/src/data")
from data import ProteinDataset, ProteinBinaryDataset, ProteinMulticlassDataset

class IPAClassificationModel(nn.Module):

    def __init__(self, args):
        super(FAEClassificationModel, self).__init__()
        self.blm = BinaryLabelMetrics()
        self.ipa = InvariantPointAttention(
            dim = args.encoder_hidden_size,   # single representation dimension
            heads = 8,                       # number of attention heads
            require_pairwise_repr = False # no pairwise representations
        )
        self.linear_layers = nn.Sequential(
                nn.Linear(args.encoder_hidden_size, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(args.mlp_hidden_size, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(args.mlp_hidden_size, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(args.mlp_hidden_size, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(args.mlp_hidden_size, args.mlp_output_dim),
        ).cuda().float()
