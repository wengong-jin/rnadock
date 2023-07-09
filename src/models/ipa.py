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
        super(IPAClassificationModel, self).__init__()
        self.blm = BinaryLabelMetrics()
        self.protein_encoder = InvariantPointAttention(
            dim = args.esm_emb_size + 3,   # single representation dimension
            heads = 3,                       # number of attention heads
            require_pairwise_repr = False # no pairwise representations
        ).cuda().float()
        self.linear_layers = nn.Sequential(
                nn.Linear(args.esm_emb_size + 3, 1000),
                nn.ReLU(),
                nn.Linear(1000, 500),
                nn.ReLU(),
                nn.Linear(500, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(args.mlp_hidden_size, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(args.mlp_hidden_size, args.mlp_output_dim),
        ).cuda().float()

        self.esm_model, self.esm_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        self.esm_model.eval()
        self.args = args

    def forward(self, prot_X, ligand_X, seqs, ligand_seqs, y_batch, res_ids, max_prot_seq_len, max_ligand_seq_len, token_repr=None):
        
        esm_input = []
        pad_mask_atom = torch.from_numpy(np.ones((prot_X.shape[0],max_prot_seq_len)))
        pad_mask_residue = torch.from_numpy(np.ones((prot_X.shape[0],max_prot_seq_len//3)))
        # pad_mask = torch.from_numpy(np.zeros((prot_X.shape[0],max_prot_seq_len)))
        for s in range(len(seqs)):
            sequence = seqs[s] + "<pad>"*(max_prot_seq_len - len(seqs[s]))
            pad_mask_atom[s,len(seqs[s]):] = 0
            pad_mask_residue[s, len(seqs[s])//3:] = 0
            # pad_mask[s, :atom_loss_masks[0].shape[0]] = atom_loss_masks[0]
            esm_input.append((f"protein_{s}",sequence))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(esm_input)

        # per residue representation
        if token_repr is None:
            with torch.no_grad():
                # self.esm_model.to(torch.float16).cuda()
                token_repr = torch.from_numpy(np.zeros((batch_tokens.shape[0],max_prot_seq_len,1280)))
                for i in range(len(batch_tokens)):
                    results = self.esm_model(batch_tokens[i:i+1,:], repr_layers=[33], return_contacts=True) #.cuda()
                    token_repr[i,:,:] = results["representations"][33][:,1:-1,:]
        elif token_repr.shape[1] * 3 == prot_X.shape[1]:
            counts = torch.tensor([3 for i in range(token_repr.shape[1])])
            token_repr = torch.repeat_interleave(token_repr, counts, dim=1)

        rotations     = repeat(torch.eye(3), '... -> b n ...', b = 1, n = max_prot_seq_len).float()
        translations  = torch.randn(1, max_prot_seq_len, 3).float()

        prot_input = torch.cat([prot_X, token_repr], dim=-1).cuda().float()
        h = self.protein_encoder(prot_input, rotations=rotations.cuda(), translations=translations.cuda(), mask=pad_mask_atom.cuda().bool())
        label = y_batch.squeeze()
        
        self.linear_layers.cuda()
        atom_pred = self.linear_layers(h).squeeze()
        res_pred = atom_pred.view(-1, 3)
        pred = res_pred.sum(dim=1)
        mask = pad_mask_residue.T.squeeze()
        if self.args.classification_type == "binary":
            loss = F.binary_cross_entropy_with_logits(pred, label.cuda(), reduction = 'none')
        elif self.args.classification_type == "multiclass":
            loss = F.cross_entropy(pred, label.cuda().type(torch.int64), reduction = 'none')
        loss = (loss * mask.cuda()).sum() / mask.sum() #masking out padded residues
        return loss, pred, label, atom_pred, mask
