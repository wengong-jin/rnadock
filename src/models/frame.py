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

sys.path.append("/home/dnori/rnadock/src/data")
from data import ProteinDataset, ProteinBinaryDataset, ProteinMulticlassDataset

class ClassificationModel(nn.Module):

    def __init__(self, args):
        super(ClassificationModel, self).__init__()
        self.blm = BinaryLabelMetrics()
        self.protein_encoder = FAEncoder(args, "protein")
        self.ligand_encoder = FAEncoder(args, args.ligand_type)
        if args.ligand_structure:
            insz = 2 * args.encoder_hidden_size
        else:
            insz = args.encoder_hidden_size
        self.linear_layers = nn.Sequential(
                nn.Linear(insz, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(args.mlp_hidden_size, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(args.mlp_hidden_size, args.mlp_output_dim),
        ).cuda().float()
        #torch.cuda.set_device(args.device)
        self.esm_model, self.esm_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        self.esm_model.eval()
        self.args = args

    def forward(self, prot_X, ligand_X, seqs, ligand_seqs, y_batch, max_prot_seq_len, max_ligand_seq_len):
        
        esm_input = []
        pad_mask = torch.from_numpy(np.ones((prot_X.shape[0],max_prot_seq_len)))
        for s in range(len(seqs)):
            sequence = seqs[s] + "<pad>"*(max_prot_seq_len - len(seqs[s]))
            pad_mask[s,len(seqs[s]):] = 0
            esm_input.append((f"protein_{s}",sequence))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(esm_input)

        if self.args.ligand_type=="peptide":
            esm_input_ligand = []
            pad_mask_ligand = torch.from_numpy(np.ones((ligand_X.shape[0],max_ligand_seq_len)))
            for s in range(len(ligand_seqs)):
                sequence = ligand_seqs[s] + "<pad>"*(max_ligand_seq_len - len(ligand_seqs[s]))
                pad_mask_ligand[s,len(ligand_seqs[s]):] = 0
                esm_input_ligand.append((f"ligand_{s}",sequence))
            batch_labels_ligand, batch_strs_ligand, batch_tokens_ligand = self.batch_converter(esm_input_ligand)

        elif self.args.ligand_type=="rna":
            pad_mask_ligand = torch.from_numpy(np.ones((ligand_X.shape[0],max_ligand_seq_len)))
            for s in range(len(ligand_seqs)):
                pad_mask_ligand[s,len(ligand_seqs[s]):] = 0

        # per residue representation
        with torch.no_grad():
            self.esm_model.to(torch.float16).cuda()

            token_repr = torch.from_numpy(np.zeros((batch_tokens.shape[0],max_prot_seq_len,1280)))
            for i in range(len(batch_tokens)):
                results = self.esm_model(batch_tokens[i:i+1,:].cuda(), repr_layers=[33], return_contacts=True) #.cuda()
                token_repr[i,:,:] = results["representations"][33][:,1:-1,:]

            if self.args.ligand_type=="peptide": 
                token_repr_ligand = torch.from_numpy(np.zeros((batch_tokens_ligand.shape[0],max_ligand_seq_len,1280)))
                for i in range(len(batch_tokens_ligand)):
                    results_ligand = self.esm_model(batch_tokens_ligand[i:i+1,:].cuda(), repr_layers=[33], return_contacts=True) #.cuda()
                    token_repr_ligand[i,:,:] = results_ligand["representations"][33][:,1:-1,:]

        self.protein_encoder.cuda()
        h_prot = self.protein_encoder(prot_X.cuda(), h_S=token_repr, mask=pad_mask.cuda()) # h_prot: (batch sz, max prot seq len, encoder hidden dim)
        full_mask = torch.unsqueeze(torch.mm(pad_mask.T, pad_mask_ligand).flatten(), dim=0)

        if self.args.ligand_structure:
            self.ligand_encoder.cuda()
            if self.args.ligand_type=="peptide":
                h_lig = self.ligand_encoder(ligand_X.cuda(), h_S=token_repr_ligand, mask=pad_mask_ligand.cuda()) #h_ligand: (batch sz, max ligand seq len, encoder hidden dim)
            elif self.args.ligand_type == "rna":
                h_lig = self.ligand_encoder(ligand_X.cuda(), h_S=None, mask=pad_mask_ligand.cuda()) #h_ligand: (batch sz, max ligand seq len, encoder hidden dim)

            # Repeat and tile h_prot and h_pep to create tensors of shape (batch_size, max_prot_seq_len, max_ligand_seq_len, encoder_hidden_dim)
            h_prot_tiled = h_prot.unsqueeze(2).repeat(1, 1, max_ligand_seq_len, 1)
            h_lig_tiled = h_lig.unsqueeze(1).repeat(1, max_prot_seq_len, 1, 1)

            # Concatenate h_prot_tiled and h_pep_tiled along the last dimension to create a tensor of shape (batch_size, max_prot_seq_len, max_ligand_seq_len, 2*encoder_hidden_dim)
            h_concat = torch.cat([h_prot_tiled, h_lig_tiled], dim=-1)

            # Reshape h_concat to create the final tensor of shape (batch_size, max_prot_seq_len * max_ligand_seq_len, 2*encoder_hidden_dim)
            h = h_concat.view(len(h_prot), max_prot_seq_len * max_ligand_seq_len, 2*args.encoder_hidden_size)
            mask = full_mask

            if self.args.classification_type == "binary":
                label = y_batch.squeeze().flatten()
            elif self.args.classification_type == "multiclass":
                y_batch = y_batch.squeeze().flatten()
                label = y_batch
        else:
            h = h_prot
            mask = pad_mask.T.squeeze()
            
            if self.args.classification_type == "binary":
                label = torch.any((y_batch.squeeze() == 1.), dim=1).long().float()
            elif self.args.classification_type == "multiclass":
                y_batch = y_batch.squeeze()
                square_mask = full_mask.reshape((max_prot_seq_len, max_ligand_seq_len))
                square_mask = np.where(square_mask == 0, True, False) #masked positions are True
                y_batch[square_mask] = 10000 # masked positions are 10000 (not min)
                label = y_batch.min(axis=1)
                label[np.where(np.all(y_batch == 10000, axis=1))] = 0
                label = torch.from_numpy(label)
        
        self.linear_layers.cuda()
        pred = self.linear_layers(h.float()).squeeze()
        if self.args.classification_type == "binary":
            loss = F.binary_cross_entropy_with_logits(pred, label.cuda(), reduction = 'none')
        elif self.args.classification_type == "multiclass":
            loss = F.cross_entropy(pred, label.cuda().type(torch.int64), reduction = 'none')
        loss = (loss * mask.cuda()).sum() / mask.sum() #masking out padded residues
        return loss, pred, label, mask

class FrameAveraging(nn.Module):

    def __init__(self):
        super(FrameAveraging, self).__init__()
        self.ops = torch.tensor([
                [i,j,k] for i in [-1,1] for j in [-1,1] for k in [-1,1]
        ]).cuda()

    def create_frame(self, X, mask):
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B,N,3]
        C = torch.bmm(X.transpose(1,2), X)  # [B,3,3] (Cov)
        L, V = torch.linalg.eigh(C.detach(), UPLO='U')
        F_ops = self.ops.unsqueeze(1).unsqueeze(0) * V.unsqueeze(1)  # [1,8,1,3] x [B,1,3,3] -> [B,8,3,3]
        h = torch.einsum('boij,bpj->bopi', F_ops.transpose(2,3), X)  # transpose is inverse [B,8,N,3]
        h = h.view(X.size(0) * 8, X.size(1), 3)
        return h, F_ops.detach(), center

    def invert_frame(self, X, mask, F_ops, center):
        X = torch.einsum('boij,bopj->bopi', F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)
        return X * mask.unsqueeze(-1)

class FAEncoder(FrameAveraging):

    def __init__(self, args, mol_type):
        super(FAEncoder, self).__init__()
        if mol_type == "protein":
            self.encoder = SRUpp(
                    args.esm_emb_size + 3,
                    args.encoder_hidden_size // 2,
                    args.encoder_hidden_size // 2,
                    num_layers=args.depth,
                    dropout=args.dropout,
                    bidirectional=True,
            ).cuda().float()
        elif mol_type == "peptide":
            self.encoder = SRUpp(
                    args.esm_emb_size + 3,
                    args.encoder_hidden_size // 2,
                    args.encoder_hidden_size // 2,
                    num_layers=args.depth,
                    dropout=args.dropout,
                    bidirectional=True,
            ).cuda().float()
        elif mol_type == "rna":
            self.encoder = SRUpp(
                    3,
                    args.encoder_hidden_size // 2,
                    args.encoder_hidden_size // 2,
                    num_layers=args.depth,
                    dropout=args.dropout,
                    bidirectional=True,
            ).cuda().float()
        self.mol_type = mol_type

    def forward(self, X, h_S=None, mask=None):
        # X is shape (number in batch, max number of residues in protein, 3)
        B = X.shape[0] # number in batch
        N = X.shape[1] # max number of residues in protein

        h_X, _, _ = self.create_frame(X, mask)
        mask = mask.unsqueeze(1).expand(-1, 8, -1).reshape(B*8, N)

        if h_S is not None:
            h_S = h_S.unsqueeze(1).expand(-1, 8, -1, -1).reshape(B*8, N, -1)
            h = torch.cat([h_X, h_S.cuda()], dim=-1)
        else:
            h = h_X

        h, _, _ = self.encoder(
                h.transpose(0, 1).float().cuda(),
                mask_pad=(~mask.transpose(0, 1).bool().cuda())
        )

        h = h.transpose(0, 1).view(B, 8, N, -1)
        return h.mean(dim=1)  # frame averaging