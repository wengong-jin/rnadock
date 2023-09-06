"""
Runs test on RNA-binding proteins and other general proteins that don't bind to RNA
"""

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
from binary_label_metrics import BinaryLabelMetrics

sys.path.append("/home/dnori/rnadock/data")
from data import ProteinStructureDataset

#### metrics helper functions for multiclass

def ovr(label, pred, names):
  bcm = BinaryLabelMetrics(); auc = list()
  print(pred.shape)
  print(label.shape)
  for n,name in enumerate(names):
    df = pd.DataFrame({"label":np.where(label==n,1,0),"score":pred[:,n]})
    if sum(df["label"].tolist()) > 0:
        bcm.add_model(name, df)
        auc.append(roc_auc_score(df["label"],df["score"]))
  return bcm

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

    def forward(self, prot_X, ligand_X, seqs, ligand_seqs, y_batch, max_prot_seq_len, max_ligand_seq_len):

        # ligand_X, ligand_seqs are None

        esm_input = []
        pad_mask = torch.from_numpy(np.ones((prot_X.shape[0],max_prot_seq_len)))
        for s in range(len(seqs)):
            sequence = seqs[s] + "<pad>"*(max_prot_seq_len - len(seqs[s]))
            pad_mask[s,len(seqs[s]):] = 0
            esm_input.append((f"protein_{s}",sequence))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(esm_input)

        pad_mask_ligand = torch.from_numpy(np.ones((prot_X.shape[0],max_ligand_seq_len)))
        # not zeroing out based on ligand length (predict till max len)

        # per residue representation
        with torch.no_grad():
            self.esm_model.to(torch.float16).cuda()
            token_repr = torch.from_numpy(np.zeros((batch_tokens.shape[0],max_prot_seq_len,1280)))
            for i in range(len(batch_tokens)):
                results = self.esm_model(batch_tokens[i:i+1,:].cuda(), repr_layers=[33], return_contacts=True) #.cuda()
                token_repr[i,:,:] = results["representations"][33][:,1:-1,:]

        self.protein_encoder.cuda()
        h_prot = self.protein_encoder(prot_X.cuda(), h_S=token_repr, mask=pad_mask.cuda()) # h_prot: (batch sz, max prot seq len, encoder hidden dim)
        full_mask = torch.unsqueeze(torch.mm(pad_mask.T, pad_mask_ligand).flatten(), dim=0)
    
        h = h_prot
        mask = pad_mask.T.squeeze()
        label = torch.any((torch.from_numpy(y_batch).squeeze() == 1.), dim=1).long().float()
        
        self.linear_layers.cuda()
        pred = self.linear_layers(h.float()).squeeze()
        loss = F.binary_cross_entropy_with_logits(pred, label.cuda(), reduction = 'none')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--esm_emb_size', type=int, default=1280)
    parser.add_argument('--encoder_hidden_size', type=int, default=200)
    parser.add_argument('--mlp_hidden_size', type=int, default=200)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--mlp_output_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--anneal_rate', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ligand_structure', type=bool, default=False)
    parser.add_argument('--ligand_type',type=str,default='rna')
    parser.add_argument('--classification_type',type=str,default='binary')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.mlp_output_dim >= 2:
        raise ValueError("for binary, output dimension must be 1")

    rbp_dataset = ProteinStructureDataset(f"dataset_rna.pickle")
    rbp_train_data = rbp_dataset.train_data
    rbp_test_data = rbp_dataset.test_data

    non_rbp_dataset = ProteinStructureDataset(f"dataset_non_rbp.pickle")
    non_rbp_train_data = non_rbp_dataset.train_data
    non_rbp_test_data = non_rbp_dataset.test_data

    # combine the train and test datasets between rbp and non-rbp
    train_data, rbp_identity_train = ProteinStructureDataset.combine_sets(rbp_train_data, non_rbp_train_data)
    test_data, rbp_identity_test = ProteinStructureDataset.combine_sets(rbp_test_data, non_rbp_test_data)

    print(len(rbp_test_data))
    print(len(non_rbp_test_data))

    print('Train/test data:', len(train_data), len(test_data))
    
    prot_len_lst = [len(entry['target_coords']) for entry in train_data+test_data]
    max_prot_coords = max(prot_len_lst)
    ligand_len_lst = [len(entry['ligand_coords']) for entry in rbp_train_data+rbp_test_data]
    max_ligand_coords = max(ligand_len_lst)
    print('Max protein len:', max_prot_coords)

    prot_coords_test, prot_seqs_test, y_test = ProteinStructureDataset.prep_rbp_mix(test_data, max_prot_coords, max_ligand_coords)

    # cast to binary problem, exclude padded residues
    y_test = np.where((y_test < 10) & (y_test != 0), 1., 0.)

    model = ClassificationModel(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    checkpoint = torch.load("model_checkpoints/rna_full_binary_noligand_epoch_9.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to("cpu")
 
    # hold out test set
    true_vals_test = []
    pred_vals_test = []

    # for visualizations
    sequences = []
    site_predictions = []
    site_ground_truth = []
    number_predicted_sites_norm = []

    for i in range(0, len(test_data), args.batch_size):
        prot_X_batch = prot_coords_test[i : i + args.batch_size]
        prot_seq_batch = prot_seqs_test[i : i+args.batch_size]
        y_batch = y_test[i : i + args.batch_size]

        # loss, y_hat, y_batch, mask = model(prot_X_batch, ligand_X_batch, prot_seq_batch, ligand_seq_batch, y_batch, max_prot_coords, max_ligand_coords)
        loss, y_hat, y_batch, mask = model(prot_X_batch, None, prot_seq_batch, None, y_batch, max_prot_coords, max_ligand_coords)
        y_hat = torch.sigmoid(y_hat)
        y_hat = y_hat[mask.bool()].cpu().detach().numpy()
        y_batch = y_batch[mask.bool()].cpu().detach().numpy()

        true_vals_test.extend(y_batch.tolist())
        pred_vals_test.extend(y_hat.tolist())
        site_predictions.append(y_hat.tolist())
        site_ground_truth.append(y_batch.tolist())
        sequences.append(prot_seq_batch[0])
        number_predicted_sites_norm.append(np.sum(np.where(y_hat > 0.19, 1, 0)) / y_hat.shape[0])

        print(rbp_identity_test[i])
        print(np.sum(np.where(y_hat > 0.19, 1, 0)) / y_hat.shape[0])

    # based on average score, identify as RBP or not
    # plot AUC of that
    scores_df = pd.DataFrame({'label':rbp_identity_test,'score':number_predicted_sites_norm})
    model.blm.add_model(f'val', scores_df)
    model.blm.plot_roc(model_names=['val'],params={"save":True,"prefix":f"charts_rbp_identification_binary_noligand/val_"})
    model.blm.plot(model_names=['val'],chart_types=[1,2,3,4,5],params={"save":True,"prefix":f"charts_rbp_identification_binary_noligand/val_"})
        