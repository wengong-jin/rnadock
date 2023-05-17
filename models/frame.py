"""
General script which takes the following options:
- dataset: RNA or ligand
- ligand_structure: True or False
- binary or 20-class
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
        esm_input = []
        pad_mask = torch.from_numpy(np.ones((prot_X.shape[0],max_prot_seq_len)))
        for s in range(len(seqs)):
            sequence = seqs[s] + "<pad>"*(max_prot_seq_len - len(seqs[s]))
            pad_mask[s,len(seqs[s]):] = 0
            esm_input.append((f"protein_{s}",sequence))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(esm_input)

        if args.ligand_type=="peptide":
            esm_input_ligand = []
            pad_mask_ligand = torch.from_numpy(np.ones((ligand_X.shape[0],max_ligand_seq_len)))
            for s in range(len(ligand_seqs)):
                sequence = ligand_seqs[s] + "<pad>"*(max_ligand_seq_len - len(ligand_seqs[s]))
                pad_mask_ligand[s,len(ligand_seqs[s]):] = 0
                esm_input_ligand.append((f"ligand_{s}",sequence))
            batch_labels_ligand, batch_strs_ligand, batch_tokens_ligand = self.batch_converter(esm_input_ligand)

        elif args.ligand_type=="rna":
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

            if args.ligand_type=="peptide": 
                token_repr_ligand = torch.from_numpy(np.zeros((batch_tokens_ligand.shape[0],max_ligand_seq_len,1280)))
                for i in range(len(batch_tokens_ligand)):
                    results_ligand = self.esm_model(batch_tokens_ligand[i:i+1,:].cuda(), repr_layers=[33], return_contacts=True) #.cuda()
                    token_repr_ligand[i,:,:] = results_ligand["representations"][33][:,1:-1,:]

        self.protein_encoder.cuda()
        h_prot = self.protein_encoder(prot_X.cuda(), h_S=token_repr, mask=pad_mask.cuda()) # h_prot: (batch sz, max prot seq len, encoder hidden dim)
        full_mask = torch.unsqueeze(torch.mm(pad_mask.T, pad_mask_ligand).flatten(), dim=0)

        if args.ligand_structure:
            self.ligand_encoder.cuda()
            if args.ligand_type=="peptide":
                h_lig = self.ligand_encoder(ligand_X.cuda(), h_S=token_repr_ligand, mask=pad_mask_ligand.cuda()) #h_ligand: (batch sz, max ligand seq len, encoder hidden dim)
            elif args.ligand_type == "rna":
                h_lig = self.ligand_encoder(ligand_X.cuda(), h_S=None, mask=pad_mask_ligand.cuda()) #h_ligand: (batch sz, max ligand seq len, encoder hidden dim)

            # Repeat and tile h_prot and h_pep to create tensors of shape (batch_size, max_prot_seq_len, max_ligand_seq_len, encoder_hidden_dim)
            h_prot_tiled = h_prot.unsqueeze(2).repeat(1, 1, max_ligand_seq_len, 1)
            h_lig_tiled = h_lig.unsqueeze(1).repeat(1, max_prot_seq_len, 1, 1)

            # Concatenate h_prot_tiled and h_pep_tiled along the last dimension to create a tensor of shape (batch_size, max_prot_seq_len, max_ligand_seq_len, 2*encoder_hidden_dim)
            h_concat = torch.cat([h_prot_tiled, h_lig_tiled], dim=-1)

            # Reshape h_concat to create the final tensor of shape (batch_size, max_prot_seq_len * max_ligand_seq_len, 2*encoder_hidden_dim)
            h = h_concat.view(len(h_prot), max_prot_seq_len * max_ligand_seq_len, 2*args.encoder_hidden_size)
            mask = full_mask

            if args.classification_type == "binary":
                label = torch.from_numpy(y_batch.squeeze().flatten())
            elif args.classification_type == "multiclass":
                y_batch = y_batch.squeeze().flatten()
                label = torch.from_numpy(y_batch)
        else:
            h = h_prot
            mask = pad_mask.T.squeeze()
            
            if args.classification_type == "binary":
                label = torch.any((torch.from_numpy(y_batch).squeeze() == 1.), dim=1).long().float()
            elif args.classification_type == "multiclass":
                y_batch = y_batch.squeeze()
                square_mask = full_mask.reshape((max_prot_seq_len, max_ligand_seq_len))
                square_mask = np.where(square_mask == 0, True, False) #masked positions are True
                y_batch[square_mask] = 10000 # masked positions are 10000 (not min)
                label = y_batch.min(axis=1)
                label[np.where(np.all(y_batch == 10000, axis=1))] = 0
                label = torch.from_numpy(label)
        
        self.linear_layers.cuda()
        pred = self.linear_layers(h.float()).squeeze()
        if args.classification_type == "binary":
            loss = F.binary_cross_entropy_with_logits(pred, label.cuda(), reduction = 'none')
        elif args.classification_type == "multiclass":
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

    if args.classification_type == "multiclass" and args.mlp_output_dim < 2:
        raise ValueError("for multiclass, output dimension must be > 1")
    elif args.classification_type == "binary" and args.mlp_output_dim >= 2:
        raise ValueError("for binary, output dimension must be 1")

    dataset = ProteinStructureDataset(f"dataset_{args.ligand_type}.pickle")
    train_data = dataset.train_data
    test_data = dataset.test_data

    # cut off data for now
    # train_data = train_data[:200]
    # test_data = test_data[:100]
    print('Train/test data:', len(train_data), len(test_data))
    
    prot_len_lst = [len(entry['target_coords']) for entry in train_data+test_data]
    max_prot_coords = max(prot_len_lst)
    ligand_len_lst = [len(entry['ligand_coords']) for entry in train_data+test_data]
    max_ligand_coords = max(ligand_len_lst)
    print('Max protein len:', max_prot_coords)
    print('Max ligand len:', max_ligand_coords)

    prot_coords_train, ligand_coords_train, y_train, prot_seqs_train, ligand_seqs_train = ProteinStructureDataset.prep_dists_for_training(train_data, max_prot_coords, max_ligand_coords)
    prot_coords_test, ligand_coords_test, y_test, prot_seqs_test, ligand_seqs_test = ProteinStructureDataset.prep_dists_for_training(test_data, max_prot_coords, max_ligand_coords)

    if args.classification_type == "multiclass":
        # 0-19 = set of distances in the unit above label, 20 = >=20 angstroms
        y_train = np.floor(y_train)
        y_train = np.where(y_train >= args.mlp_output_dim - 1, float(args.mlp_output_dim - 1), y_train)
        y_test = np.ceil(y_test)
        y_test = np.where(y_test >= args.mlp_output_dim - 1, float(args.mlp_output_dim - 1), y_test)
    elif args.classification_type == "binary":
        # cast to binary problem, exclude padded residues
        y_train = np.where((y_train < 10) & (y_train != 0), 1., 0.)
        y_test = np.where((y_test < 10) & (y_test != 0), 1., 0.)
    else:
        raise ValueError("classification_type must by multiclass or binary")

    model = ClassificationModel(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    sequences = []
    predictions = []
    ground_truth = []
    
    for e in range(args.epochs):
        model.train()
        random.shuffle(train_data)
        true_vals = []
        pred_vals = []
        for i in range(0, len(train_data), args.batch_size):
            print(f'batch_{i}_epoch_{e}')
            optimizer.zero_grad()
            try:
                prot_X_batch = prot_coords_train[i : i + args.batch_size]
                ligand_X_batch = ligand_coords_train[i : i + args.batch_size]
                prot_seq_batch = prot_seqs_train[i : i+args.batch_size]
                ligand_seq_batch = ligand_seqs_train[i : i+args.batch_size]
                y_batch = y_train[i : i + args.batch_size]
            except:
                prot_X_batch = prot_coords_train[i :]
                ligand_X_batch = ligand_coords_train[i :]
                prot_seq_batch = prot_seqs_train[i :]
                ligand_seq_batch = ligand_seqs_train[i :]
                y_batch = y_train[i :]

            loss, y_hat, y_batch, mask = model(prot_X_batch, ligand_X_batch, prot_seq_batch, ligand_seq_batch, y_batch, max_prot_coords, max_ligand_coords)

            if args.classification_type == "binary":
                y_hat = torch.sigmoid(y_hat)
            elif args.classification_type == "multiclass":
                y_hat_mm = F.softmax(y_hat, dim=1)
                y_hat = torch.sum(y_hat_mm.cpu() * torch.arange(args.mlp_output_dim), dim=1) #weighted average

            if args.ligand_structure:
                square = (max_prot_coords, max_ligand_coords)
                protein_mask = np.invert(np.all((mask.T.squeeze().reshape(square) == 0).cpu().detach().numpy(), axis=1)) # if all zero, set to False (protein residue nonexistent)

                masked_y_hat = np.ma.masked_where(~mask.T.squeeze().reshape(square).cpu().detach().numpy().astype(bool), 
                                                    y_hat.reshape(square).cpu().detach().numpy())
                y_hat = masked_y_hat.mean(axis=1) # for y_hat, just take average probability value
                y_hat = y_hat[y_hat.mask == False]

                if args.classification_type == "multiclass":
                    masked_y_batch = np.ma.masked_where(~mask.T.squeeze().reshape(square).cpu().detach().numpy().astype(bool), 
                                                    y_batch.reshape(square).cpu().detach().numpy())
                    y_batch = masked_y_batch.min(axis=1)
                    y_batch = y_batch[y_batch.mask == False].T.squeeze()
                    y_hat = np.round(y_hat).T.squeeze()
                    
                    expanded_mask = np.repeat(np.expand_dims(~mask.T.squeeze().
                                    reshape(square).cpu().detach().numpy().astype(bool), 
                                    axis=2), args.mlp_output_dim, axis=2)
                    y_hat_mm = np.ma.masked_where(expanded_mask, 
                                    y_hat_mm.reshape((square[0],square[1],
                                    args.mlp_output_dim)).cpu().detach().numpy())
                    y_hat_mm = y_hat_mm.mean(axis=1)
                    y_hat_mm = y_hat_mm[protein_mask]

                elif args.classification_type == "binary":
                    y_batch = np.any((y_batch.reshape(square) == 1.).cpu().detach().numpy(), axis=1).astype(int)
                    y_batch = y_batch[protein_mask]

            else:
                y_batch = y_batch[mask.bool()].cpu().detach().numpy()
                if args.classification_type == "multiclass":
                    y_hat = np.round(y_hat[mask.bool()].cpu().detach().numpy())
                    y_hat_mm = y_hat_mm[mask.bool()].cpu().detach().numpy()
                elif args.classification_type == "binary":
                    y_hat = y_hat[mask.bool()].cpu().detach().numpy()

            if args.classification_type == "binary":
                true_vals.extend(y_batch.tolist())
                pred_vals.extend(y_hat.tolist())
                predictions.append(y_hat.tolist())
                ground_truth.append(y_batch.tolist())
                sequences.append(prot_seq_batch[0])
                try:
                    print(roc_auc_score(y_batch,y_hat))
                except:
                    print('one class present')
            else:
                true_vals.extend(y_batch.tolist())
                pred_vals.append(y_hat_mm)
                print(f1_score(y_batch,y_hat, average='weighted'))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

        if args.classification_type == "binary":

            visualization_df = pd.DataFrame(
            {'sequences': sequences,
                'predictions': predictions,
                'ground_truth': ground_truth
            })
            visualization_df.to_csv("charts_rna_binary_noligand_fullrun/visualization_train_info.csv",index=False)

            # scores_df = pd.DataFrame({'label':true_vals,'score':pred_vals})
            # model.blm.add_model(f'epoch_{e}', scores_df)
            # model.blm.plot_roc(model_names=[f'epoch_{e}'],params={"save":True,"prefix":f"charts/epoch_{e}_"})
        else:
            names = [f"{i} Angstroms" for i in range(args.mlp_output_dim)]
            true_vals = np.array(true_vals)
            pred_vals = np.concatenate(pred_vals, axis=0)
            blm = ovr(true_vals, pred_vals, names)
            blm.plot_roc(chart_types=[1,2], params={"legloc":4, "addsz":False, "save":True, "prefix":f"charts/epoch_{e}_"})

        filename = f'model_checkpoints/rna_full_binary_noligand_epoch_{e}.pt'
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, filename)

    checkpoint = torch.load("model_checkpoints/rna_full_binary_noligand_epoch_9.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to("cpu")
 
    # hold out test set
    true_vals_test = []
    pred_vals_test = []

    # for visualizations
    sequences = []
    predictions = []
    ground_truth = []


    for i in range(0, len(test_data), args.batch_size):
        prot_X_batch = prot_coords_test[i : i + args.batch_size]
        ligand_X_batch = ligand_coords_test[i : i + args.batch_size]
        prot_seq_batch = prot_seqs_test[i : i+args.batch_size]
        ligand_seq_batch = ligand_seqs_test[i : i+args.batch_size]
        y_batch = y_test[i : i + args.batch_size]

        loss, y_hat, y_batch, mask = model(prot_X_batch, ligand_X_batch, prot_seq_batch, ligand_seq_batch, y_batch, max_prot_coords, max_ligand_coords)
        if args.classification_type == "binary":
            y_hat = torch.sigmoid(y_hat)
        elif args.classification_type == "multiclass":
            y_hat_mm = F.softmax(y_hat, dim=1)
            y_hat = torch.sum(y_hat_mm.cpu() * torch.arange(args.mlp_output_dim), dim=1) #weighted average

        if args.ligand_structure:
            square = (max_prot_coords, max_ligand_coords)
            protein_mask = np.invert(np.all((mask.T.squeeze().reshape(square) == 0).cpu().detach().numpy(), axis=1)) # if all zero, set to False (protein residue nonexistent)

            masked_y_hat = np.ma.masked_where(~mask.T.squeeze().reshape(square).cpu().detach().numpy().astype(bool), 
                                                y_hat.reshape(square).cpu().detach().numpy())
            y_hat = masked_y_hat.mean(axis=1) # for y_hat, just take average probability value
            y_hat = y_hat[y_hat.mask == False]

            if args.classification_type == "multiclass":
                masked_y_batch = np.ma.masked_where(~mask.T.squeeze().reshape(square).cpu().detach().numpy().astype(bool), 
                                                y_batch.reshape(square).cpu().detach().numpy())
                y_batch = masked_y_batch.min(axis=1)
                y_batch = y_batch[y_batch.mask == False].T.squeeze()
                y_hat = np.round(y_hat).T.squeeze()
                
                expanded_mask = np.repeat(np.expand_dims(~mask.T.squeeze().
                                reshape(square).cpu().detach().numpy().astype(bool), 
                                axis=2), args.mlp_output_dim, axis=2)
                y_hat_mm = np.ma.masked_where(expanded_mask, 
                                y_hat_mm.reshape((square[0],square[1],
                                args.mlp_output_dim)).cpu().detach().numpy())
                y_hat_mm = y_hat_mm.mean(axis=1)
                y_hat_mm = y_hat_mm[protein_mask]

            elif args.classification_type == "binary":
                y_batch = np.any((y_batch.reshape(square) == 1.).cpu().detach().numpy(), axis=1).astype(int)
                y_batch = y_batch[protein_mask]

        else:
            y_batch = y_batch[mask.bool()].cpu().detach().numpy()
            if args.classification_type == "multiclass":
                y_hat = np.round(y_hat[mask.bool()].cpu().detach().numpy())
                y_hat_mm = y_hat_mm[mask.bool()].cpu().detach().numpy()
            elif args.classification_type == "binary":
                y_hat = y_hat[mask.bool()].cpu().detach().numpy()

        if args.classification_type == "binary":
            true_vals_test.extend(y_batch.tolist())
            pred_vals_test.extend(y_hat.tolist())
            predictions.append(y_hat.tolist())
            ground_truth.append(y_batch.tolist())
            sequences.append(prot_seq_batch[0])
            try:
                print(roc_auc_score(y_batch,y_hat))
            except:
                print('one class present')
        else:
            true_vals_test.extend(y_batch.tolist())
            pred_vals_test.append(y_hat_mm)
            print(f1_score(y_batch,y_hat, average='weighted'))


    if args.classification_type == "binary":
        # visualization_df = pd.DataFrame(
        #     {'sequences': sequences,
        #     'predictions': predictions,
        #     'ground_truth': ground_truth
        # })
        # visualization_df.to_csv("charts_rna_binary_noligand_fullrun/visualization_info.csv",index=False)

        scores_df = pd.DataFrame({'label':true_vals_test,'score':pred_vals_test})
        model.blm.add_model(f'val', scores_df)
        model.blm.plot_roc(model_names=['val'],params={"save":True,"prefix":f"charts_rna_binary_noligand_fullrun/val_"})
        model.blm.plot(model_names=['val'],chart_types=[1,2,3,4,5],params={"save":True,"prefix":f"charts_rna_binary_noligand_fullrun/val_"})
    else:
        names = [f"{i} Angstroms" for i in range(args.mlp_output_dim)]
        true_vals_test = np.array(true_vals_test)
        pred_vals_test = np.concatenate(pred_vals_test, axis=0)
        blm = ovr(true_vals_test, pred_vals_test, names)
        blm.plot_roc(chart_types=[1,2], params={"legloc":4, "addsz":False, "save":True, "prefix":f"charts/val_"})
        