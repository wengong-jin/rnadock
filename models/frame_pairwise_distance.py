import argparse
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sru import SRUpp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

sys.path.append("/home/dnori/rnadock/data")
from data import ProteinStructureDataset

class PairwiseDistanceModel(nn.Module):

    def __init__(self, args):
        super(PairwiseDistanceModel, self).__init__()
        self.protein_encoder = FAEncoder(args, "protein")
        self.rna_encoder = FAEncoder(args, "rna")
        self.linear_layers = nn.Sequential(
                nn.Linear(2 * args.encoder_hidden_size, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(args.mlp_hidden_size, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(args.mlp_hidden_size, args.mlp_output_dim),
        ).cuda().float()
        torch.cuda.set_device(args.device)

        self.esm_model, self.esm_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        self.esm_model.eval()

    def forward(self, prot_X, rna_X, seqs, rna_seqs, label, max_prot_seq_len, max_rna_seq_len):

        # prot_X: (batch sz, max prot seq len, 3)
        # rna_X: (batch sz, max rna seq len, 3)
        # seqs: batch sz-len list
        # label: (batch sz, max prot seq len, max rna seq len)
        # max_prot_seq_len: max length of protein sequence
        # max_rna_seq_len: max length of rna sequence

        esm_input = []
        pad_mask = torch.from_numpy(np.ones((prot_X.shape[0],max_prot_seq_len)))
        for s in range(len(seqs)):
            sequence = seqs[s] + "<pad>"*(max_prot_seq_len - len(seqs[s]))
            pad_mask[s,len(seqs[s]):] = 0
            esm_input.append((f"protein_{s}",sequence))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(esm_input)

        rna_pad_mask = torch.from_numpy(np.ones((rna_X.shape[0],max_rna_seq_len)))
        for s in range(len(rna_seqs)):
            rna_pad_mask[s,len(rna_seqs[s]):] = 0

        # per residue representation
        with torch.no_grad():
            self.esm_model.to(torch.float16).cuda()

            token_repr = torch.from_numpy(np.zeros((batch_tokens.shape[0],max_prot_seq_len,1280)))
            for i in range(len(batch_tokens)):
                results = self.esm_model(batch_tokens[i:i+1,:].cuda(), repr_layers=[33], return_contacts=True)
                token_repr[i,:,:] = results["representations"][33][:,1:-1,:]

        h_prot = self.protein_encoder(prot_X.cuda(), h_S=token_repr, mask=pad_mask.cuda()) # h_prot: (batch sz, max prot seq len, encoder hidden dim)
        h_rna = self.rna_encoder(rna_X.cuda(), mask=rna_pad_mask.cuda()) #h_rna: (batch sz, max rna seq len, encoder hidden dim)

        # flatten the encoded sequences and concatenate them
        h_prot = h_prot.reshape(h_prot.shape[0], h_prot.shape[1], 1, h_prot.shape[2]).detach().cpu().numpy()  # shape: (batch_sz, max_prot_seq_len, 1, encoder_hidden_dim)
        h_rna = h_rna.reshape(h_rna.shape[0], 1, h_rna.shape[1], h_rna.shape[2]).detach().cpu().numpy()  # shape: (batch_sz, 1, max_rna_seq_len, encoder_hidden_dim)
        h = torch.from_numpy(np.concatenate((h_prot.repeat(h_rna.shape[2], axis=2), h_rna.repeat(h_prot.shape[1], axis=1)), axis=-1)).cuda().view(1, -1, 2*args.encoder_hidden_size)  # shape: (batch_sz, max_prot_seq_len * max_rna_seq_len, 2*encoder_hidden_dim)
        print(torch.max(h))
        print(torch.min(h))

        label = label.view(1, -1)
        mask = torch.unsqueeze(torch.mm(pad_mask.T, rna_pad_mask).flatten(), dim=0)
        
        # h = torch.cat((h,token_repr.cuda()),dim=-1)
        pred = self.linear_layers(h.float()).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(pred, label.cuda(), reduction = 'none')
        loss = (loss * mask.cuda()).sum() / mask.sum() #masking out padded residues
        return loss, pred, label

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
        # h_S is ESM embedding
        B = X.shape[0] # number in batch
        N = X.shape[1] # max number of residues in protein

        h, _, _ = self.create_frame(X, mask)
        
        mask = mask.unsqueeze(1).expand(-1, 8, -1).reshape(B*8, N)

        if self.mol_type == "protein":
            h_S = h_S.unsqueeze(1).expand(-1, 8, -1, -1).reshape(B*8, N, -1)
            h = torch.cat([h, h_S.cuda()], dim=-1)

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
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset = ProteinStructureDataset("dataset.pickle")
    train_data = dataset.train_data
    test_data = dataset.test_data
    print('Train/test data:', len(train_data), len(test_data))

    train_data = train_data[:10]
    test_data = test_data[:10]
    
    max_prot_coords = max([len(entry['target_coords']) for entry in train_data+test_data])
    max_rna_coords = max([len(entry['rna_coords']) for entry in train_data+test_data])
    print('Max protein len:', max_prot_coords)
    print('Max RNA len:', max_rna_coords)

    # prot coords: (num data points, max prot seq len, 3)
    # rna coords: (num data points, max rna seq len, 3)
    # y: (num data points, max prot seq len, max rna seq len)
    prot_coords_train, rna_coords_train, y_train, prot_seqs_train, rna_seqs_train = ProteinStructureDataset.prep_dists_for_training(train_data, max_prot_coords, max_rna_coords)
    prot_coords_test, rna_coords_test, y_test, prot_seqs_test, rna_seqs_test = ProteinStructureDataset.prep_dists_for_training(test_data, max_prot_coords, max_rna_coords)

    model = PairwiseDistanceModel(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    r2_dict = {}

    for e in range(args.epochs):
        model.train()
        random.shuffle(train_data)
        train_r2 = []
        for i in range(0, len(train_data), args.batch_size):
            print(f'batch_{i}_epoch_{e}')
            optimizer.zero_grad()
            try:
                prot_X_batch = prot_coords_train[i : i + args.batch_size]
                rna_X_batch = rna_coords_train[i : i + args.batch_size]
                prot_seq_batch = prot_seqs_train[i : i+args.batch_size]
                rna_seq_batch = rna_seqs_train[i : i+args.batch_size]
                y_batch = y_train[i : i + args.batch_size]
            except:
                prot_X_batch = prot_coords_train[i :]
                rna_X_batch = rna_coords_train[i :]
                prot_seq_batch = prot_seqs_train[i :]
                rna_seq_batch = rna_seqs_train[i :]
                y_batch = y_train[i :]

            loss, y_hat, y_batch = model(prot_X_batch, rna_X_batch, prot_seq_batch, rna_seq_batch, y_batch, max_prot_coords, max_rna_coords)
            y_batch = y_batch.reshape((y_batch.shape[0]*y_batch.shape[1],))
            y_hat = y_hat.reshape((y_hat.shape[0]*y_hat.shape[1],))

            r2 = r2_score(y_batch.cpu().detach().numpy(),y_hat.cpu().detach().numpy())
            print(f"Train R^2: {r2}")
            train_r2.append(r2)

            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

        r2_dict[f"Train epoch {e}"] = sum(train_r2) / len(train_r2)

    # hold out test set
    test_r2 = []
    for i in range(0, len(test_data), args.batch_size):
        prot_X_batch = prot_coords_test[i : i + args.batch_size]
        rna_X_batch = rna_coords_test[i : i + args.batch_size]
        prot_seq_batch = prot_seqs_test[i : i+args.batch_size]
        rna_seq_batch = rna_seqs_test[i : i+args.batch_size]
        y_batch = y_test[i : i + args.batch_size]

        loss, y_hat, y_batch = model(prot_X_batch, rna_X_batch, prot_seq_batch, rna_seq_batch, y_batch, max_prot_coords, max_rna_coords)

        y_batch = y_batch.reshape((y_batch.shape[0]*y_batch.shape[1],))
        y_hat = y_hat.reshape((y_hat.shape[0]*y_hat.shape[1],))

        r2 = r2_score(y_batch.cpu().detach().numpy(),y_hat.cpu().detach().numpy())
        print(f"Test R^2: {r2}")
        test_r2.append(r2)

    r2_dict["Test"] = sum(test_r2) / len(test_r2)
    print(r2_dict)

    # r2_df = pd.DataFrame(r2_dict)
    # r2_df.to_csv('charts/r2_metric.csv', index=False, header=True)