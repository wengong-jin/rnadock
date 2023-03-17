import argparse
import numpy as np
import pandas as pd
import random
from sklearn.metrics import r2_score
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

class RegressionModel(nn.Module):

    def __init__(self, args):
        super(RegressionModel, self).__init__()
        self.blm = BinaryLabelMetrics()
        self.encoder = FAEncoder(args)
        self.linear_layers = nn.Sequential(
                nn.Linear(args.input_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.output_dim),
        )
        #torch.cuda.set_device(args.device)

        self.esm_model, self.esm_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        self.esm_model.eval()

    def mean(self, X, mask):
        return (X * mask[...,None]).sum(dim=1) / mask[...,None].sum(dim=1).clamp(min=1e-6)

    def forward(self, X, mask, seqs, label):

        # pad sequences till length 500, convert to batch
        esm_input = []
        for s in range(len(seqs)):
            sequence = seqs[s] + "<pad>"*(500 - len(seqs[s]))
            esm_input.append((f"protein_{s}",sequence))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(esm_input)

        # per residue representation (CPU)
        with torch.no_grad():
            self.esm_model.cpu()
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_repr = results["representations"][33][:,1:-1,:]

        sequence_representations = []
        for i, (_, seq) in enumerate(esm_input):
            sequence_representations.append(token_repr[i, 1 : len(seq) + 1].mean(0))
        seq_repr = torch.stack(sequence_representations)

        h = self.encoder(X, mask)
        h = self.mean(h, mask)
        h = torch.cat((h, seq_repr.cuda()),dim=1)
        print(h.shape)
        
        pred = self.linear_layers(h).squeeze(-1).clamp(0,1)
        loss = F.binary_cross_entropy(pred, label)
        return loss, pred

class FrameAveraging(nn.Module):

    def __init__(self):
        super(FrameAveraging, self).__init__()
        torch.cuda.set_device("cuda:3")
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

    def __init__(self, args):
        super(FAEncoder, self).__init__()
        self.encoder = SRUpp(
                3,
                args.hidden_size // 2,
                args.hidden_size // 2,
                num_layers=args.depth,
                dropout=args.dropout,
                bidirectional=True,
        )

    def forward(self, X, mask):
        # X is shape (number in batch, max number of residues in protein, 3)
        B = X.shape[0] # number in batch
        N = X.shape[1] # max number of residues in protein

        # X shape - (B, N, 3)
        # token_repr shape - (B, N, esm emb sz 1280)

        h, _, _ = self.create_frame(X, mask)
        mask = mask.unsqueeze(1).expand(-1, 8, -1).reshape(B*8, N)

        h, _, _ = self.encoder(
                h.transpose(0, 1),
                mask_pad=(~mask.transpose(0, 1).bool())
        )

        h = h.transpose(0, 1).view(B, 8, N, -1)
        return h.mean(dim=1)  # frame averaging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=4280)
    parser.add_argument('--hidden_size', type=int, default=3000)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--anneal_rate', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:3')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # replace with a better split
    dataset = ProteinStructureDataset("dataset.pickle")
    train_data = dataset.train_data
    test_data = dataset.test_data
    print('Train/test data:', len(train_data), len(test_data))
    
    coords_train, mask_train, seqs_train, y_train = ProteinStructureDataset.prep_for_training(train_data)
    coords_test, mask_test, seqs_test, y_test = ProteinStructureDataset.prep_for_training(test_data)

    model = RegressionModel(args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    for e in range(args.epochs):
        model.train()
        random.shuffle(train_data)
        true_vals = []
        pred_vals = []
        for i in range(0, len(train_data), args.batch_size):
            print(f'batch_{i}_epoch_{e}')
            optimizer.zero_grad()
            try:
                tgt_X_batch = coords_train[i : i + args.batch_size].cuda()
                tgt_mask_batch = mask_train[i : i + args.batch_size].cuda()
                tgt_seq_batch = seqs_train[i : i+args.batch_size]
                y_batch = y_train[i : i + args.batch_size].cuda()
            except:
                tgt_X_batch = coords_train[i :].cuda()
                tgt_mask_batch = mask_train[i :].cuda()
                tgt_seq_batch = seqs_train[i :]
                y_batch = y_train[i :].cuda()

            loss, y_hat = model(tgt_X_batch, tgt_mask_batch, tgt_seq_batch, y_batch)
            
            y_batch = y_batch.reshape((y_batch.shape[0]*y_batch.shape[1],))
            y_hat = y_hat.reshape((y_hat.shape[0]*y_hat.shape[1],))

            true_vals.extend(y_batch.cpu().detach().numpy().tolist())
            pred_vals.extend(y_hat.cpu().detach().numpy().tolist())
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

        scores_df = pd.DataFrame({'label':true_vals,'score':pred_vals})
        model.blm.add_model(f'epoch_{e}', scores_df)
        model.blm.plot_roc(model_names=[f'epoch_{e}'],params={"save":True,"prefix":f"charts/epoch_{e}_"})

    # this gets CUDA out of memory issue - do it in batches
    # _, pred = model(coords_test.cuda(), mask_test.cuda(), y_test.cuda())
    # scores_df = pd.DataFrame({'label':y_test.cpu().detach().numpy().tolist(),'score':pred.cpu().detach().numpy().tolist()})
    # model.blm.add_model('val', scores_df)
    # model.blm.plot_roc(model_names=['val'],params={"save":True,"prefix":"charts/val_"})

    #do hyperparameter sweep for MLP

    #add protein amino acid sequence + ESM2 features
    #model input (B, N, esm emb size, 3)