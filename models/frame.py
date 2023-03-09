import argparse
import numpy as np
import random
from sklearn.metrics import r2_score
from sru import SRUpp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

sys.path.append("/home/dnori/rnadock/data")
from data import ProteinStructureDataset

class RegressionModel(nn.Module):

    def __init__(self, args):
        super(RegressionModel, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.encoder = FAEncoder(args)
        self.linear_layers = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.output_dim),
        )

    def mean(self, X, mask):
        return (X * mask[...,None]).sum(dim=1) / mask[...,None].sum(dim=1).clamp(min=1e-6)

    def forward(self, X, mask, label):
        h = self.encoder(X, mask)
        h = self.mean(h, mask)
        pred = self.linear_layers(h).squeeze(-1)
        loss = self.mse_loss(pred, label)
        return loss, pred


class FrameAveraging(nn.Module):

    def __init__(self):
        super(FrameAveraging, self).__init__()
        self.ops = torch.tensor([
                [i,j,k] for i in [-1,1] for j in [-1,1] for k in [-1,1]
        ])

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
    parser.add_argument('--hidden_size', type=int, default=5000)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--output_dim', type=int, default=2496)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--anneal_rate', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset = ProteinStructureDataset("dataset.pickle")
    tgt_X, tgt_mask, y = ProteinStructureDataset.prep_for_training(dataset)

    # replace with a better split
    train_data = dataset[:int(.8 * len(dataset))]
    val_data = dataset[int(.8 * len(dataset)):]
    print('Training/Val data:', len(train_data), len(val_data))

    # getting rid of some bad data points - fix later
    train_data = train_data[:20]

    model = RegressionModel(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    for e in range(args.epochs):
        model.train()
        random.shuffle(train_data)
        for i in range(0, len(train_data), args.batch_size):
            optimizer.zero_grad()
            tgt_X_batch = tgt_X[i : i + args.batch_size]
            tgt_mask_batch = tgt_mask[i : i + args.batch_size]
            y_batch = y[i : i + args.batch_size]
            loss, pred = model(tgt_X_batch, tgt_mask_batch, y_batch)
            print(f"Batch {i} R^2.")
            pred = pred.detach().numpy()
            pred = np.where(pred<0, 0, 1)
            print(r2_score(y_batch.detach().numpy().astype(int), pred))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

    val_X = tgt_X[int(.8 * len(tgt_X)):]
    val_mask = tgt_mask[int(.8 * len(tgt_X)):]
    val_y = tgt_y[int(.8 * len(tgt_X)):]
    _, pred = model(val_X, val_mask, val_y)
    print("Validation R^2.")
    pred = pred.detach().numpy()
    pred = np.where(pred<0, 0, 1)
    print(r2_score(val_y.detach().numpy(), pred))