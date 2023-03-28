import argparse
import gzip
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
from binary_label_metrics import BinaryLabelMetrics

sys.path.append("/home/dnori/rnadock/data")
from data import ProteinStructureDataset

class ClassificationModel(nn.Module):

    def __init__(self, args):
        super(ClassificationModel, self).__init__()
        self.blm = BinaryLabelMetrics()
        self.encoder = FAEncoder(args)
        self.linear_layers = nn.Sequential(
                nn.Linear(args.esm_emb_size + args.encoder_hidden_size, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(args.mlp_hidden_size, args.mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(args.mlp_hidden_size, args.mlp_output_dim),
        ).cuda().float()
        torch.cuda.set_device(args.device)

        self.esm_model, self.esm_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        self.esm_model.eval()

    def forward(self, X, seqs, label, max_seq_len):

        esm_input = []
        pad_mask = torch.from_numpy(np.ones((X.shape[0],max_seq_len)))
        for s in range(len(seqs)):
            sequence = seqs[s] + "<pad>"*(max_seq_len - len(seqs[s]))
            pad_mask[s,len(seqs[s]):] = 0
            esm_input.append((f"protein_{s}",sequence))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(esm_input)

        # per residue representation
        with torch.no_grad():
            self.esm_model.to(torch.float16).cuda()

            token_repr = torch.from_numpy(np.zeros((batch_tokens.shape[0],max_seq_len,1280)))
            for i in range(len(batch_tokens)):
                results = self.esm_model(batch_tokens[i:i+1,:].cuda(), repr_layers=[33], return_contacts=True)
                token_repr[i,:,:] = results["representations"][33][:,1:-1,:]

        h = self.encoder(X.cuda(), token_repr, pad_mask.cuda())
        h = torch.cat((h,token_repr.cuda()),dim=-1)
        pred = self.linear_layers(h.float()).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(pred, label.cuda(), reduction = 'none')
        loss = (loss * pad_mask.cuda()).sum() / pad_mask.sum() #masking out padded residues
        return loss, pred

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

    def __init__(self, args):
        super(FAEncoder, self).__init__()
        self.encoder = SRUpp(
                args.esm_emb_size + 3,
                args.encoder_hidden_size // 2,
                args.encoder_hidden_size // 2,
                num_layers=args.depth,
                dropout=args.dropout,
                bidirectional=True,
        ).cuda().float()

    def forward(self, X, h_S, mask):
        # X is shape (number in batch, max number of residues in protein, 3)
        B = X.shape[0] # number in batch
        N = X.shape[1] # max number of residues in protein

        h_X, _, _ = self.create_frame(X, mask)
        h_S = h_S.unsqueeze(1).expand(-1, 8, -1, -1).reshape(B*8, N, -1)
        mask = mask.unsqueeze(1).expand(-1, 8, -1).reshape(B*8, N)

        h = torch.cat([h_X, h_S.cuda()], dim=-1)
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

    # replace with a better split
    dataset = ProteinStructureDataset("dataset.pickle")
    train_data = dataset.train_data
    test_data = dataset.test_data
    print('Train/test data:', len(train_data), len(test_data))
    
    max_seq_len = max([len(entry['binary_mask']) for entry in train_data+test_data])
    print('Max seq len:', max_seq_len)

    coords_train, seqs_train, y_train = ProteinStructureDataset.prep_for_training(train_data, max_seq_len)
    coords_test, seqs_test, y_test = ProteinStructureDataset.prep_for_training(test_data, max_seq_len)


    model = ClassificationModel(args)
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
                tgt_X_batch = coords_train[i : i + args.batch_size]
                tgt_seq_batch = seqs_train[i : i+args.batch_size]
                y_batch = y_train[i : i + args.batch_size]
            except:
                tgt_X_batch = coords_train[i :]
                tgt_seq_batch = seqs_train[i :]
                y_batch = y_train[i :]

            loss, y_hat = model(tgt_X_batch, tgt_seq_batch, y_batch, max_seq_len)
            
            y_batch = y_batch.reshape((y_batch.shape[0]*y_batch.shape[1],))
            y_hat = y_hat.reshape((y_hat.shape[0]*y_hat.shape[1],))

            print(roc_auc_score(y_batch.cpu().detach().numpy(),y_hat.cpu().detach().numpy()))

            true_vals.extend(y_batch.cpu().detach().numpy().tolist())
            pred_vals.extend(torch.sigmoid(y_hat).cpu().detach().numpy().tolist())
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

        scores_df = pd.DataFrame({'label':true_vals,'score':pred_vals})
        model.blm.add_model(f'epoch_{e}', scores_df)
        model.blm.plot_roc(model_names=[f'epoch_{e}'],params={"save":True,"prefix":f"charts/epoch_{e}_"})

        with gzip.GzipFile("model.pkl.gz", mode="wb", compresslevel=6) as fil:
            pickle.dump(model.blm, fil)

    # hold out test set
    true_vals_test = []
    pred_vals_test = []
    for i in range(0, len(test_data), args.batch_size):
        tgt_X_batch = coords_test[i : i + args.batch_size]
        tgt_seq_batch = seqs_test[i : i+args.batch_size]
        y_batch = y_test[i : i + args.batch_size]

        _, pred = model(tgt_X_batch.cuda(), tgt_seq_batch, y_batch.cuda(), max_seq_len)

        y_batch = y_batch.reshape((y_batch.shape[0]*y_batch.shape[1],))
        pred = pred.reshape((pred.shape[0]*pred.shape[1],))

        true_vals_test.extend(y_batch.cpu().detach().numpy().tolist())
        pred_vals_test.extend(torch.sigmoid(pred).cpu().detach().numpy().tolist())

    scores_df = pd.DataFrame({'label':true_vals_test,'score':pred_vals_test})
    model.blm.add_model('val', scores_df)
    model.blm.plot_roc(model_names=['val'],params={"save":True,"prefix":"charts/val_"})

    #TODO
    # run validation set at every epoch - 5 fold cross val (do cross val after debug code)
    # train more epochs, monitor loss on validation list



