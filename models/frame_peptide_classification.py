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

class PeptideClassificationModel(nn.Module):

    def __init__(self, args):
        super(PeptideClassificationModel, self).__init__()
        self.blm = BinaryLabelMetrics()
        self.protein_encoder = FAEncoder(args, "protein")
        self.peptide_encoder = FAEncoder(args, "peptide")
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

    def forward(self, prot_X, peptide_X, seqs, peptide_seqs, label, max_prot_seq_len, max_peptide_seq_len):
        esm_input = []
        pad_mask = torch.from_numpy(np.ones((prot_X.shape[0],max_prot_seq_len)))
        for s in range(len(seqs)):
            sequence = seqs[s] + "<pad>"*(max_prot_seq_len - len(seqs[s]))
            pad_mask[s,len(seqs[s]):] = 0
            esm_input.append((f"protein_{s}",sequence))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(esm_input)

        esm_input_peptide = []
        pad_mask_peptide = torch.from_numpy(np.ones((peptide_X.shape[0],max_peptide_seq_len)))
        for s in range(len(peptide_seqs)):
            sequence = peptide_seqs[s] + "<pad>"*(max_peptide_seq_len - len(peptide_seqs[s]))
            pad_mask_peptide[s,len(seqs[s]):] = 0
            esm_input_peptide.append((f"peptide_{s}",sequence))
        batch_labels_peptide, batch_strs_peptide, batch_tokens_peptide = self.batch_converter(esm_input_peptide)

        # per residue representation
        with torch.no_grad():
            self.esm_model.to(torch.float16).cuda()

            token_repr = torch.from_numpy(np.zeros((batch_tokens.shape[0],max_prot_seq_len,1280)))
            for i in range(len(batch_tokens)):
                results = self.esm_model(batch_tokens[i:i+1,:].cuda(), repr_layers=[33], return_contacts=True)
                token_repr[i,:,:] = results["representations"][33][:,1:-1,:]

            token_repr_peptide = torch.from_numpy(np.zeros((batch_tokens_peptide.shape[0],max_peptide_seq_len,1280)))
            for i in range(len(batch_tokens_peptide)):
                results_peptide = self.esm_model(batch_tokens_peptide[i:i+1,:].cuda(), repr_layers=[33], return_contacts=True)
                token_repr_peptide[i,:,:] = results_peptide["representations"][33][:,1:-1,:]

        h_prot = self.protein_encoder(prot_X.cuda(), h_S=token_repr, mask=pad_mask.cuda()) # h_prot: (batch sz, max prot seq len, encoder hidden dim)
        h_pep = self.peptide_encoder(peptide_X.cuda(), h_S=token_repr_peptide, mask=pad_mask_peptide.cuda()) #h_peptide: (batch sz, max peptide seq len, encoder hidden dim)

        # Repeat and tile h_prot and h_pep to create tensors of shape (batch_size, max_prot_seq_len, max_peptide_seq_len, encoder_hidden_dim)
        h_prot_tiled = h_prot.unsqueeze(2).repeat(1, 1, max_peptide_seq_len, 1)
        h_pep_tiled = h_pep.unsqueeze(1).repeat(1, max_prot_seq_len, 1, 1)

        # Concatenate h_prot_tiled and h_pep_tiled along the last dimension to create a tensor of shape (batch_size, max_prot_seq_len, max_peptide_seq_len, 2*encoder_hidden_dim)
        h_concat = torch.cat([h_prot_tiled, h_pep_tiled], dim=-1)

        # Reshape h_concat to create the final tensor of shape (batch_size, max_prot_seq_len * max_peptide_seq_len, 2*encoder_hidden_dim)
        h = h_concat.view(len(h_prot), max_prot_seq_len * max_peptide_seq_len, 2*args.encoder_hidden_size)

        label = torch.from_numpy(y_batch).view(-1)
        mask = torch.unsqueeze(torch.mm(pad_mask.T, pad_mask_peptide).flatten(), dim=0)
        
        pred = self.linear_layers(h.float()).squeeze()
        #loss = F.cross_entropy(pred, label.cuda(), reduction = 'none') # pred is logits, label is OHE
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
        elif mol_type == "peptide":
            self.encoder = SRUpp(
                    args.esm_emb_size + 3,
                    args.encoder_hidden_size // 2,
                    args.encoder_hidden_size // 2,
                    num_layers=args.depth,
                    dropout=args.dropout,
                    bidirectional=True,
            ).cuda().float()
        self.mol_type = mol_type

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

    dataset = ProteinStructureDataset("dataset_peptide.pickle")
    train_data = dataset.train_data
    test_data = dataset.test_data
    print('Train/test data:', len(train_data), len(test_data))
    
    max_prot_coords = max([len(entry['target_coords']) for entry in train_data+test_data])
    max_peptide_coords = max([len(entry['ligand_coords']) for entry in train_data+test_data])
    print('Max protein len:', max_prot_coords)
    print('Max peptide len:', max_peptide_coords)

    prot_coords_train, peptide_coords_train, y_train, prot_seqs_train, peptide_seqs_train = ProteinStructureDataset.prep_dists_for_training(train_data, max_prot_coords, max_peptide_coords)
    prot_coords_test, peptide_coords_test, y_test, prot_seqs_test, peptide_seqs_test = ProteinStructureDataset.prep_dists_for_training(test_data, max_prot_coords, max_peptide_coords)

    # num_pos = torch.sum(y_train) + torch.sum(y_test)
    # num_neg = (y_train.shape[0]*y_train.shape[1] + y_test.shape[0]*y_test.shape[1]) - num_pos
    # loss_positive_weight = num_neg / num_pos # loss_positive_weight negative examples for every 1 positive example
    # print(num_neg, num_pos)

    # cast to multiclass problem
    # y_train = np.where(y_train < 10, 0, np.where(y_train > 20, 2, 1))
    # y_test = np.where(y_test < 10, 0, np.where(y_test > 20, 2, 1))

    # cast to binary problem
    y_train = np.where(y_train < 10, 1., 0.)
    y_test = np.where(y_test < 10, 1., 0.)

    model = PeptideClassificationModel(args)
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
                prot_X_batch = prot_coords_train[i : i + args.batch_size]
                peptide_X_batch = peptide_coords_train[i : i + args.batch_size]
                prot_seq_batch = prot_seqs_train[i : i+args.batch_size]
                peptide_seq_batch = peptide_seqs_train[i : i+args.batch_size]
                y_batch = y_train[i : i + args.batch_size]
            except:
                prot_X_batch = prot_coords_train[i :]
                peptide_X_batch = peptide_coords_train[i :]
                prot_seq_batch = prot_seqs_train[i :]
                peptide_seq_batch = peptide_seqs_train[i :]
                y_batch = y_train[i :]

            loss, y_hat, y_batch = model(prot_X_batch, peptide_X_batch, prot_seq_batch, peptide_seq_batch, y_batch, max_prot_coords, max_peptide_coords)
            # y_hat = torch.argmax(y_hat, dim=1)

            # print(f1_score(y_batch.cpu().detach().numpy(),y_hat.cpu().detach().numpy(), average='macro'))
            print(roc_auc_score(y_batch.cpu().detach().numpy(),y_hat.cpu().detach().numpy()))

            true_vals.extend(y_batch.cpu().detach().numpy().tolist())
            pred_vals.extend(y_hat.cpu().detach().numpy().tolist())
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

        scores_df = pd.DataFrame({'label':true_vals,'score':pred_vals})
        model.blm.add_model(f'epoch_{e}', scores_df)
        model.blm.plot_roc(model_names=[f'epoch_{e}'],params={"save":True,"prefix":f"charts/epoch_{e}_"})

        # n_classes = 3
        # y_true_bin = label_binarize(true_vals, classes=range(n_classes))
        # y_pred_bin = label_binarize(pred_vals, classes=range(n_classes))

        # precision = dict()
        # recall = dict()
        # average_precision = dict()
        # for i in range(n_classes): 
        #     precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
        #     average_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_bin[:, i])

        # # plot precision-recall curves for each class
        # colors = ['red', 'blue', 'green']
        # plt.figure(figsize=(8,6))
        # for i, color in zip(range(n_classes), colors):
        #     plt.plot(recall[i], precision[i], color=color, lw=2,
        #             label='Precision-Recall curve of class {0} (area = {1:0.2f})'
        #             ''.format(i, average_precision[i]))
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.legend(loc="lower left")
        # plt.title("Precision-Recall curve")
        # plt.savefig(f"charts/epoch_{e}_pr.png")

        filename = f'model_checkpoints/peptide_model_multiclass_epoch_{e}.pt'
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, filename)

    checkpoint = torch.load("model_checkpoints/peptide_model_multiclass_epoch_9.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    # hold out test set
    true_vals_test = []
    pred_vals_test = []
    for i in range(0, len(test_data), args.batch_size):
        prot_X_batch = prot_coords_test[i : i + args.batch_size]
        peptide_X_batch = peptide_coords_test[i : i + args.batch_size]
        prot_seq_batch = prot_seqs_test[i : i+args.batch_size]
        peptide_seq_batch = peptide_seqs_test[i : i+args.batch_size]
        y_batch = y_test[i : i + args.batch_size]

        loss, y_hat, y_batch = model(prot_X_batch, peptide_X_batch, prot_seq_batch, peptide_seq_batch, y_batch, max_prot_coords, max_peptide_coords)
        y_hat = torch.argmax(y_hat, dim=1)

        print(f1_score(y_batch.cpu().detach().numpy(),y_hat.cpu().detach().numpy(), average='macro'))

        true_vals_test.extend(y_batch.cpu().detach().numpy().tolist())
        pred_vals_test.extend(y_hat.cpu().detach().numpy().tolist())

    scores_df = pd.DataFrame({'label':true_vals_test,'score':pred_vals_test})
    model.blm.add_model('val', scores_df)
    model.blm.plot_roc(model_names=['val'],params={"save":True,"prefix":"charts/val_"})

    # n_classes = 3
    # y_true_bin = label_binarize(true_vals_test, classes=range(n_classes))
    # y_pred_bin = label_binarize(pred_vals_test, classes=range(n_classes))

    # precision = dict()
    # recall = dict()
    # average_precision = dict()
    # for i in range(n_classes): 
    #     precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
    #     average_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_bin[:, i])

    # # plot precision-recall curves for each class
    # colors = ['red', 'blue', 'green']
    # plt.figure(figsize=(8,6))
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(recall[i], precision[i], color=color, lw=2,
    #             label='Precision-Recall curve of class {0} (area = {1:0.2f})'
    #             ''.format(i, average_precision[i]))
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.legend(loc="lower left")
    # plt.title("Precision-Recall curve")
    # plt.savefig(f"charts/val_pr.png")