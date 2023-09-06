"""
Uses GeoBind datasplits on the following modes:
- dataset: RNA
- ligand_structure: alse
- binary
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
from tqdm import tqdm
from binary_label_metrics import BinaryLabelMetrics
from frame import FAEClassificationModel, FAEncoder, FrameAveraging
from ipa import IPAClassificationModel

sys.path.append("/home/dnori/rnadock/src/data")
from data import ProteinDataset, ProteinBinaryDataset, ProteinMulticlassDataset

def step(model, prot_coords, prot_seqs, pdb, y_residue, y_atom, res_ids, normal_modes, esm_repr=None):
    # train_dataset.max_protein_length = test_dataset.max_protein_length (same for rna length)
    if esm_repr is not None:
        token_repr = esm_repr[pdb[0]]
    else:
        token_repr = None
    loss, y_hat, y_batch, mask = model(prot_coords, None, prot_seqs, None, y_residue, res_ids, normal_modes, train_dataset.max_protein_length, train_dataset.max_rna_length, token_repr)
    y_batch = y_batch[mask.bool()].cpu().detach().numpy()
    y_hat = torch.sigmoid(y_hat)
    y_hat = y_hat[mask.bool()].cpu().detach().numpy()
    normal_modes = normal_modes[0,mask.bool(),0].cpu().detach().numpy()
    return y_batch, y_hat, loss, prot_seqs, pdb, normal_modes

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
    train_pkl_path = f"./src/data/geobind_train_rna_alphacarbons.pickle"
    test_pkl_path = f"./src/data/geobind_test_rna_alphacarbons.pickle"

    train_dataset = ProteinBinaryDataset(train_pkl_path)
    train_dataset.prep_raw_data()
    train_dataset.prep_geobind_data(split="Train")
    test_dataset = ProteinBinaryDataset(test_pkl_path)
    test_dataset.prep_raw_data()
    test_dataset.prep_geobind_data(split="Test")

    if train_dataset.max_protein_length > test_dataset.max_protein_length:
        test_dataset.max_protein_length = train_dataset.max_protein_length
    else:
        train_dataset.max_protein_length = test_dataset.max_protein_length

    if train_dataset.max_rna_length > test_dataset.max_rna_length:
        test_dataset.max_rna_length = train_dataset.max_rna_length
    else:
        train_dataset.max_rna_length = test_dataset.max_rna_length

    train_dataset.prep_for_non_pairwise_model(split="Train")
    test_dataset.prep_for_non_pairwise_model(split="Test")

    num_train_points = len(train_dataset.data["Train"]["Protein"]["Seqs"])
    num_test_points = len(test_dataset.data["Test"]["Protein"]["Seqs"])

    model = FAEClassificationModel(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    sequences = []
    predictions = []
    ground_truth = []
    pdbs = []

    file = open('src/data/geobind_train_esm_embeddings.pickle', 'rb')
    esm_repr = pickle.load(file)
    file.close() 

    for e in range(args.epochs):
        model.train()
        true_vals = []
        pred_vals = []
        for i in tqdm(range(0, num_train_points, args.batch_size)):
            y_batch, y_hat, loss, prot_seq_batch, pdb_batch, nm = step(model, *train_dataset.get_batch(i, args.batch_size, "Train"), esm_repr=esm_repr)
            true_vals.extend(y_batch.tolist())
            pred_vals.extend(y_hat.tolist())

            predictions.append(y_hat.tolist())
            ground_truth.append(y_batch.tolist())
            sequences.append(prot_seq_batch[0])
            pdbs.append(pdb_batch[0])

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

        scores_df = pd.DataFrame({'label':true_vals,'score':pred_vals})
        model.blm.add_model(f'epoch_{e}', scores_df)
        model.blm.plot_roc(model_names=[f'epoch_{e}'],params={"save":True,"prefix":f"output/geobind_comparison/epoch_{e}_"})

        filename = f'output/model_checkpoints/geobind_comparison/epoch_{e}.pt'
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, filename)

    checkpoint = torch.load(f"/home/dnori/rnadock/output/model_checkpoints/geobind_comparison/epoch_7.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    # hold out test set
    true_vals_test = []
    pred_vals_test = []

    # for visualizations
    sequences = []
    predictions = []
    ground_truth = []
    normal_modes = []

    file = open('src/data/geobind_test_esm_embeddings.pickle', 'rb')
    esm_repr_test = pickle.load(file)
    file.close() 

    for i in tqdm(range(0, num_test_points, args.batch_size)):
        y_batch, y_hat, loss, prot_seq_batch, pdb_batch, nm = step(model, *test_dataset.get_batch(i, args.batch_size, "Test"), esm_repr=esm_repr_test)
        true_vals_test.extend(y_batch.tolist())
        pred_vals_test.extend(y_hat.tolist())
        predictions.append(y_hat.tolist())
        ground_truth.append(y_batch.tolist())
        sequences.append(prot_seq_batch[0])
        normal_modes.append(nm.tolist())

    data = {'sequences': sequences,
         'predictions': predictions,
         'ground_truth': ground_truth,
         'normal_modes': normal_modes}

    df = pd.DataFrame.from_dict(data)
    df.to_csv('output/geobind_comparison/visualization_pred_info.csv')

    scores_df = pd.DataFrame({'label':true_vals_test,'score':pred_vals_test})
    model.blm.add_model(f'test', scores_df)
    model.blm.plot_roc(model_names=['test'],params={"save":True,"prefix":f"output/geobind_comparison/test_8th_epoch_"})
    model.blm.plot(model_names=['test'],chart_types=[1,2,3,4,5],params={"save":True,"prefix":f"output/geobind_comparison/test_8th_epoch_"})