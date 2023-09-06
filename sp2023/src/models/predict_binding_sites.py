"""
General script which takes the following options:
- dataset: RNA or peptide
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
from tqdm import tqdm
from binary_label_metrics import BinaryLabelMetrics
from frame import ClassificationModel, FAEncoder, FrameAveraging

sys.path.append("/home/dnori/rnadock/src/data")
from data import ProteinDataset, ProteinBinaryDataset, ProteinMulticlassDataset

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

def step(model, prot_coords, ligand_coords, prot_seqs, ligand_seqs, y, pdb, chains):
    loss, y_hat, y_batch, mask = model(prot_coords, ligand_coords, prot_seqs, ligand_seqs, y, dataset.max_protein_length, dataset.max_rna_length)

    if args.classification_type == "binary":
        y_hat = torch.sigmoid(y_hat)
    elif args.classification_type == "multiclass":
        y_hat_mm = F.softmax(y_hat, dim=1)
        y_hat = torch.sum(y_hat_mm.cpu() * torch.arange(args.mlp_output_dim), dim=1) #weighted average

    if args.ligand_structure:
        square = (dataset.max_protein_length, dataset.max_rna_length)
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
        return y_batch, y_hat, loss, prot_seqs, pdb, chains
    else:
        return y_batch, y_hat_mm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--esm_emb_size', type=int, default=1280)
    parser.add_argument('--encoder_hidden_size', type=int, default=200)
    parser.add_argument('--mlp_hidden_size', type=int, default=200)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--mlp_output_dim', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--anneal_rate', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ligand_structure', type=bool, default=False)
    parser.add_argument('--ligand_type',type=str,default='rna')
    parser.add_argument('--classification_type',type=str,default='binary')
    parser.add_argument('--k_fold_cross_val',type=int,default=4)
    parser.add_argument('--train_test_split',type=float,default=1.0) # all in in train for cross val
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    pkl_path = f"./src/data/dataset_{args.ligand_type}_2.pickle"

    if args.classification_type == "multiclass" and args.mlp_output_dim < 2:
        raise ValueError("for multiclass, output dimension must be > 1")
    elif args.classification_type == "binary" and args.mlp_output_dim >= 2:
        raise ValueError("for binary, output dimension must be 1")

    if args.classification_type == "binary":
        dataset = ProteinBinaryDataset(pkl_path)
    elif args.classification_type == "multiclass":
        dataset = ProteinMulticlassDataset(pkl_path)

    dataset.prep_raw_data()
    dataset.train_test_split(train_test_split=args.train_test_split)
    dataset.split_train_into_folds(args.k_fold_cross_val)
    dataset.prep_for_model(args.k_fold_cross_val)

    # make new splits with data.py fix and run
    file = open('output/cross_val/rna_dataset_static.pickle', 'wb')
    pickle.dump(dataset, file)
    file.close()

    # file = open('output/cross_val/rna_dataset_static.pickle', 'rb')
    # dataset = pickle.load(file)
    # file.close()


    # 4 fold cross val
    for f in range(1, args.k_fold_cross_val + 1):
        print(f"Training Fold {f}")
        num_train_points = len(dataset.data["Train"][f"Fold_{f}"]["Train"]["Protein"]["Seqs"])
        num_val_points = len(dataset.data["Train"][f"Fold_{f}"]["Val"]["Protein"]["Seqs"])

        model = ClassificationModel(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

        sequences = []
        predictions = []
        ground_truth = []
        pdbs = []
        chain_ids = []
        
        # train
        for e in range(args.epochs):
            model.train()
            true_vals = []
            pred_vals = []
            for i in tqdm(range(0, num_train_points, args.batch_size)):
                if args.classification_type == "binary":
                    y_batch, y_hat, loss, prot_seq_batch, pdb_batch, chains_batch = step(model, *dataset.get_batch(i, args.batch_size, f, "Train","Train"))
                    true_vals.extend(y_batch.tolist())
                    pred_vals.extend(y_hat.tolist())
                    predictions.append(y_hat.tolist())
                    ground_truth.append(y_batch.tolist())
                    sequences.append(prot_seq_batch[0])
                    pdbs.append(pdb_batch[0])
                    chain_ids.append(chains_batch[0])
                else:
                    y_batch, y_hat_mm = step(model, *dataset.get_batch(i, args.batch_size, f, "Train","Train"))
                    true_vals.extend(y_batch.tolist())
                    pred_vals.append(y_hat_mm)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()

            if args.classification_type == "binary":
                visualization_df = pd.DataFrame(
                {'pdbs': pdbs,
                    'sequences': sequences,
                    'chain_ids': chain_ids,
                    'predictions': predictions,
                    'ground_truth': ground_truth
                })
                visualization_df.to_csv(f"output/cross_val/charts_rna_binary_noligand_struct_seq/visualization_epoch_{e}_fold_{f}_train_info.csv",index=False)

                scores_df = pd.DataFrame({'label':true_vals,'score':pred_vals})
                model.blm.add_model(f'fold_{f}_epoch_{e}', scores_df)
                model.blm.plot_roc(model_names=[f'fold_{f}_epoch_{e}'],params={"save":True,"prefix":f"output/cross_val/charts_rna_binary_noligand_struct_seq/fold_{f}_epoch_{e}_"})
            else:
                names = [f"{i} Angstroms" for i in range(args.mlp_output_dim)]
                true_vals = np.array(true_vals)
                pred_vals = np.concatenate(pred_vals, axis=0)
                blm = ovr(true_vals, pred_vals, names)
                blm.plot_roc(chart_types=[1,2], params={"legloc":4, "addsz":False, "save":True, "prefix":f"output/cross_val/charts_rna_binary_noligand_struct_seq/fold_{f}_epoch_{e}_"})

            filename = f'output/model_checkpoints/cross_val/fold_{f}_epoch_{e}.pt'
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, filename)

            # cross val
            model.eval()
            sequences = []
            predictions = []
            ground_truth = []
            pdbs = []
            chain_ids = []
            print(f"Validating Fold {f} Epoch {e}")
            for i in tqdm(range(0, num_val_points, args.batch_size)):
                if args.classification_type == "binary":
                    y_batch, y_hat, loss, prot_seq_batch, pdb_batch, chains_batch = step(model, *dataset.get_batch(i, args.batch_size, f, "Train","Val"))
                    true_vals.extend(y_batch.tolist())
                    pred_vals.extend(y_hat.tolist())
                    predictions.append(y_hat.tolist())
                    ground_truth.append(y_batch.tolist())
                    sequences.append(prot_seq_batch[0])
                    pdbs.append(pdb_batch[0])
                    chain_ids.append(chains_batch[0])
                else:
                    y_batch, y_hat_mm = step(model, *dataset.get_batch(i, args.batch_size, f, "Train", "Val"))
                    true_vals.extend(y_batch.tolist())
                    pred_vals.append(y_hat_mm)

            if args.classification_type == "binary":
                visualization_df = pd.DataFrame(
                {'pdbs': pdbs,
                    'sequences': sequences,
                    'chain_ids': chain_ids,
                    'predictions': predictions,
                    'ground_truth': ground_truth
                })
                visualization_df.to_csv(f"output/cross_val/charts_rna_binary_noligand_struct_seq/visualization_epoch_{e}_fold_{f}_val_info.csv",index=False)

                scores_df = pd.DataFrame({'label':true_vals,'score':pred_vals})
                model.blm.add_model(f'fold_{f}_epoch_{e}_val', scores_df)
                model.blm.plot_roc(model_names=[f'fold_{f}_epoch_{e}_val'],params={"save":True,"prefix":f"output/cross_val/charts_rna_binary_noligand_struct_seq/fold_{f}_epoch_{e}_val_"})
            else:
                names = [f"{i} Angstroms" for i in range(args.mlp_output_dim)]
                true_vals = np.array(true_vals)
                pred_vals = np.concatenate(pred_vals, axis=0)
                blm = ovr(true_vals, pred_vals, names)
                blm.plot_roc(chart_types=[1,2], params={"legloc":4, "addsz":False, "save":True, "prefix":f"output/cross_val/charts_rna_binary_noligand_struct_seq/fold_{f}_epoch_{e}_val_"})


    # checkpoint = torch.load("output/model_checkpoints/trial/fold_2_epoch_1.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.to("cpu")
    # num_test_points = len(dataset.data["Test"]["Protein"]["Seqs"])
 
    # # hold out test set
    # true_vals_test = []
    # pred_vals_test = []

    # # for visualizations
    # sequences = []
    # predictions = []
    # ground_truth = []

    # for i in range(0, num_test_points, args.batch_size):
    #     if args.classification_type == "binary":
    #         y_batch, y_hat, loss, prot_seq_batch = step(model, *dataset.get_batch(i, args.batch_size, f, "Test",""))
    #         true_vals_test.extend(y_batch.tolist())
    #         pred_vals_test.extend(y_hat.tolist())
    #         predictions.append(y_hat.tolist())
    #         ground_truth.append(y_batch.tolist())
    #         sequences.append(prot_seq_batch[0])
    #     else:
    #         y_batch, y_hat_mm = step(model, *dataset.get_batch(i, args.batch_size, f, "Test", ""))
    #         true_vals_test.extend(y_batch.tolist())
    #         pred_vals_test.append(y_hat_mm)

    # if args.classification_type == "binary":
    #     # visualization_df = pd.DataFrame(
    #     #     {'sequences': sequences,
    #     #     'predictions': predictions,
    #     #     'ground_truth': ground_truth
    #     # })
    #     # visualization_df.to_csv("charts_rna_binary_noligand_fullrun/visualization_info.csv",index=False)

    #     scores_df = pd.DataFrame({'label':true_vals_test,'score':pred_vals_test})
    #     model.blm.add_model(f'test', scores_df)
    #     model.blm.plot_roc(model_names=['val'],params={"save":True,"prefix":f"charts_rna_binary_noligand_cross_val/test_undertrained_"})
    #     model.blm.plot(model_names=['val'],chart_types=[1,2,3,4,5],params={"save":True,"prefix":f"charts_rna_binary_noligand_cross_val/test_undertrained_"})
    # else:
    #     names = [f"{i} Angstroms" for i in range(args.mlp_output_dim)]
    #     true_vals_test = np.array(true_vals_test)
    #     pred_vals_test = np.concatenate(pred_vals_test, axis=0)
    #     blm = ovr(true_vals_test, pred_vals_test, names)
    #     blm.plot_roc(chart_types=[1,2], params={"legloc":4, "addsz":False, "save":True, "prefix":f"charts_rna_binary_noligand_cross_val/test_"})
        