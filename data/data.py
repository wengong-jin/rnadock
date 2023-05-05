"""
Contains classes to represent protein + RNA data
"""
import Bio
import numpy as np
import pickle
import torch
import tqdm

class ProteinStructureDataset():
    
    def __init__(self, data_path):

        self.train_data = []
        self.test_data = []
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        cluster_representatives = list(set([d.get('cluster', None) for d in data]))
        
        # i = 0
        for entry in tqdm.tqdm(data, desc = 'data'):
            if type(entry['target_coords']) != torch.Tensor:
                entry['target_coords'] = torch.from_numpy(entry['target_coords']).float()
            if type(entry['ligand_coords']) != torch.Tensor:
                entry['ligand_coords'] = torch.from_numpy(entry['ligand_coords']).float()
            if type(entry['binary_mask']) != torch.Tensor:
                entry['binary_mask'] = torch.tensor(entry['binary_mask']).int()
            if type(entry['pairwise_dists']) != torch.Tensor:
                entry['pairwise_dists'] = torch.from_numpy(entry['pairwise_dists']).float()

            # sequence similarity split
            if cluster_representatives.index(entry['cluster']) < (0.8 * len(cluster_representatives)):
                self.train_data.append(entry)
            else:
                self.test_data.append(entry)
            
            # random split
            # if i < 0.8*len(data):
            #     self.train_data.append(entry)
            # else:
            #     self.test_data.append(entry)
            # i+=1

    def prep_for_training(data, N):
        tgt_X = torch.zeros(len(data), N, 3)
        y = torch.zeros(len(data), N)
        tgt_seqs = []
        rna_seqs = []
        for i,b in enumerate(data):
            L = len(b['target_coords'])
            tgt_X[i,:L,:] = b['target_coords']
            y[i,:L] = b['binary_mask']
            tgt_seqs.append(b['target_seq'])
            rna_seqs.append(b['ligand_seq']) #change rna seq name so same code used
        return tgt_X, tgt_seqs, y

    # A = max_num_protein_coords
    # B = max_num_rna_coords
    def prep_dists_for_training(data, A, B):
        protein_X = torch.zeros(len(data), A, 3)
        rna_X = torch.zeros(len(data), B, 3)
        y = torch.zeros(len(data), A, B)
        prot_seqs = []
        rna_seqs = []
        for i, b in enumerate(data):
            P = len(b['target_coords'])
            R = len(b['ligand_coords'])
            protein_X[i, :P, :] = b['target_coords']
            rna_X[i, :R, :] = b['ligand_coords']
            y[i, :P, :R] = b['pairwise_dists'] # torch.nn.functional.normalize(b['pairwise_dists']) #normalize dists for each complex
            prot_seqs.append(b['target_seq'])
            rna_seqs.append(b['ligand_seq'])
        
        return protein_X, rna_X, y, prot_seqs, rna_seqs

