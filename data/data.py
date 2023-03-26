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
        
        i = 0
        for entry in tqdm.tqdm(data, desc = 'data'):
            if type(entry['target_coords']) != torch.Tensor:
                entry['target_coords'] = torch.from_numpy(entry['target_coords']).float()
            if type(entry['binary_mask']) != torch.Tensor:
                entry['binary_mask'] = torch.tensor(entry['binary_mask']).int()
            if i < 0.7 * len(data):
                self.train_data.append(entry)
            else:
                self.test_data.append(entry)
            i += 1
            

    def prep_for_training(data):
        N = max([len(entry['binary_mask']) for entry in data])
        tgt_X = torch.zeros(len(data), N, 3)
        y = torch.zeros(len(data), N)
        tgt_seqs = []
        for i,b in enumerate(data):
            L = len(b['target_coords'])
            tgt_X[i,:L,:] = b['target_coords']
            y[i,:L] = b['binary_mask']
            tgt_seqs.append(b['target_seq'])
        return tgt_X, tgt_seqs, y