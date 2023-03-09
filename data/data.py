"""
Contains classes to represent protein + RNA data
"""
import numpy as np
import pickle
import torch
import tqdm

class ProteinStructureDataset():
    
    def __init__(self, data_path):

        self.data = []
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        for entry in tqdm.tqdm(data, desc = 'data'):
            if type(entry['target_coords']) != torch.Tensor:
                entry['target_coords'] = torch.from_numpy(entry['target_coords']).float()
            if type(entry['binary_mask']) != torch.Tensor:
                entry['binary_mask'] = torch.tensor(entry['binary_mask']).int()
            self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def prep_for_training(data):
        N = max([len(entry['binary_mask']) for entry in data])
        tgt_X = torch.zeros(len(data), N, 3)
        tgt_mask = torch.zeros(len(data), N)
        y = torch.zeros(len(data), N)
        for i,b in enumerate(data):
            L = len(b['target_coords'])
            tgt_X[i,:L,:] = b['target_coords']
            tgt_mask[i,:L] = 1
            y[i,:L] = b['binary_mask']
        return tgt_X, tgt_mask, y