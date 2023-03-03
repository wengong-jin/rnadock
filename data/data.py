"""
Contains classes to represent protein + RNA data
"""
import numpy as np
import torch_geometric
import torch.utils.data as data

class ProteinStructureDataset(data.Dataset):
    """
    A 'torch.utils.data.Dataset' which converts list of dictionaries built in
    process_data.py into protein graphs
    # starting with simple coord info, will add atom featurizations,
    # dihedrals, etc.

    Portions from https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py
    """

    def __init__(self, data_list, device = "cpu"):

        super(ProteinGraphDataset, self).__init__()

        self.data_list = data_list
        self.device = device

    def __len__(self): return len(self.data_list)

    def __getprotein__(self, i): return self.featurize_as_graph(self.data_list[i])

    def __getmask__(self, i): return self.create_mask(self.data_list[i])

    def featurize_as_graph(self, entry):

        coords = torch.as_tensor(entry['coords'], 
                        device=self.device, 
                        dtype=torch.float32)   # shape (# of residues, 3)

        data = torch_geometric.data.Data(x=coords)

        return data

    def create_mask(self, entry):
        
        mask = torch.as_tensor(np.array(entry['binary_mask']), 
                        device=self.device, 
                        dtype=torch.int32) # shape (# of residues,)

        return mask