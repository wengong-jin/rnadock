"""
Contains classes to represent protein + RNA data
"""
import torch.utils.data as data

class ProteinStructureDataset(data.Dataset):
    """
    A 'torch.utils.data.Dataset' which converts list of dictionaries built in
    process_data.py into protein graphs
    # starting with simple coord info, will add atom featurizations,
    # dihedrals, etc.
    """

    def __init__(self, data_list, device = "cpu"):

        super(ProteinGraphDataset, self).__init__()

        self.data_list = data_list
        self.device = device

    