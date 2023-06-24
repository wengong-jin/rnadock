"""
Contains classes to represent protein + ligand data
"""

import pickle
import random
import torch
from tqdm import tqdm

class ProteinDataset():
    """
    Parent class to:
    ProteinBinaryDataset
    ProteinMulticlassDataset

    Data is stored as a nested dictionary:
        Train
            Fold 1
                Train
                    Protein
                        Coords
                        Seq
                    Ligand
                        Coords
                        Seq
                    Y
                Val
                    Same Keys
            Remaining Folds
        Test
            Same Keys
    """
    def __init__(self, path_to_pickle):
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)

    def prep_raw_data(self):

        for entry in tqdm.tqdm(data, desc = 'data'):
            if type(entry['target_coords']) != torch.Tensor:
                entry['target_coords'] = torch.from_numpy(entry['target_coords']).float()
            if 'ligand_coords' in data[0].keys() and type(entry['ligand_coords']) != torch.Tensor:
                entry['ligand_coords'] = torch.from_numpy(entry['ligand_coords']).float()
            if 'binary_mask' in data[0].keys() and type(entry['binary_mask']) != torch.Tensor:
                entry['binary_mask'] = torch.tensor(entry['binary_mask']).int()
            if 'pairwise_dists' in data[0].keys() and type(entry['pairwise_dists']) != torch.Tensor:
                entry['pairwise_dists'] = torch.from_numpy(entry['pairwise_dists']).float()

    def train_test_split(self, train_test_split=0.8, seq_split=True):

        self.data = {"Train": {}, "Test": {}}
        cluster_representatives = list(set([d.get('cluster', None) for d in self.raw_data]))

        train_target_coords = []
        train_target_seqs = []
        train_ligand_coords = []
        train_ligand_seqs = []
        train_masks = []
        train_y = []

        test_target_coords = []
        test_target_seqs = []
        test_ligand_coords = []
        test_ligand_seqs = []
        test_masks = []
        test_y = []

        if seq_split:
            # sequence similarity split - sequences with >=80% similarity will not be 
            # devided between train and test
            # train_test_split gives percentage of data points in train (0.8)
            for entry in tqdm.tqdm(data, desc = 'data'):
                if cluster_representatives.index(entry['cluster']) < (0.8 * len(cluster_representatives)):
                    train_target_coords.append(entry['target_coords'])
                    train_target_seqs.append(entry['target_seq'])
                    train_ligand_coords.append(entry['ligand_coords'])
                    train_ligand_seqs.append(entry['ligand_seqs'])
                    train_masks.append(entry['binary_mask'])
                    train_y.append(entry['pairwise_dists'])
                else:
                    test_target_coords.append(entry['target_coords'])
                    test_target_seqs.append(entry['target_seq'])
                    test_ligand_coords.append(entry['ligand_coords'])
                    test_ligand_seqs.append(entry['ligand_seqs'])
                    test_masks.append(entry['binary_mask'])
                    test_y.append(entry['pairwise_dists'])
        
        self.data["Train"]["Protein"]["Coords"] = train_target_coords
        self.data["Train"]["Protein"]["Seqs"] = train_target_seqs
        self.data["Train"]["Ligand"]["Coords"] = train_ligand_coords
        self.data["Train"]["Ligand"]["Seqs"] = train_ligand_seqs
        self.data["Train"]["Binary_Masks"] = train_masks
        self.data["Train"]["Pairwise_Dists"] = train_y

        self.data["Test"]["Protein"]["Coords"] = test_target_coords
        self.data["Test"]["Protein"]["Seqs"] = test_target_seqs
        self.data["Test"]["Ligand"]["Coords"] = test_ligand_coords
        self.data["Test"]["Ligand"]["Seqs"] = test_ligand_seqs
        self.data["Test"]["Binary_Masks"] = test_masks
        self.data["Test"]["Pairwise_Dists"] = test_y

        self.max_protein_length = max([len(seq) for seq in train_target_seqs+test_target_seqs])
        self.max_rna_length = max([len(seq) for seq in train_ligand_seqs+test_ligand_seqs])

    def split_train_into_folds(self, k):
        
        num_train = len(self.data["Train"]["Protein"]["Seqs"])
        rand_list = [random.randint(1,k) for n in range(num_train)]

        for i in range(k):
            self.data["Train"][f"Fold_{i}"]["Train"]["Protein"]["Coords"] = []
            self.data["Train"][f"Fold_{i}"]["Train"]["Protein"]["Seqs"] = []
            self.data["Train"][f"Fold_{i}"]["Train"]["Ligand"]["Coords"] = []
            self.data["Train"][f"Fold_{i}"]["Train"]["Ligand"]["Seqs"] = []
            self.data["Train"][f"Fold_{i}"]["Train"]["Binary_Masks"] = []
            self.data["Train"][f"Fold_{i}"]["Train"]["Pairwise_Dists"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["Protein"]["Coords"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["Protein"]["Seqs"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["Ligand"]["Coords"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["Ligand"]["Seqs"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["Binary_Masks"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["Pairwise_Dists"] = []

        for idx in rand_list:
            self.data["Train"][f"Fold_{idx}"]["Val"]["Protein"]["Coords"].append(self.data["Train"]["Protein"]["Coords"][idx])
            self.data["Train"][f"Fold_{idx}"]["Val"]["Protein"]["Seqs"].append(self.data["Train"]["Protein"]["Seqs"][idx])
            self.data["Train"][f"Fold_{idx}"]["Val"]["Ligand"]["Coords"].append(self.data["Train"]["Ligand"]["Coords"][idx])
            self.data["Train"][f"Fold_{idx}"]["Val"]["Ligand"]["Seqs"].append(self.data["Train"]["Ligand"]["Seqs"][idx])
            self.data["Train"][f"Fold_{idx}"]["Val"]["Binary_Masks"].append(self.data["Train"]["Binary_Masks"][idx])
            self.data["Train"][f"Fold_{idx}"]["Val"]["Pairwise_Dists"].append(self.data["Train"]["Pairwise_Dists"][idx])
            for i in range(k):
                if idx != i:
                    self.data["Train"][f"Fold_{idx}"]["Train"]["Protein"]["Coords"].append(self.data["Train"]["Protein"]["Coords"][idx])
                    self.data["Train"][f"Fold_{idx}"]["Train"]["Protein"]["Seqs"].append(self.data["Train"]["Protein"]["Seqs"][idx])
                    self.data["Train"][f"Fold_{idx}"]["Train"]["Ligand"]["Coords"].append(self.data["Train"]["Ligand"]["Coords"][idx])
                    self.data["Train"][f"Fold_{idx}"]["Train"]["Ligand"]["Seqs"].append(self.data["Train"]["Ligand"]["Seqs"][idx])
                    self.data["Train"][f"Fold_{idx}"]["Train"]["Binary_Masks"].append(self.data["Train"]["Binary_Masks"][idx])
                    self.data["Train"][f"Fold_{idx}"]["Train"]["Pairwise_Dists"].append(self.data["Train"]["Pairwise_Dists"][idx])

        self.data["Train"].pop("Protein")
        self.data["Train"].pop("Ligand")
        self.data["Train"].pop("Binary_Masks")
        self.data["Train"].pop("Pairwise_Dists")

    # A = max_num_protein_coords
    # B = max_num_rna_coords
    def prep_for_model(self, k):

        for split in ["Train", "Test"]:
            if split == "Train":
                for idx in range(k):
                    for split_2 in ["Train", "Val"]:
                        protein_X = torch.zeros(len(data[split][f"Fold_{idx}"][split_2]["Protein"]["Coords"]), self.max_protein_length, 3)
                        ligand_X = torch.zeros(len(data[split][f"Fold_{idx}"][split_2]["Ligand"]["Coords"]), self.max_protein_length, 3)
                        y = torch.zeros(len(data[split][f"Fold_{idx}"][split_2]["Pairwise_Dists"]), A, B)
                        for i in range(len(data[split][f"Fold_{idx}"][split_2]["Protein"]["Coords"])):
                            prot_len = len(data[split][f"Fold_{idx}"][split_2]["Protein"]["Coords"][i])
                            ligand_len = len(data[split][f"Fold_{idx}"][split_2]["Ligand"]["Coords"][i])
                            protein_X[i, :prot_len, :] = data[split][f"Fold_{idx}"][split_2]["Protein"]["Coords"][i]
                            ligand_X[i, :ligand_len, :] = data[split][f"Fold_{idx}"][split_2]["Ligand"]["Coords"][i]
                            y[i, :prot_len, :ligand_len] = data[split][f"Fold_{idx}"][split_2]["Pairwise_Dists"][i]
                        data[split][f"Fold_{idx}"][split_2]["Protein"]["Coords"] = protein_X
                        data[split][f"Fold_{idx}"][split_2]["Ligand"]["Coords"] = ligand_X
                        data[split][f"Fold_{idx}"][split_2]["Pairwise_Dists"] = y
            else:
                protein_X = torch.zeros(len(data[split]["Protein"]["Coords"]), self.max_protein_length, 3)
                ligand_X = torch.zeros(len(data[split]["Ligand"]["Coords"]), self.max_protein_length, 3)
                y = torch.zeros(len(data[split]["Pairwise_Dists"]), A, B)
                for i in range(len(data[split]["Protein"]["Coords"])):
                    prot_len = len(data[split]["Protein"]["Coords"][i])
                    ligand_len = len(data[split]["Ligand"]["Coords"][i])
                    protein_X[i, :prot_len, :] = data[split]["Protein"]["Coords"][i]
                    ligand_X[i, :ligand_len, :] = data[split]["Ligand"]["Coords"][i]
                    y[i, :prot_len, :ligand_len] = data[split]["Pairwise_Dists"][i]
                data[split]["Protein"]["Coords"] = protein_X
                data[split]["Ligand"]["Coords"] = ligand_X
                data[split]["Pairwise_Dists"] = y


    def get_batch(self, data, idx, batch_size):
        # get starting at index idx in data
        assert "Protein" in data and "Ligand" in data
        
        if idx+batch_size < data["Protein"]["Coords"].shape[0]:
            prot_X = data["Protein"]["Coords"][idx:idx+batch_size]
            ligand_X = data["Ligand"]["Coords"][idx:idx+batch_size]
            prot_seqs = data["Protein"]["Seqs"][idx:idx+batch_size]
            ligand_seqs = data["Ligand"]["Seqs"][idx:idx+batch_size]
            y = data["Pairwise_Dists"][idx:idx+batch_size]
        else:
            prot_X = data["Protein"]["Coords"][idx:]
            ligand_X = data["Ligand"]["Coords"][idx:]
            prot_seqs = data["Protein"]["Seqs"][idx:]
            ligand_seqs = data["Ligand"]["Seqs"][idx:]
            y = data["Pairwise_Dists"][idx:]

        return prot_X, ligand_X, prot_seqs, ligand_seqs, y


class ProteinBinaryDataset():
    def __init__(path_to_pickle):
        super.__init__(path_to_pickle)

    def prep_for_model(self, k):
        super.prep_for_model(k)
        for idx in range(k):
            data["Train"][f"Fold_{k}"]["Pairwise_Dists"] = np.where((data["Train"][f"Fold_{k}"]["Pairwise_Dists"] < 10) & (data["Train"][f"Fold_{k}"]["Pairwise_Dists"] != 0), 1., 0.)
        data["Test"]["Pairwise_Dists"] = np.where((data["Test"]["Pairwise_Dists"] < 10) & (data["Test"]["Pairwise_Dists"] != 0), 1., 0.)

class ProteinMulticlassDataset():
    def __init__(path_to_pickle):
        super.__init__(path_to_pickle)

    def prep_for_model(self, k, num_classes):
        super.prep_for_model(k)
        for idx in range(k):
            data["Train"][f"Fold_{k}"]["Pairwise_Dists"] = np.floor(data["Train"][f"Fold_{k}"]["Pairwise_Dists"])
            data["Train"][f"Fold_{k}"]["Pairwise_Dists"] = np.where(data["Train"][f"Fold_{k}"]["Pairwise_Dists"] >= num_classes - 1, float(num_classes - 1), data["Train"][f"Fold_{k}"]["Pairwise_Dists"])
        data["Test"]["Pairwise_Dists"] = np.floor(data["Test"]["Pairwise_Dists"])
        data["Test"]["Pairwise_Dists"] = np.where(data["Test"]["Pairwise_Dists"] >= num_classes - 1, float(num_classes - 1), data["Test"]["Pairwise_Dists"])