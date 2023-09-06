"""
Contains classes to represent protein + ligand data
"""
import numpy as np
import pickle
import random
import torch
import tqdm

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
        with open(path_to_pickle, 'rb') as f:
            self.raw_data = pickle.load(f)
        self.data = {}

    def prep_raw_data(self):

        for entry in tqdm.tqdm(self.raw_data, desc = 'prep raw data'):
            if type(entry['target_coords']) != torch.Tensor:
                entry['target_coords'] = torch.from_numpy(entry['target_coords']).float()
            if type(entry['res_ids']) != torch.Tensor:
                entry['res_ids'] = torch.from_numpy(entry['res_ids'])
            if type(entry['normal_modes']) != torch.Tensor:
                entry['normal_modes'] = torch.from_numpy(entry['normal_modes'])
            if type(entry['atom_binary_mask']) != torch.Tensor:
                entry['atom_binary_mask'] = torch.tensor(entry['atom_binary_mask'])
            if type(entry['residue_binary_mask']) != torch.Tensor:
                entry['residue_binary_mask'] = torch.tensor(entry['residue_binary_mask'])
            if 'ligand_coords' in self.raw_data[0].keys() and type(entry['ligand_coords']) != torch.Tensor:
                entry['ligand_coords'] = torch.from_numpy(entry['ligand_coords']).float()
            if 'binary_mask' in self.raw_data[0].keys() and type(entry['binary_mask']) != torch.Tensor:
                entry['binary_mask'] = torch.tensor(entry['binary_mask']).int()
            if 'pairwise_dists' in self.raw_data[0].keys() and type(entry['pairwise_dists']) != torch.Tensor:
                entry['pairwise_dists'] = torch.from_numpy(entry['pairwise_dists']).float()

    def train_test_split(self, train_test_split=0.8, seq_split=True, k=4):
        self.data = {"Train": {"Protein":{}, "Ligand":{}}, "Test": {"Protein":{}, "Ligand":{}}}
        cluster_representatives = list(set([d.get('cluster', None) for d in self.raw_data]))
        split_idx = int((train_test_split * len(cluster_representatives)))
        rand_cluster_list = [random.randint(1,k) for n in range(split_idx)]

        train_target_coords = []
        train_target_seqs = []
        train_target_chain_ids = []
        train_ligand_coords = []
        train_ligand_seqs = []
        train_masks = []
        train_y = []
        train_pdbs = []

        test_target_coords = []
        test_target_seqs = []
        test_target_chain_ids = []
        test_ligand_coords = []
        test_ligand_seqs = []
        test_masks = []
        test_y = []
        test_pdbs = []

        self.rand_fold_list = []

        if seq_split:
            # sequence similarity split - sequences with >=80% similarity will not be 
            # devided between train and test
            # train_test_split gives percentage of data points in train (0.8)
            for entry in tqdm.tqdm(self.raw_data, desc = 'train test split'):
                if cluster_representatives.index(entry['cluster']) < split_idx:
                    train_target_coords.append(entry['target_coords'])
                    train_target_seqs.append(entry['target_seq'])
                    train_ligand_coords.append(entry['ligand_coords'])
                    train_ligand_seqs.append(entry['ligand_seq'])
                    train_masks.append(entry['binary_mask'])
                    train_y.append(entry['pairwise_dists'])
                    train_pdbs.append(entry['pdb'])
                    train_target_chain_ids.append(entry['target_chain_ids'])
                    self.rand_fold_list.append(rand_cluster_list[cluster_representatives.index(entry['cluster'])])

                else:
                    test_target_coords.append(entry['target_coords'])
                    test_target_seqs.append(entry['target_seq'])
                    test_ligand_coords.append(entry['ligand_coords'])
                    test_ligand_seqs.append(entry['ligand_seq'])
                    test_masks.append(entry['binary_mask'])
                    test_y.append(entry['pairwise_dists'])
                    test_pdbs.append(entry['pdb'])
                    test_target_chain_ids.append(entry['target_chain_ids'])

        self.data["Train"]["Protein"]["Coords"] = train_target_coords
        self.data["Train"]["Protein"]["Seqs"] = train_target_seqs
        self.data["Train"]["Protein"]["Chains"] = train_target_chain_ids
        self.data["Train"]["Ligand"]["Coords"] = train_ligand_coords
        self.data["Train"]["Ligand"]["Seqs"] = train_ligand_seqs
        self.data["Train"]["Binary_Masks"] = train_masks
        self.data["Train"]["Pairwise_Dists"] = train_y
        self.data["Train"]["PDB"] = train_pdbs

        self.data["Test"]["Protein"]["Coords"] = test_target_coords
        self.data["Test"]["Protein"]["Seqs"] = test_target_seqs
        self.data["Test"]["Protein"]["Chains"] = test_target_chain_ids
        self.data["Test"]["Ligand"]["Coords"] = test_ligand_coords
        self.data["Test"]["Ligand"]["Seqs"] = test_ligand_seqs
        self.data["Test"]["Binary_Masks"] = test_masks
        self.data["Test"]["Pairwise_Dists"] = test_y
        self.data["Test"]["PDB"] = test_pdbs

        self.max_protein_length = max([len(seq) for seq in train_target_seqs+test_target_seqs])
        self.max_rna_length = max([len(seq) for seq in train_ligand_seqs+test_ligand_seqs])

    def prep_geobind_data(self, split="Train"):
        self.data[split] = {"Protein":{}, "Ligand":{},"Binary_Masks":{}}
        target_coords = []
        target_seqs = []
        ligand_coords = []
        ligand_seqs = []
        residue_masks = []
        atom_masks = []
        pdbs = []
        res_ids = []
        normal_modes = []

        for entry in tqdm.tqdm(self.raw_data, desc = f'prep geobind {split}'):
            target_coords.append(entry['target_coords'])
            target_seqs.append(entry['target_seq'])
            ligand_coords.append(entry['ligand_coords'])
            ligand_seqs.append(entry['ligand_seq'])
            atom_masks.append(entry['atom_binary_mask'])
            residue_masks.append(entry['residue_binary_mask'])
            pdbs.append(entry['pdb'])
            res_ids.append(entry['res_ids'])
            normal_modes.append(entry['normal_modes'])

        self.data[split]["Protein"]["Coords"] = target_coords
        self.data[split]["Protein"]["Seqs"] = target_seqs
        self.data[split]["Protein"]["Res_Ids"] = res_ids
        self.data[split]["Ligand"]["Coords"] = ligand_coords
        self.data[split]["Ligand"]["Seqs"] = ligand_seqs
        self.data[split]["Binary_Masks"]["Residue"] = residue_masks
        self.data[split]["Binary_Masks"]["Atom"] = atom_masks
        self.data[split]["PDB"] = pdbs
        self.data[split]["Normal_Modes"] = normal_modes

        self.max_protein_length = max([coord.shape[0] for coord in target_coords])
        self.max_rna_length = max([coord.shape[0] for coord in ligand_coords])

    def split_train_into_folds(self, k):
        
        num_train = len(self.data["Train"]["Protein"]["Seqs"])
        rand_list = self.rand_fold_list

        for i in range(1,k+1):
            self.data["Train"][f"Fold_{i}"] = {"Train": {"Protein":{}, "Ligand":{}}, "Val": {"Protein":{}, "Ligand":{}}}
            self.data["Train"][f"Fold_{i}"]["Train"]["Protein"]["Coords"] = []
            self.data["Train"][f"Fold_{i}"]["Train"]["Protein"]["Seqs"] = []
            self.data["Train"][f"Fold_{i}"]["Train"]["Protein"]["Chains"] = []
            self.data["Train"][f"Fold_{i}"]["Train"]["Ligand"]["Coords"] = []
            self.data["Train"][f"Fold_{i}"]["Train"]["Ligand"]["Seqs"] = []
            self.data["Train"][f"Fold_{i}"]["Train"]["Binary_Masks"] = []
            self.data["Train"][f"Fold_{i}"]["Train"]["Pairwise_Dists"] = []
            self.data["Train"][f"Fold_{i}"]["Train"]["PDB"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["Protein"]["Coords"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["Protein"]["Seqs"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["Protein"]["Chains"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["Ligand"]["Coords"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["Ligand"]["Seqs"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["Binary_Masks"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["Pairwise_Dists"] = []
            self.data["Train"][f"Fold_{i}"]["Val"]["PDB"] = []

        for i in range(len(rand_list)):
            idx = rand_list[i]
            self.data["Train"][f"Fold_{idx}"]["Val"]["Protein"]["Coords"].append(self.data["Train"]["Protein"]["Coords"][i])
            self.data["Train"][f"Fold_{idx}"]["Val"]["Protein"]["Seqs"].append(self.data["Train"]["Protein"]["Seqs"][i])
            self.data["Train"][f"Fold_{idx}"]["Val"]["Protein"]["Chains"].append(self.data["Train"]["Protein"]["Chains"][i])
            self.data["Train"][f"Fold_{idx}"]["Val"]["Ligand"]["Coords"].append(self.data["Train"]["Ligand"]["Coords"][i])
            self.data["Train"][f"Fold_{idx}"]["Val"]["Ligand"]["Seqs"].append(self.data["Train"]["Ligand"]["Seqs"][i])
            self.data["Train"][f"Fold_{idx}"]["Val"]["Binary_Masks"].append(self.data["Train"]["Binary_Masks"][i])
            self.data["Train"][f"Fold_{idx}"]["Val"]["Pairwise_Dists"].append(self.data["Train"]["Pairwise_Dists"][i])
            self.data["Train"][f"Fold_{idx}"]["Val"]["PDB"].append(self.data["Train"]["PDB"][i])
            for i in range(1,k+1):
                if idx != i:
                    self.data["Train"][f"Fold_{idx}"]["Train"]["Protein"]["Coords"].append(self.data["Train"]["Protein"]["Coords"][i])
                    self.data["Train"][f"Fold_{idx}"]["Train"]["Protein"]["Seqs"].append(self.data["Train"]["Protein"]["Seqs"][i])
                    self.data["Train"][f"Fold_{idx}"]["Train"]["Protein"]["Chains"].append(self.data["Train"]["Protein"]["Chains"][i])
                    self.data["Train"][f"Fold_{idx}"]["Train"]["Ligand"]["Coords"].append(self.data["Train"]["Ligand"]["Coords"][i])
                    self.data["Train"][f"Fold_{idx}"]["Train"]["Ligand"]["Seqs"].append(self.data["Train"]["Ligand"]["Seqs"][i])
                    self.data["Train"][f"Fold_{idx}"]["Train"]["Binary_Masks"].append(self.data["Train"]["Binary_Masks"][i])
                    self.data["Train"][f"Fold_{idx}"]["Train"]["Pairwise_Dists"].append(self.data["Train"]["Pairwise_Dists"][i])
                    self.data["Train"][f"Fold_{idx}"]["Train"]["PDB"].append(self.data["Train"]["PDB"][i])

        self.data["Train"].pop("Protein")
        self.data["Train"].pop("Ligand")
        self.data["Train"].pop("Binary_Masks")
        self.data["Train"].pop("Pairwise_Dists")
        self.data["Train"].pop("PDB")

    # A = max_num_protein_coords
    # B = max_num_rna_coords
    def prep_for_model(self, k):

        for split in ["Train", "Test"]:
            if split == "Train":
                for idx in range(1,k+1):
                    for split_2 in ["Train", "Val"]:
                        protein_X = torch.zeros(len(self.data[split][f"Fold_{idx}"][split_2]["Protein"]["Coords"]), self.max_protein_length, 3)
                        ligand_X = torch.zeros(len(self.data[split][f"Fold_{idx}"][split_2]["Ligand"]["Coords"]), self.max_rna_length, 3)
                        y = torch.zeros(len(self.data[split][f"Fold_{idx}"][split_2]["Pairwise_Dists"]), self.max_protein_length, self.max_rna_length)
                        for i in range(len(self.data[split][f"Fold_{idx}"][split_2]["Protein"]["Coords"])):
                            prot_len = len(self.data[split][f"Fold_{idx}"][split_2]["Protein"]["Coords"][i])
                            ligand_len = len(self.data[split][f"Fold_{idx}"][split_2]["Ligand"]["Coords"][i])
                            protein_X[i, :prot_len, :] = self.data[split][f"Fold_{idx}"][split_2]["Protein"]["Coords"][i]
                            ligand_X[i, :ligand_len, :] = self.data[split][f"Fold_{idx}"][split_2]["Ligand"]["Coords"][i]
                            y[i, :prot_len, :ligand_len] = self.data[split][f"Fold_{idx}"][split_2]["Pairwise_Dists"][i]
                        self.data[split][f"Fold_{idx}"][split_2]["Protein"]["Coords"] = protein_X
                        self.data[split][f"Fold_{idx}"][split_2]["Ligand"]["Coords"] = ligand_X
                        self.data[split][f"Fold_{idx}"][split_2]["Pairwise_Dists"] = y
            else:
                protein_X = torch.zeros(len(self.data[split]["Protein"]["Coords"]), self.max_protein_length, 3)
                ligand_X = torch.zeros(len(self.data[split]["Ligand"]["Coords"]), self.max_rna_length, 3)
                y = torch.zeros(len(self.data[split]["Pairwise_Dists"]), self.max_protein_length, self.max_rna_length)
                for i in range(len(self.data[split]["Protein"]["Coords"])):
                    prot_len = len(self.data[split]["Protein"]["Coords"][i])
                    ligand_len = len(self.data[split]["Ligand"]["Coords"][i])
                    protein_X[i, :prot_len, :] = self.data[split]["Protein"]["Coords"][i]
                    ligand_X[i, :ligand_len, :] = self.data[split]["Ligand"]["Coords"][i]
                    y[i, :prot_len, :ligand_len] = self.data[split]["Pairwise_Dists"][i]
                self.data[split]["Protein"]["Coords"] = protein_X
                self.data[split]["Ligand"]["Coords"] = ligand_X
                self.data[split]["Pairwise_Dists"] = y

    

    def get_batch(self, idx, batch_size, f, split_1, split_2):
        if split_1 == "Train":
            data = self.data[split_1][f"Fold_{f}"][split_2]
        else:
            data = self.data[split_1]
        
        if idx+batch_size < data["Protein"]["Coords"].shape[0]:
            prot_X = data["Protein"]["Coords"][idx:idx+batch_size]
            ligand_X = data["Ligand"]["Coords"][idx:idx+batch_size]
            prot_seqs = data["Protein"]["Seqs"][idx:idx+batch_size]
            ligand_seqs = data["Ligand"]["Seqs"][idx:idx+batch_size]
            chains = data["Protein"]["Chains"][idx:idx+batch_size]
            y = data["Pairwise_Dists"][idx:idx+batch_size]
            pdb = data["PDB"][idx:idx+batch_size]
        else:
            prot_X = data["Protein"]["Coords"][idx:]
            ligand_X = data["Ligand"]["Coords"][idx:]
            prot_seqs = data["Protein"]["Seqs"][idx:]
            ligand_seqs = data["Ligand"]["Seqs"][idx:]
            y = data["Pairwise_Dists"][idx:]
            pdb = data["PDB"][idx:]
            chains = data["Protein"]["Chains"][idx:]

        return prot_X, ligand_X, prot_seqs, ligand_seqs, y, pdb, chains


class ProteinBinaryDataset(ProteinDataset):
    def __init__(self, path_to_pickle):
        super().__init__(path_to_pickle)

    def prep_for_model(self, k):
        super().prep_for_model(k)
        for idx in range(1,k+1):
            train_mask = torch.where((self.data["Train"][f"Fold_{idx}"]["Train"]["Pairwise_Dists"] < 10) & (self.data["Train"][f"Fold_{idx}"]["Train"]["Pairwise_Dists"] != 0), 1., 0.)
            val_mask = torch.where((self.data["Train"][f"Fold_{idx}"]["Val"]["Pairwise_Dists"] < 10) & (self.data["Train"][f"Fold_{idx}"]["Val"]["Pairwise_Dists"] != 0), 1., 0.)
            self.data["Train"][f"Fold_{idx}"]["Train"]["Pairwise_Dists"] = train_mask
            self.data["Train"][f"Fold_{idx}"]["Val"]["Pairwise_Dists"] = val_mask
        test_mask = torch.where((self.data["Test"]["Pairwise_Dists"] < 10) & (self.data["Test"]["Pairwise_Dists"] != 0), 1., 0.)
        self.data["Test"]["Pairwise_Dists"] = test_mask

    def prep_for_non_pairwise_model(self, split):
        protein_X = torch.zeros(len(self.data[split]["Protein"]["Coords"]), self.max_protein_length, 3)
        binary_y_atom = torch.zeros(len(self.data[split]["Binary_Masks"]["Atom"]), self.max_protein_length, 1)
        # binary_y_residue = torch.zeros(len(self.data[split]["Binary_Masks"]["Residue"]), self.max_protein_length//3, 1)
        binary_y_residue = torch.zeros(len(self.data[split]["Binary_Masks"]["Residue"]), self.max_protein_length, 1)
        normal_modes = torch.zeros(len(self.data[split]["Protein"]["Coords"]), self.max_protein_length, 50)
        for i in range(len(self.data[split]["Protein"]["Coords"])):
            prot_len = len(self.data[split]["Protein"]["Coords"][i])
            num_nm = self.data[split]["Normal_Modes"][i].shape[1]
            protein_X[i, :prot_len, :] = self.data[split]["Protein"]["Coords"][i]
            binary_y_atom[i, :prot_len, 0] = self.data[split]["Binary_Masks"]["Atom"][i]
            binary_y_residue[i, :prot_len, 0] = self.data[split]["Binary_Masks"]["Residue"][i]
            normal_modes[i, :prot_len, :num_nm] = self.data[split]["Normal_Modes"][i]
        self.data[split]["Protein"]["Coords"] = protein_X
        self.data[split]["Binary_Masks"]["Residue"] = binary_y_residue
        self.data[split]["Binary_Masks"]["Atom"] = binary_y_atom
        self.data[split]["Normal_Modes"] = normal_modes

    def get_batch(self, idx, batch_size, split):
        data = self.data[split]

        if idx+batch_size < data["Protein"]["Coords"].shape[0]:
            prot_X = data["Protein"]["Coords"][idx:idx+batch_size]
            prot_seqs = data["Protein"]["Seqs"][idx:idx+batch_size]
            pdb = data["PDB"][idx:idx+batch_size]
            binary_y_atom = data["Binary_Masks"]["Atom"][idx:idx+batch_size]
            binary_y_residue = data["Binary_Masks"]["Residue"][idx:idx+batch_size]
            res_ids = data["Protein"]["Res_Ids"][idx:idx+batch_size]
            normal_modes = data["Normal_Modes"][idx:idx+batch_size]
        else:
            prot_X = data["Protein"]["Coords"][idx:]
            prot_seqs = data["Protein"]["Seqs"][idx:]
            pdb = data["PDB"][idx:]
            binary_y_atom = data["Binary_Masks"]["Atom"][idx:]
            binary_y_residue = data["Binary_Masks"]["Residue"][idx:]
            res_ids = data["Protein"]["Res_Ids"][idx:]
            normal_modes = data["Normal_Modes"][idx:]

        return prot_X, prot_seqs, pdb, binary_y_residue, binary_y_atom, res_ids, normal_modes

class ProteinMulticlassDataset(ProteinDataset):
    def __init__(self, path_to_pickle):
        super().__init__(path_to_pickle)

    def prep_for_model(self, k, num_classes):
        super().prep_for_model(k)
        for idx in range(1,k+1):
            self.data["Train"][f"Fold_{k}"]["Train"]["Pairwise_Dists"] = torch.floor(self.data["Train"][f"Fold_{idx}"]["Train"]["Pairwise_Dists"])
            self.data["Train"][f"Fold_{k}"]["Train"]["Pairwise_Dists"] = torch.where(self.data["Train"][f"Fold_{idx}"]["Train"]["Pairwise_Dists"] >= num_classes - 1, float(num_classes - 1), self.data["Train"][f"Fold_{idx}"]["Train"]["Pairwise_Dists"])
            self.data["Train"][f"Fold_{k}"]["Val"]["Pairwise_Dists"] = torch.floor(self.data["Train"][f"Fold_{idx}"]["Val"]["Pairwise_Dists"])
            self.data["Train"][f"Fold_{k}"]["Val"]["Pairwise_Dists"] = torch.where(self.data["Train"][f"Fold_{idx}"]["Val"]["Pairwise_Dists"] >= num_classes - 1, float(num_classes - 1), self.data["Train"][f"Fold_{idx}"]["Val"]["Pairwise_Dists"])
        self.data["Test"]["Pairwise_Dists"] = torch.floor(self.data["Test"]["Pairwise_Dists"])
        self.data["Test"]["Pairwise_Dists"] = torch.where(self.data["Test"]["Pairwise_Dists"] >= num_classes - 1, float(num_classes - 1), self.data["Test"]["Pairwise_Dists"])