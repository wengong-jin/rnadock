"""
Processes select.csv from https://beta.nakb.org/download.html
Filters for protein-RNA complexes, saves appropriate PDB complexes
Pickles graph representations of protein + RNA complexes,
generates binary mask over structure based on distance to RNA
"""
import biotite
import biotite.structure.io.pdb as pdb
import numpy as np
import os
import pandas as pd
import pickle
import urllib.request
from scipy.spatial.distance import cdist
import tqdm

protein_letters_3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

nucleotides = ['A','U','C','G']

def load_and_filter(csv_file, pdb_out_dir):

    print(f"Processing {csv_file}.")

    # filter to protein-RNA complexes
    orig_df = pd.read_csv(csv_file)
    rna_df = orig_df[orig_df["polyclass"] == "Protein/RNA"][["pdbid"]].reset_index(drop=True)

    print(f"Number of Data Points: {len(rna_df)}.")

    # retrieve PDB complexes
    pdb_ids = list(set(rna_df["pdbid"].tolist()))
    i = 0
    for pdb_id in pdb_ids:
        try:
            urllib.request.urlretrieve(f'http://files.rcsb.org/download/{pdb_id}.pdb', 
            f'{pdb_out_dir}/{pdb_id}.pdb')
            i+=1
        except:
            print(f"PDB not found for {pdb_id}.")

    print(f"Saved {i} complexes in {pdb_out_dir}.")

def bound(rec_atom_coords, rna_coords_array):

    dists = (rna_coords_array - rec_atom_coords)**2
    dists = np.sum(dists, axis=1)
    dists = np.sqrt(dists)
    
    if np.min(dists) < 10:
        return True
    else:
        return False

def create_datapoint(filepath):


    pdb_file = pdb.PDBFile.read(filepath)
    structure = pdb_file.get_structure()[0]

    target_coords = structure[structure.atom_name == 'CA'].coord
    target_seq = structure[structure.atom_name == 'CA'].res_name
    rna_coords = structure[structure.atom_name == "C3\'"].coord
    rna_seq = structure[structure.atom_name == "C3\'"].res_name
    print(len(rna_seq))
    name = filepath[9:-4]

    if len(target_coords) > 1200 or len(rna_coords) > 1000:
        raise Exception(f"{filepath} had >1500 residues or >1000 nucleotides")

    # compute binary mask
    mask = [0 for i in range(len(target_coords))]
    for j in range(len(target_coords)):
        rec_atom_coord = target_coords[j]
        if bound(rec_atom_coord, rna_coords):
            mask[j] = 1
    distances = cdist(target_coords, rna_coords) # number of protein coords x number of rna coords

    # modify target sequence (remove nonstandard amino acids)
    try:
        target_seq = ''.join([protein_letters_3to1[three_letter_code] for three_letter_code 
                                                in target_seq.tolist()])
    except Exception as e:
        return None
    
    # filter out complexes where no alpha carbons are within 7 Angstroms of RNA
    if sum (mask) > 0:
        return {'target_coords': target_coords, 'ligand_seq': rna_seq, 'ligand_coords': rna_coords, 'target_seq':target_seq, 'binary_mask': mask, 'cluster':'na', 'pairwise_dists': distances}, name, target_seq
    else:
        return None

def create_peptide_datapoint(filepath):


    pdb_file = pdb.PDBFile.read(filepath)
    structure = pdb_file.get_structure()[0]

    chain_1, chain_2 = tuple(set(structure.chain_id))
    chain_1 = structure[structure.chain_id == chain_1]
    chain_2 = structure[structure.chain_id == chain_2]

    chain1_seq = chain_1[chain_1.atom_name == 'CA'].res_name
    chain2_seq = chain_2[chain_2.atom_name == 'CA'].res_name

    if len(chain2_seq) < len(chain1_seq):
        target_coords = chain_1[chain_1.atom_name == 'CA'].coord
        target_seq = chain1_seq
        peptide_coords = chain_2[chain_2.atom_name == 'CA'].coord
        peptide_seq = chain2_seq
    else:
        peptide_coords = chain_1[chain_1.atom_name == 'CA'].coord
        peptide_seq = chain1_seq
        target_coords = chain_2[chain_2.atom_name == 'CA'].coord
        target_seq = chain2_seq

    name = filepath[13:-4]
    print(name)

    if len(target_coords) > 1200:
        raise Exception(f"{filepath} had >1200 residues")

    # compute binary mask
    mask = [0 for i in range(len(target_coords))]
    for j in range(len(target_coords)):
        rec_atom_coord = target_coords[j]
        if bound(rec_atom_coord, peptide_coords):
            mask[j] = 1
    distances = cdist(target_coords, peptide_coords) # number of protein coords x number of peptide coords

    # modify target sequence (remove nonstandard amino acids)
    try:
        target_seq = ''.join([protein_letters_3to1[three_letter_code] for three_letter_code 
                                                in target_seq.tolist()])
        peptide_seq = ''.join([protein_letters_3to1[three_letter_code] for three_letter_code 
                                                in peptide_seq.tolist()])
    except Exception as e:
        return None
    
    # filter out complexes where no alpha carbons are within 10 Angstroms of RNA
    if sum (mask) > 0:
        return {'target_coords': target_coords, 'ligand_seq': peptide_seq, 'ligand_coords': peptide_coords, 'target_seq':target_seq, 'binary_mask': mask, 'cluster':'na', 'pairwise_dists': distances}, name, target_seq
    else:
        return None
    
if __name__ == "__main__":
    pdb_out_dir = "data/pdb"
    data_csv = "data/select.csv"

    # load_and_filter(data_csv, pdb_out_dir)

    dirs = os.listdir(pdb_out_dir)

    dataset = {}
    list_seq = []
    list_name = []
    for f in tqdm.tqdm(dirs, desc = 'dirs'):
        filepath = os.path.join(pdb_out_dir, f)

        # returns entry dict with keys target_coords,
        # rna_seq, binary_mask
        try:
            entry, name, seq = create_datapoint(filepath)
            if entry:
                dataset[name] = entry
                list_seq.append(seq)
                list_name.append(name)
            else:
                raise Exception(f"{filepath} has no alpha carbons within 10 Angstroms.")
            print(f"Successfully processed {filepath}.")
        except Exception as e:
            print(f"Error on {filepath}.")
            print(e)

    ofile = open("data/fasta.txt", "w")
    for i in range(len(list_seq)):
        ofile.write(">" + list_name[i] + "\n" +list_seq[i] + "\n")
    ofile.close()

    os.system("mmseqs easy-cluster data/fasta.txt clusterRes tmp --min-seq-id 0.5 --cov-mode 1")

    with open("clusterRes_all_seqs.fasta") as f:
        lines = f.readlines()
        current_cluster = lines[0][1:-1]
        print(current_cluster)
        for i in range(1, len(lines)-1):
            if lines[i][0] == ">" and lines[i-1][0] == ">":
                # new cluster label
                current_cluster = lines[i-1][1:-1]
                dataset[lines[i][1:-1]]["cluster"] = current_cluster
            elif lines[i][0] == ">":
                dataset[lines[i][1:-1]]["cluster"] = current_cluster
            else:
                pass
    
    #peptide dataset
    with open('dataset.pickle', 'wb') as handle:
        pickle.dump(list(dataset.values()), handle)


# first letter is peptide, second letter is receptor

