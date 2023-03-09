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
import tqdm

amino_acids ={'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',    \
'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',    \
'GLY':'G', 'PRO':'P', 'CYS':'C'}

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

def bound_to_rna(rec_atom_coords, rna_coords_array):

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
    rna_coords = structure[structure.atom_name == "C3\'"].coord
    rna_seq = structure[structure.atom_name == "C3\'"].res_name
    print(target_coords.shape, rna_coords.shape, len(rna_seq))

    # don't use proteins with >2500 amino acids
    if len(target_coords) > 2500:
        raise Exception(f"{filepath} had more than 2500 residues")

    # compute binary mask
    mask = [0 for i in range(len(target_coords))]
    for j in range(len(target_coords)):
        rec_atom_coord = target_coords[j]
        if bound_to_rna(rec_atom_coord, rna_coords):
            mask[j] = 1

    return {'target_coords': target_coords, 'rna_seq': rna_seq, 'binary_mask': mask}
    

if __name__ == "__main__":
    pdb_out_dir = "data/pdb"
    data_csv = "data/select.csv"

    #load_and_filter(data_csv, pdb_out_dir)

    dirs = os.listdir(pdb_out_dir)

    dataset = []
    for f in tqdm.tqdm(dirs, desc = 'dirs'):
        filepath = os.path.join(pdb_out_dir, f)

        # returns entry dict with keys target_coords,
        # rna_seq, binary_mask
        try:
            entry = create_datapoint(filepath)
            dataset.append(entry)
            print(f"Successfully processed {filepath}.")
        except:
            print(f"Error on {filepath}.")

    with open('dataset.pickle', 'wb') as handle:
        pickle.dump(dataset, handle)