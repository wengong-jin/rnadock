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
    
    if np.min(dists) < 5:
        return True
    else:
        return False

def create_rna_datapoint(filepath):


    pdb_file = pdb.PDBFile.read(filepath)
    structure = pdb_file.get_structure()[0]

    target_coords = structure[structure.atom_name == 'CA'].coord # operating just on backbone carbon
    target_seq = structure[structure.atom_name == 'CA'].res_name
    target_chain_ids = structure[structure.atom_name == 'CA'].chain_id.tolist()
    rna_coords = structure[structure.atom_name == "C3\'"].coord
    rna_seq = structure[structure.atom_name == "C3\'"].res_name
    name = filepath[9:-4]

    if len(target_coords) > 1200 or len(rna_coords) > 800 or len(target_coords) < 20 or len(rna_coords) < 5:
        raise Exception(f"{filepath} had >1200 residues or >800 nucleotides or too short.")

    # compute binary mask
    mask = [0 for i in range(len(target_coords))]
    for j in range(len(target_coords)):
        rec_atom_coord = target_coords[j]
        if bound(rec_atom_coord, rna_coords):
            mask[j] = 1
    distances = cdist(target_coords, rna_coords) # number of protein coords x number of rna coords

    # modify target/rna sequences (remove nonstandard amino acids/nucleotides)
    clean_target_seq = []
    for three_letter_code in target_seq.tolist():
        if three_letter_code in protein_letters_3to1:
            clean_target_seq.append(protein_letters_3to1[three_letter_code])
        else:
            clean_target_seq.append('X') #nonstandard amino acid
    
    clean_rna_seq = []
    for nucleotide in rna_seq:
        if nucleotide in ['A', 'U', 'C', 'G']:
            clean_rna_seq.append(nucleotide)
        else:
            clean_rna_seq.append('Z') #nonstandard nucleotide

    # filter out complexes where no alpha carbons are within 10 Angstroms of RNA
    if sum (mask) > 0:
        return {'pdb': name[4:], 'target_chain_ids':target_chain_ids,'target_coords': target_coords, 'ligand_seq': ''.join(clean_rna_seq), 'ligand_coords': rna_coords, 'target_seq': ''.join(clean_target_seq), 'binary_mask': mask, 'cluster':'na', 'pairwise_dists': distances}, name, ''.join(clean_target_seq)
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

def create_geobind_datapoint(line):
    # input is one line in txt file
    pdb_id = line.split(":")[0]
    protein_chain = line.split(":")[1]
    ligand_chains = line.split(":")[2].split("\t")[0].split("_")
    filepath = f'/home/dnori/rnadock/GeoBind/Dataset/downloaded_pdbs_test/{pdb_id}.pdb'

    try:
        urllib.request.urlretrieve(f'http://files.rcsb.org/download/{pdb_id}.pdb', filepath)
        pdb_file = pdb.PDBFile.read(filepath)
        structure = pdb_file.get_structure()[0]
    except:
        print(pdb_id)
        return []

    dps = []

    for lc in ligand_chains:
        chain_1 = structure[structure.chain_id == protein_chain]
        chain_2 = structure[structure.chain_id == lc]

        bound_res_ids = " ".join(line[line.index(f"{protein_chain}_{lc}"):].split(" ")[1:])
        if ":" in bound_res_ids:
            bound_res_ids = bound_res_ids[:bound_res_ids.index(":")]
        else:
            bound_res_ids = bound_res_ids[:-1]
        bound_res_ids = [int(s[1:]) for s in bound_res_ids.split(" ")]

        full_target_coords = chain_1.coord
        target_coords = chain_1[chain_1.atom_name == 'CA'].coord # operating just on backbone carbon
        res_ids = chain_1[chain_1.atom_name == 'CA'].res_id
        target_seq = chain_1[chain_1.atom_name == 'CA'].res_name
        rna_coords = chain_2[chain_2.atom_name == "C3\'"].coord
        rna_seq = chain_2[chain_2.atom_name == "C3\'"].res_name
        name = f"{pdb_id}_{protein_chain}_{lc}"

        # compute binary mask
        mask = [0 for i in range(len(target_coords))]
        for j in range(len(target_coords)):
            if res_ids[j] in bound_res_ids:
                mask[j] = 1

        # modify target/rna sequences (remove nonstandard amino acids/nucleotides)
        clean_target_seq = []
        for three_letter_code in target_seq.tolist():
            if three_letter_code in protein_letters_3to1:
                clean_target_seq.append(protein_letters_3to1[three_letter_code])
            else:
                clean_target_seq.append('X') #nonstandard amino acid
        
        clean_rna_seq = []
        for nucleotide in rna_seq:
            if nucleotide in ['A', 'U', 'C', 'G']:
                clean_rna_seq.append(nucleotide)
            else:
                clean_rna_seq.append('Z') #nonstandard nucleotide

        dps.append({'pdb': name, 'target_coords': target_coords, 'ligand_seq': ''.join(clean_rna_seq), 'ligand_coords': rna_coords, 'target_seq': ''.join(clean_target_seq), 'binary_mask': mask})

    return dps        

def create_non_rbp_datapoint(filepath):


    pdb_file = pdb.PDBFile.read(filepath)
    structure = pdb_file.get_structure()[0]


    cid = list(set(structure.chain_id))[0]
    chain = structure[structure.chain_id == cid]
    target_seq = chain[chain.atom_name == 'CA'].res_name
    target_coords = chain[chain.atom_name == 'CA'].coord
    name = filepath[13:-4]

    if len(target_coords) > 1000 or len(target_coords) < 100:
        raise Exception(f"{filepath} had >1000 or <100 residues.")

    # modify target sequence (remove nonstandard amino acids)
    try:
        target_seq = ''.join([protein_letters_3to1[three_letter_code] for three_letter_code 
                                                in target_seq.tolist()])
    except Exception as e:
        return None
    
    return {'target_coords': target_coords, 'target_seq':target_seq, 'cluster':'na'}, name, target_seq
    
if __name__ == "__main__":
    pdb_out_dir = "src/data/pdb"
    data_csv = "src/data/select.csv"

    # load_and_filter(data_csv, pdb_out_dir)

    # dirs = os.listdir(pdb_out_dir)

    # dataset = {}
    # list_seq = []
    # list_name = []
    # for f in tqdm.tqdm(dirs, desc = 'dirs'):
    #     filepath = os.path.join(pdb_out_dir, f)

    #     # returns entry dict with keys target_coords,
    #     # rna_seq, binary_mask
    #     try:
    #         entry, name, seq = create_rna_datapoint(filepath)
    #         if entry:
    #             dataset[name] = entry
    #             list_seq.append(seq)
    #             list_name.append(name)
    #         else:
    #             raise Exception(f"{filepath} has no alpha carbons within 10 Angstroms.")
    #         print(f"Successfully processed {filepath}.")
    #     except Exception as e:
    #         print(f"Error on {filepath}.")
    #         print(e)

    # ofile = open("src/data/fasta_binding_site_prediction.txt", "w")
    # for i in range(len(list_seq)):
    #     print(list_name[i])
    #     print(list_seq[i])
    #     ofile.write(">" + list_name[i] + "\n" +list_seq[i] + "\n")
    # ofile.close()

    # os.system("mmseqs easy-cluster src/data/fasta_binding_site_prediction.txt clusterRes tmp --min-seq-id 0.8 --cov-mode 1")

    # with open("clusterRes_all_seqs.fasta") as f:
    #     lines = f.readlines()
    #     current_cluster = lines[0][1:-1]
    #     print(current_cluster)
    #     for i in range(1, len(lines)-1):
    #         if lines[i][0] == ">" and lines[i-1][0] == ">":
    #             # new cluster label
    #             current_cluster = lines[i-1][1:-1]
    #             dataset[lines[i][1:-1]]["cluster"] = current_cluster
    #         elif lines[i][0] == ">":
    #             dataset[lines[i][1:-1]]["cluster"] = current_cluster
    #         else:
    #             pass
    
    # with open('src/data/dataset_rna_2.pickle', 'wb') as handle:
    #     pickle.dump(list(dataset.values()), handle)

    file1 = open("/home/dnori/rnadock/GeoBind/Dataset_lists/GeoBind/RNA-663_Train.txt", "r")
    lines = file1.readlines()
    datapoints = []
    for line in tqdm.tqdm(lines):
        datapoints.extend(create_geobind_datapoint(line))
    with open('src/data/geobind_train_rna.pickle', 'wb') as handle:
        pickle.dump(datapoints, handle)
