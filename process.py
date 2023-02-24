import csv
import sys
import os
import math
import pickle
import numpy as np
import json
from prody import *
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
from sidechainnet.utils.measure import get_seq_coords_and_angles
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


na_map = {
    "DA": "a", "DC": "c", "DG": "g", "DT": "t", "DU": "u",
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "U",
}
RESTYPE_1to3 = {
     "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN","E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
aa_map = {y:x for x,y in RESTYPE_1to3.items()}

NA_ATOM14 = {
    "a": ["P","C3'","C1'","N9","C4","N3","C2","N1","C6","C5","N7","C8","N6",""],
    "c": ["P","C3'","C1'","N1","C2","O2","N3","C4","N4","C5","C6","","",""],
    "g": ["P","C3'","C1'","N9","C4","N3","C2","N1","C6","C5","N7","C8","N2","O6"],
    "t": ["P","C3'","C1'","N1","C2","O2","N3","C4","O4","C5","C7","C6","",""],
    "u": ["P","C3'","C1'","","","","","","","","","","",""],
    "A": ["P","C3'","C1'","N1","C2","N3","C4","C5","C6","N6","N7","C8","N9",""],
    "C": ["P","C3'","C1'","N1","C2","O2","N3","C4","N4","C5","C6","","",""],
    "G": ["P","C3'","C1'","N1","C2","N2","N3","C4","C5","C6","O6","N7","C8","N9"],
    "U": ["P","C3'","C1'","N1","C2","O2","N3","C4","O4","C5","C6","","",""],
    "T": ["P","C3'","C1'","","","","","","","","","","",""],
}

def get_coord(res):
    coord = np.zeros((14,3))
    for i,atype in enumerate(NA_ATOM14[na_map[res.getResname()]]):
        if atype != "":
            atom = res.select(f'name {atype}')
            if atom: coord[i, :] = atom.getCoords()[0]
    return coord

def process(tup):
    pdb_id, aff = tup
    structure = parsePDB(f'raw/{pdb_id}.ent.pdb', model=1)
    if structure.select('protein') is None or structure.select('nucleic') is None:
        return None

    proteins = []
    for pchain in structure.select('protein').copy().iterChains():
        try:
            pid = pchain.getChid()
            _, pcoords, pseq, _, _ = get_seq_coords_and_angles(pchain)
            pcoords = pcoords.reshape((len(pseq), 14, 3))
            proteins.append((pid, pseq, pcoords))
        except:
            continue

    nucleics = []
    for nchain in structure.select('nucleic').copy().iterChains():
        try:
            nid = nchain.getChid()
            nseq = ''.join([na_map[res.getResname()] for res in nchain.iterResidues()])
            ncoords = np.concatenate([get_coord(res) for res in nchain.iterResidues()], axis=0)
            ncoords = ncoords.reshape((len(nseq), 14, 3))
            nucleics.append((nid, nseq, ncoords))
        except:
            continue

    if len(proteins) == 0 or len(nucleics) == 0:
        return None
    
    dnas = [(cid, seq, coords) for cid, seq, coords in nucleics if seq.lower() == seq]
    rnas = [(cid, seq, coords) for cid, seq, coords in nucleics if seq.upper() == seq]
    dna_pairs = []
    dna_singles = []
    matched = set()
    for i,(aid, aseq, acoords) in enumerate(dnas):
        if aid in matched: continue
        count = 0
        for j,(bid, bseq, bcoords) in enumerate(dnas):
            if i >= j: continue
            acenter = acoords[:, 1].mean(axis=0)
            bcenter = bcoords[:, 1].mean(axis=0)
            if np.linalg.norm(acenter - bcenter) < 10:
                count += 1
                matched.add(aid)
                matched.add(bid)
                dna_pairs.append(
                        (aid + bid, aseq + bseq, np.concatenate([acoords, bcoords], axis=0))
                )
        assert count <= 1
        if count == 0:
            dna_singles.append((aid, aseq, acoords))
    
    nucleics = dna_pairs + dna_singles + rnas
    data = defaultdict(list)
    for (nid, nseq, ncoords) in nucleics:
        plist = []
        for (pid, pseq, pcoords) in proteins:
            X = pcoords[None,:,1,:]  # [1,N,3]
            Y = ncoords[:,None,1,:]  # [M,1,3]
            D = np.linalg.norm(X - Y, axis=-1)
            if D.min() < 10:
                plist.append((pid, pseq, pcoords))
        if len(plist) > 0:
            pid, pseq, pcoords = zip(*plist)
            pid = ''.join(pid)
            pseq = ''.join(pseq)
            pcoords = np.concatenate(pcoords, axis=0)
            data[nseq].append({
                    "pdb": pdb_id + "_" + pid + "_" + nid, "affinity": aff,
                    "protein_seq": pseq, "protein_coords": pcoords,
                    "nucleic_seq": nseq, "nucleic_coords": ncoords,
            })

    final_data = []
    for k,vlist in data.items():
        L = [len(entry['protein_seq']) for entry in vlist]
        entry = vlist[np.argmax(L)]
        final_data.append(entry)

    if len(final_data) > 0:
        L = [len(entry['nucleic_seq']) for entry in final_data]
        return final_data[np.argmax(L)]
    else:
        print(pdb_id, 'empty', file=sys.stderr)
        return None


if __name__ == "__main__":
    data = []
    with open("PDBBind/index/INDEX_general_PN.2020") as f:
        for line in f:
            if line[0] == '#': continue
            pdb, _, _, aff = line.split()[:4]
            if '=' not in aff: continue  # indefinite affinity
            aff = aff.split('=')[-1]
            aff, met = math.log10(float(aff[:-2])), aff[-2:]
            if met == 'mM':
                aff -= 3
            elif met == 'uM':
                aff -= 6
            elif met == 'nM':
                aff -= 9
            elif met == 'pM':
                aff -= 12
            elif met == 'fM':
                aff -= 15
            else:
                raise ValueError()
            data.append((pdb, aff))

    with Pool(80) as pool:
        data = pool.map(process, data)
        data = [d for d in data if d is not None]

    print(len(data))
    with open("data.pkl", 'wb') as f:
        pickle.dump(data, f)
