import torch
import gemmi
from boltz.data import const


# Returns (N, 3) array of all atoms in sequence, and (N,) mask of modeled atoms
def load_cif(cif_file, seq, ca_only=False):
    structure = gemmi.read_structure(cif_file)
    structure.merge_chain_parts()
    structure.remove_waters()
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    structure.remove_empty_chains()
    
    if isinstance(seq, str) and seq.isalpha(): # string of one letter codes
        tricodes = [const.prot_letter_to_token[aa] for aa in seq]
    elif isinstance(seq, list) and isinstance(seq[0], str) and len(seq[0]==1): # list of one letter codes
        tricodes = [const.prot_letter_to_token[aa] for aa in seq]
    elif isinstance(seq, list) and isinstance(seq[0], str) and len(seq[0]==1): # list of three letter codes
        tricodes = seq
    elif isinstance(seq, list) and isinstance(seq[0], int): # list of indices
        tricodes = [const.tokens[aa] for aa in seq]
    else:
        raise ValueError("Sequence format not recognized")
    
    res_natoms = torch.tensor([len(const.ref_atoms[aa]) for aa in tricodes])
    res_starts = torch.nn.functional.pad(torch.cumsum(res_natoms, 0), (1, 0))[:-1]
    coords = torch.zeros((res_natoms.sum(), 3), dtype=torch.float)
    mask = torch.zeros(len(coords), dtype=torch.bool)

    chain_start=0
    prev_chain = structure[0][0]
    for chain in structure[0]:
        if chain.name[0] != prev_chain.name[0]:
            chain_start += len(prev_chain)

        for res in chain:
            if res.name not in const.tokens[2:22]: continue
            res_num = chain_start + res.label_seq
            if res.name != tricodes[res_num-1]:
                print(f"WARNING: {res.name} in structure does not match \
                      {tricodes[res_num-1]} in sequence at position {res_num}")
                continue
            for atom in res:
                if atom.name == "OXT": continue
                if ca_only and atom.name != "CA": continue
                atom_idx = res_starts[chain_start + res.label_seq - 1] 
                atom_idx += const.ref_atoms[res.name].index(atom.name)
                coords[atom_idx] = torch.tensor(list(atom.pos))
                mask[atom_idx] = True

        prev_chain = chain

    return coords, mask
