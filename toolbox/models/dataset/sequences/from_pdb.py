from pathlib import Path
from typing import List

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


def get_sequence_from_pdbs(file_paths: List[str]):
    pdb_parser = PDBParser()

    res = []
    for path in file_paths:
        structure = pdb_parser.get_structure("", path)
        chains = {chain.id: seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}

        path = Path(path)
        protein_name = path.stem

        # batch_number = int(path.parent.stem)
        # sequence_path = Path(sequences_path) / str(batch_number)
        # with open(sequence_path / f"{protein_name}_sequence.pickle", 'wb') as f:
        #     pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

        res.append({protein_name: chains})

    return res

    # return [{record.id: record.seq for record in SeqIO.parse(path, 'pdb-seqres')} for path in file_paths]