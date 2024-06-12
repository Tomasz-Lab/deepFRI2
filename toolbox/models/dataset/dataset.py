from enum import Enum
from typing import List
from pathlib import Path

from pydantic import BaseModel

from toolbox.models.dataset.handle_index import read_index
from toolbox.models.dataset.structures_dataset import StructuresDataset
from toolbox.models.protein.protein import Protein
from toolbox.models.proteinType import ProteinType


class Dataset(BaseModel):
    name: str
    proteins: List[Protein]
    type: ProteinType
    n_batches: int # number of batches max 1k-10k proteins and max 10k subfolders
    disto_location: Path
    structures_dataset: StructuresDataset

    @property
    def n_proteins(self) -> int:
        return len(self.proteins)

    @property
    def struct_location(self) -> Path:
        return self.structures_dataset.structures_path()

    def fetch_structures(self, overwrite: bool = False):
        pass

    def generate_distograms(self, overwrite: bool = False):
        pass

    def sequence_dict_to_fasta(self, new_fasta_file_name):
        sequences_index_path = self.structures_dataset.sequences_path() / "sequences.idx"

        index = read_index(sequences_index_path)

