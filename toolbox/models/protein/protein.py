from pathlib import Path
from typing import List, Union

from pydantic import BaseModel

from toolbox.models.distogram import Distogram
from toolbox.models.proteinType import ProteinType

class Sequence(BaseModel):
    amino_acids: str
    header: str # sequence metadata
    # TODO pdb and seqres amino_acids


class Structure(BaseModel):
    atom_coordinates: dict[str, tuple[float]]

class Metadata(BaseModel):
    origin: str
    proteinType: ProteinType
    location: Path

class Annotation(BaseModel):
    pass

class Protein(BaseModel):
    name: str
    sequence: Sequence
    structure: Structure
    metadata: Metadata
    annotation: Annotation
    distogram: Distogram

