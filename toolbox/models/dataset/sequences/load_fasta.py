from pathlib import Path
from typing import List, Dict, Iterable

from Bio import SeqIO
from dask.delayed import Delayed, delayed


def handle_batch(sequences_dict: Dict, batch_ids: List[str]):
    return [
        sequences_dict[protein_id].seq
        for protein_id in batch_ids
    ]


def extract_sequences_from_fasta(file_path: Path, batched_ids: Iterable[List[str]]) -> List[Delayed]:
    record_dict = SeqIO.to_dict(SeqIO.parse(file_path, "fasta"))
    return [
        delayed(handle_batch)(record_dict, batch)
        for batch in batched_ids
    ]
