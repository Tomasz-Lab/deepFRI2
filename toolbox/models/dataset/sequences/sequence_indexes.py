from functools import reduce
from pathlib import Path
from typing import List, Dict, Tuple, Iterable

from dask import delayed
from distributed import Client, progress

from toolbox.models.dataset.handle_index import read_index
SEPARATOR = "-"


def search_index(path: Path, ids_searched_for: List[str]):
    index = read_index(path)

    res = {}

    for id_ in ids_searched_for:
        if id_ in index:
            res[id_] = index[id_]

    return res


def search_sequence_indexes(
        db_type: str,
        base_path: Path,
        batched_ids: Iterable[List[str]]
) -> Tuple[Dict[str, str], List[str]]:
    sequence_indexes = base_path.glob(f'**/{db_type}{SEPARATOR}*/sequences.idx')

    tasks = [
        delayed(search_index)(file, batch)
        for batch in batched_ids
        for file in sequence_indexes
    ]

    with Client() as client:
        futures = client.compute(tasks)
        progress(futures)
        results = client.gather(futures)

        results = reduce(lambda a, b: {**a, **b}, results, {})
        return results, find_missing_ids(results, batched_ids)


def find_missing_ids(index: Dict[str, str], ids: Iterable[List[str]]) -> List[str]:
    return [
        id_
        for batch in ids
        for id_ in batch
        if id_ not in index
    ]
