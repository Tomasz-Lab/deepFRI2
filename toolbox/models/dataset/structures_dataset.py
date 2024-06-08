import json
import pickle
from datetime import datetime
from enum import Enum
from itertools import islice
from operator import iconcat
from pathlib import Path
from typing import List

from functools import reduce

import foldcomp
from Bio import SeqIO
from Bio.PDB import PDBList, PDBParser
from Bio.SeqUtils import seq1
from dask import delayed, compute
from distributed import Client, progress
from pydantic import BaseModel

from toolbox.models.dataset.database_type import DatabaseType
from toolbox.models.dataset.dataset_origin import dataset_path, repo_path, foldcomp_download

SEPARATOR = "-"


# class Subset(BaseModel):
#     ids: List[str]


class CollectionType(Enum):
    all = "all"
    clust = "clust"
    part = "part"
    subset = "subset"


def mkdir_for_batches(base_path: Path, batch_count: int):
    for i in range(batch_count):
        (base_path / f"{i}").mkdir(exist_ok=True, parents=True)


def get_sequence_from_pdbs(file_paths: List[str], sequences_path: Path):
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


def get_sequence_from_pdb(path_str, sequences_path):
    pdb_parser = PDBParser()

    path = Path(path_str)

    structure = pdb_parser.get_structure("", path_str)
    chains = {chain.id: seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}

    protein_name = path.stem
    res = {protein_name: chains}

    # batch_number = int(path.parent.stem)
    # sequence_path = Path(sequences_path) / str(batch_number)
    # with open(sequence_path / f"{protein_name}_sequence.pickle", 'wb') as f:
    #     pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    return res

class StructuresDataset(BaseModel):
    db_type: DatabaseType
    collection_type: CollectionType
    type_str: str = ""
    version: str = datetime.now().strftime('%Y%m%d')
    ids_file: Path = None
    seqres_file: Path = None
    overwrite: bool = False
    batch_size: int = 10_000

    def dataset_repo_path(self):
        return Path(repo_path) / self.db_type.name / f"{self.collection_type.name}_{self.type_str}" / self.version

    def dataset_path(self):
        return Path(f"{dataset_path}/{self.dataset_index_file_name()}")

    def structures_path(self):
        return self.dataset_repo_path() / "structures"

    def dataset_index_file_name(self):
        return f"{self.db_type.name}{SEPARATOR}{self.collection_type.name}{SEPARATOR}{self.type_str}{SEPARATOR}{self.version}"

    def dataset_index_file_path(self) -> Path:
        return self.dataset_path() / "dataset.idx"

    def sequences_path(self):
        return self.dataset_path() / "sequences"

    def batches_count(self) -> int:
        return sum(1 for item in self.structures_path().iterdir() if item.is_dir())

    def create_dataset(self) -> "Dataset":

        self.dataset_repo_path().mkdir(exist_ok=True, parents=True)

        if self.collection_type == CollectionType.subset:
            assert self.ids_file is not None

        if self.collection_type is CollectionType.subset or self.collection_type is CollectionType.all:
            dataset_index_file_name = self.dataset_index_file_name()

            # find existing files of the same DB
            if self.db_type == DatabaseType.other:
                all_files = Path(repo_path).rglob("*.*")
            else:
                db_path = Path(repo_path) / self.db_type.name
                all_files = db_path.rglob("*.*")

            file_paths = {file_path.stem: file_path for file_path in all_files if
                          file_path.is_file() and file_path.suffix in {'.cif', '.pdb', '.ent'}}

            missing_ids = []

            ids = None
            if self.collection_type is CollectionType.subset:
                with open(self.ids_file, 'r') as f:
                    ids = f.read().splitlines()
            else:
                ids = self.get_all_ids()

            Path(f"{dataset_path}/{dataset_index_file_name}").mkdir(exist_ok=True, parents=True)
            with open(self.dataset_index_file_path(), "a") as index_file:
                for id_ in ids:
                    if id_ in file_paths:
                        index_file.write(str(file_paths[id_]) + '\n')
                    else:
                        missing_ids.append(id_)

            print(len(missing_ids))

            if self.db_type == DatabaseType.other and self.collection_type == CollectionType.subset and len(
                    missing_ids) > 0:
                print(missing_ids)
                raise RuntimeError("Missing ids are not allowed when subsetting all DBs!")

            self.download_ids(missing_ids)
        else:
            if self.overwrite:
                print("Overwriting ")
                # TODO
                # find previous version of the same db_type and type
                # remove previous
                # download new one
            else:
                self.download_ids(None)

        self.save_dataset_metadata()

        self.generate_sequence()

    def generate_sequence(self):
        with self.dataset_index_file_path().open("r") as ids_file:
            all_ids = ids_file.readlines()
            batched_ids = list(self.chunk(map(lambda l: l.rstrip('\n'), all_ids)))

            # self.sequences_path().mkdir(exist_ok=True, parents=True)
            # mkdir_for_batches(self.sequences_path(), self.batches_count())
            # unused when only one result file

            tasks = [
                delayed(get_sequence_from_pdb)(ids)
                for ids in batched_ids
            ]
            with Client() as client:
                futures = client.compute(tasks)
                progress(futures)
                results = client.gather(futures)

                results = reduce(iconcat, results, [])

                results = reduce(lambda a, b: {**a, **b}, results)
                with open(self.dataset_path() / "pdb_sequence.pickle", 'wb') as f:
                    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    def get_all_ids(self):
        res = None
        match self.db_type:
            case DatabaseType.PDB:
                res = PDBList().get_all_entries()
            case DatabaseType.AFDB:
                res = []
            case DatabaseType.ESMatlas:
                res = []
            case _:
                res = open(self.ids_file).readlines()
        return res

    def download_ids(self, ids):
        print("Downloading ids")
        match self.db_type:
            case DatabaseType.PDB:
                self.handle_pdb(ids)
            case DatabaseType.AFDB:
                self.handle_afdb()
            case DatabaseType.ESMatlas:
                self.handle_esma()
            case DatabaseType.other:
                self

    def handle_pdb(self, ids: List[str]):
        match self.collection_type:
            case CollectionType.all:
                self._download_pdb_(ids)
            case CollectionType.part:
                pass
            case CollectionType.clust:
                pass
            case CollectionType.subset:
                self._download_pdb_(ids)

    def retrieve_pdb_file(self, pdb: str, pdb_repo_path: str):
        pdb_list = PDBList()
        # Ensure the directory string is properly passed

        # PDB ids from PDBList() are upper case without 'pdb' prefix
        pdb = pdb.removeprefix("pdb")
        pdb = pdb.upper()

        pdb_list.retrieve_pdb_file(
            pdb,
            pdir=pdb_repo_path,
            file_format="pdb"
        )

    def save_new_files_to_index(self):
        self.dataset_path().mkdir(exist_ok=True, parents=True)

        with (self.dataset_path() / "dataset.idx").open("a") as index_file:
            structures_path = self.structures_path()
            files = [f for f in structures_path.rglob('*.*') if f.is_file()]
            for file_path in files:
                index_file.write(f"{file_path}\n")

    def _download_pdb_(self, ids: List[str]):
        Path(self.structures_path()).mkdir(exist_ok=True, parents=True)
        pdb_repo_path = self.structures_path()
        chunks = list(self.chunk(ids))

        mkdir_for_batches(pdb_repo_path, len(chunks))

        tasks = [
            delayed(self.retrieve_pdb_file)(pdb, pdb_repo_path / f"{batch_number}")
            for batch_number, pdb_ids_chunk in enumerate(chunks)
            for pdb in pdb_ids_chunk
        ]
        # Dask distributed client
        with Client() as client:
            # Submit tasks to the client
            futures = client.compute(tasks)
            # Display progress of tasks
            progress(futures)
            # Wait for all tasks to complete
            results = client.gather(futures)

        self.save_new_files_to_index()

    def foldcomp_decompress(self):

        db_path = self.dataset_repo_path() / self.type_str

        lookup_path = self.dataset_repo_path() / f"{self.type_str}.lookup"

        ids_lines = lookup_path.open().readlines()
        ids_lines = filter(
            lambda line: ".pdb" in line if (".pdb" in line or ".cif" in line) else True, ids_lines
        )
        ids = list(map(lambda line: line.split()[1], ids_lines))

        structures_path = self.structures_path()
        structures_path.mkdir(exist_ok=True, parents=True)

        def process_chunk(batch_number: int, ids: List[str]):
            # Assuming the capability to open and handle slices of `db_type`
            # `db_type` must support slicing or an equivalent method to fetch a range of items
            with foldcomp.open(db_path, ids=ids) as db:
                for (_, pdb), file_name in zip(db, ids):
                    if ".pdb" not in file_name:
                        file_name = f"{file_name}.pdb"
                    file_path = structures_path / f"{batch_number}" / f"{file_name}"
                    with open(file_path, 'w') as f:
                        f.write(pdb)

        with Client() as client:
            cpu_cores = len(client.ncores())
            structures_path = self.structures_path()
            structures_path.mkdir(exist_ok=True, parents=True)

            batches = list(self.chunk(ids))
            mkdir_for_batches(structures_path, len(batches))

            futures = []
            for number, batch in enumerate(batches):
                future = client.submit(process_chunk, number, list(batch))
                futures.append(future)

            progress(futures)
            results = client.gather(futures)

        self.save_new_files_to_index()

    def handle_afdb(self):
        match self.collection_type:
            case CollectionType.all:
                pass
            case CollectionType.part:
                foldcomp_download(self.type_str, str(self.dataset_repo_path()))
                self.foldcomp_decompress()
            case CollectionType.clust:
                foldcomp_download(self.type_str, str(self.dataset_repo_path()))
                self.foldcomp_decompress()
            case CollectionType.subset:
                pass

    def handle_esma(self):
        match self.collection_type:
            case CollectionType.all:
                pass
            case CollectionType.part:
                pass
            case CollectionType.clust:
                foldcomp_download(self.type_str, str(self.dataset_repo_path()))
                self.foldcomp_decompress()
            case CollectionType.subset:
                pass

    def chunk(self, it):
        it = iter(it)
        return iter(lambda: tuple(islice(it, self.batch_size)), ())

    def save_dataset_metadata(self):
        with (self.dataset_path() / "dataset.json").open("w+") as json_dataset_file:
            json_dataset_file.write(self.model_dump_json())


def create_subset():
    StructuresDataset(
        db_type=DatabaseType.other,
        collection_type=CollectionType.subset,
        ids_file=Path("./mix_ids1.txt")
    ).create_dataset()


def create_e_coli():
    StructuresDataset(
        db_type=DatabaseType.AFDB,
        collection_type=CollectionType.part,
        type_str="e_coli"
    ).create_dataset()


def create_swissprot():
    StructuresDataset(
        db_type=DatabaseType.AFDB,
        collection_type=CollectionType.part,
        type_str="afdb_swissprot_v4"
    ).create_dataset()


def test():
    chunk = lambda xd: iter(lambda: tuple(islice(iter(xd), 10_000)), ())

    with Path(
            "/Users/youngdashu/sano/deepFRI2-toolbox-dev/data/dataset/AFDB-part-afdb_swissprot_v4-20240604/dataset.idx").open(
        "r") as ids_file:
        all_ids = ids_file.readlines()
        # batched_ids = list(map(lambda l: l.rstrip('\n'), all_ids))

        batched_ids = list(chunk(map(lambda l: l.rstrip('\n'), all_ids)))
        batched_ids = batched_ids[:10]

        sequences_path = Path(
            "/Users/youngdashu/sano/deepFRI2-toolbox-dev/data/dataset/AFDB-part-afdb_swissprot_v4-20240604/sequences")

        # sequences_path.mkdir(exist_ok=True, parents=True)
        # mkdir_for_batches(sequences_path, 55)

        tasks = [
            delayed(get_sequence_from_pdbs)(ids, sequences_path)
            for ids in batched_ids
        ]
        with Client() as client:
            futures = client.compute(tasks)
            progress(futures)
            results = client.gather(futures)

            results = reduce(iconcat, results, [])

            results = reduce(lambda a, b: {**a, **b}, results, {})

            with open(
                    Path(
                        "/Users/youngdashu/sano/deepFRI2-toolbox-dev/data/dataset/AFDB-part-afdb_swissprot_v4-20240604") / "pdb_sequence.pickle",
                    'wb') as f:
                pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    # import pickle
    # import bz2
    #
    # with open("/Users/youngdashu/sano/deepFRI2-toolbox-dev/data/dataset/AFDB-part-afdb_swissprot_v4-20240604/pdb_sequence.pickle", 'rb') as f:
    #     results = pickle.load(f)
    #
    #
    #     with open(
    #             Path(
    #                 "/Users/youngdashu/sano/deepFRI2-toolbox-dev/data/dataset/AFDB-part-afdb_swissprot_v4-20240604") / "compressed_sequence.data",
    #             'wb') as f:
    #         compressed_data = bz2.compress(json.dumps(results).encode())
    #         f.write(compressed_data)

    test()
    pass