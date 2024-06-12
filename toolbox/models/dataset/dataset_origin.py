import asyncio
import os

from foldcomp.setup import download
import dotenv
dotenv.load_dotenv()

pdir = os.getenv('DATA_PATH')
repo_path = pdir + "/repo"
dataset_path = pdir + "/dataset"

# class DatasetOriginBase:
#     pdir = "../../../data"
#
#     def fetch(self, *args, **kwargs):
#         raise NotImplementedError("Fetch method must be implemented by the specific dataset origin type.")
#
#
# class PDB(DatasetOriginBase):
#     def fetch(self, outdir):
#         pdb_list = PDBList()
#         ids = pdb_list.get_all_entries()
#         ids = ids[:10]
#         for pdb in ids:
#             pdb_list.retrieve_pdb_file(pdb, pdir=outdir)
#
#
# class PDBClust(DatasetOriginBase):
#     def fetch(self):
#         print("Fetching PDB_clust dataset...")
#
#
# class PDBSubset(DatasetOriginBase):
#     def fetch(self, ids, outdir: str):
#         pdir = self.pdir + outdir
#
#
# class AFDB(DatasetOriginBase):
#     def fetch(self):
#         print("Fetching AFDB dataset... (Warning: This is several TBs)")
#
#
# class AFDBSubset(DatasetOriginBase):
#     def fetch(self, subset_list):
#         print(f"Fetching AFDB_subset dataset for IDs: {subset_list}")
#
#
# class AF_clust_dark(DatasetOriginBase):
#     def fetch(self):
#         foldcomp_download('afdb_rep_dark_v4', self.pdir + "/afdark")
#
#
# class AF_clust_all(DatasetOriginBase):
#     def fetch(self):
#         foldcomp_download('afdb_rep_v4')
#
#
# class AFUniprot(DatasetOriginBase):
#     def fetch(self):
#         foldcomp_download('afdb_uniprot_v4', self.pdir + "/afuniprot")
#
#         foldcomp.setup()
#
#
# class AFSwissprot(DatasetOriginBase):
#     def fetch(self):
#         foldcomp_download('afdb_swissprot', self.pdir + "/afswissprot")
#
#
# class ESMAtlas(DatasetOriginBase):
#     def fetch(self):
#         foldcomp_download('esmatlas' + self.pdir + "/esmatlas")
#
#
# class ESMAtlasHClust30(DatasetOriginBase):
#     def fetch(self):
#         foldcomp_download('highquality_clust30', self.pdir + "/esmatlashclust30")
#
#
# class ESMAtlasSubset(DatasetOriginBase):
#     def fetch(self, subset_list):
#         print(f"Fetching ESMAtlas_subset dataset for IDs: {subset_list}")
#
#
# class DatasetOriginType(Enum):
#     PDB = PDB()
#     PDB_clust = PDBClust()
#     PDB_subset = PDBSubset()
#     AFDB = AFDB()
#     AFDB_subset = AFDBSubset()
#     AFDB_dark = AF_clust_dark()
#     AFDB_clust_all = AF_clust_all()
#     AFDB_uniprot = AFUniprot()
#     AFDB_swissprot = AFSwissprot()
#     ESMAtlas = ESMAtlas()
#     ESMAtlas_hclust30 = ESMAtlasHClust30()
#     ESMAtlas_subset = ESMAtlasSubset()
#
#
# class DatasetOriginType(Enum):
#     PDB = "PDB"
#     PDB_clust = "PDB_clust"


def foldcomp_download(db: str, output_dir: str):
    print(f"Foldcomp downloading db: {db} to {output_dir}")
    download_chunks = 16
    for i in ["", ".index", ".dbtype", ".lookup", ".source"]:
        asyncio.run(
            download(
                f"https://foldcomp.steineggerlab.workers.dev/{db}{i}",
                f"{output_dir}/{db}{i}",
                chunks=download_chunks,
            )
        )
