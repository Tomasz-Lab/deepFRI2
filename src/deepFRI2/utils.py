"""Shared utilities for deepFRI2 inference.

This module holds everything the inference pipeline needs that is not part of the
model definitions (see ``model.py``):

- residue/atom lookup tables (``substitutions``, ``aa_dict``, ``unwanted_residues``),
- logging helpers (``logger`` + console/file handler management),
- tensor preprocessing (``process_distogram``, ``pad_embedding``, ``pad_distogram``),
- structure parsing (``parse_cif_file`` and friends, for both ``.cif`` and ``.pdb``),
- distogram / ESM-embedding generation,
- batch preparation and per-batch inference.

The functions are deliberately free of hidden global state: the ESM tokenizer,
model, device and the model/data constants (``MAX_SEQ_LEN``, ``ESM_DIM``,
``SIGMA_DIST``, atom name) are passed in explicitly by the caller (the notebook /
inference script owns that configuration).
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from logging import INFO, getLogger
from pathlib import Path

import fastobo
import numpy as np
import pandas as pd
import torch
import tqdm
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from scipy.spatial.distance import pdist, squareform


# ========
# Mappings
# ========

# https://github.com/openmm/pdbfixer/blob/master/pdbfixer/pdbfixer.py
substitutions = {
    "2AS": "ASP",
    "3AH": "HIS",
    "5HP": "GLU",
    "ACL": "ARG",
    "AGM": "ARG",
    "AIB": "ALA",
    "ALM": "ALA",
    "ALO": "THR",
    "ALY": "LYS",
    "ARM": "ARG",
    "ASA": "ASP",
    "ASB": "ASP",
    "ASK": "ASP",
    "ASL": "ASP",
    "ASQ": "ASP",
    "AYA": "ALA",
    "BCS": "CYS",
    "BHD": "ASP",
    "BMT": "THR",
    "BNN": "ALA",
    "BUC": "CYS",
    "BUG": "LEU",
    "C5C": "CYS",
    "C6C": "CYS",
    "CAS": "CYS",
    "CCS": "CYS",
    "CEA": "CYS",
    "CGU": "GLU",
    "CHG": "ALA",
    "CLE": "LEU",
    "CME": "CYS",
    "CSD": "ALA",
    "CSO": "CYS",
    "CSP": "CYS",
    "CSS": "CYS",
    "CSW": "CYS",
    "CSX": "CYS",
    "CXM": "MET",
    "CY1": "CYS",
    "CY3": "CYS",
    "CYG": "CYS",
    "CYM": "CYS",
    "CYQ": "CYS",
    "DAH": "PHE",
    "DAL": "ALA",
    "DAR": "ARG",
    "DAS": "ASP",
    "DCY": "CYS",
    "DGL": "GLU",
    "DGN": "GLN",
    "DHA": "ALA",
    "DHI": "HIS",
    "DIL": "ILE",
    "DIV": "VAL",
    "DLE": "LEU",
    "DLY": "LYS",
    "DNP": "ALA",
    "DPN": "PHE",
    "DPR": "PRO",
    "DSN": "SER",
    "DSP": "ASP",
    "DTH": "THR",
    "DTR": "TRP",
    "DTY": "TYR",
    "DVA": "VAL",
    "EFC": "CYS",
    "FLA": "ALA",
    "FME": "MET",
    "GGL": "GLU",
    "GL3": "GLY",
    "GLZ": "GLY",
    "GMA": "GLU",
    "GSC": "GLY",
    "HAC": "ALA",
    "HAR": "ARG",
    "HIC": "HIS",
    "HIP": "HIS",
    "HMR": "ARG",
    "HPQ": "PHE",
    "HTR": "TRP",
    "HYP": "PRO",
    "IAS": "ASP",
    "IIL": "ILE",
    "IYR": "TYR",
    "KCX": "LYS",
    "LLP": "LYS",
    "LLY": "LYS",
    "LTR": "TRP",
    "LYM": "LYS",
    "LYZ": "LYS",
    "MAA": "ALA",
    "MEN": "ASN",
    "MHS": "HIS",
    "MIS": "SER",
    "MLE": "LEU",
    "MPQ": "GLY",
    "MSA": "GLY",
    "MSE": "MET",
    "MVA": "VAL",
    "NEM": "HIS",
    "NEP": "HIS",
    "NLE": "LEU",
    "NLN": "LEU",
    "NLP": "LEU",
    "NMC": "GLY",
    "OAS": "SER",
    "OCS": "CYS",
    "OMT": "MET",
    "PAQ": "TYR",
    "PCA": "GLU",
    "PEC": "CYS",
    "PHI": "PHE",
    "PHL": "PHE",
    "PR3": "CYS",
    "PRR": "ALA",
    "PTR": "TYR",
    "PYX": "CYS",
    "SAC": "SER",
    "SAR": "GLY",
    "SCH": "CYS",
    "SCS": "CYS",
    "SCY": "CYS",
    "SEL": "SER",
    "SEP": "SER",
    "SET": "SER",
    "SHC": "CYS",
    "SHR": "LYS",
    "SMC": "CYS",
    "SOC": "CYS",
    "STY": "TYR",
    "SVA": "SER",
    "TIH": "ALA",
    "TPL": "TRP",
    "TPO": "THR",
    "TPQ": "ALA",
    "TRG": "LYS",
    "TRO": "TRP",
    "TYB": "TYR",
    "TYI": "TYR",
    "TYQ": "TYR",
    "TYS": "TYR",
    "TYY": "TYR",
    # additional added by us:
    "SEC": "CYS",
}

rnaResidues = ["A", "G", "C", "U", "I"]
dnaResidues = ["DA", "DG", "DC", "DT", "DI"]

unwanted_residues = rnaResidues + dnaResidues

aa_dict = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "UNK": "X",
}


# =======
# Logging
# =======

logger = getLogger("inference")
_console_handlers = []


def set_console_logging(enabled=True):
    root_logger = logging.getLogger()
    global _console_handlers
    if not enabled:
        _console_handlers = [
            handler
            for handler in list(root_logger.handlers)
            if isinstance(handler, logging.StreamHandler)
            and not getattr(handler, "_deepfri_log_file", False)
        ]
        for handler in _console_handlers:
            root_logger.removeHandler(handler)
    elif _console_handlers:
        for handler in _console_handlers:
            if handler not in root_logger.handlers:
                root_logger.addHandler(handler)
        _console_handlers = []


def configure_log_file(log_path):
    root_logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    for handler in list(root_logger.handlers):
        if getattr(handler, "_deepfri_log_file", False):
            root_logger.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(INFO)
    file_handler.setFormatter(formatter)
    file_handler._deepfri_log_file = True
    root_logger.addHandler(file_handler)
    return log_path


def log_file_only(message, level=INFO):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if getattr(handler, "_deepfri_log_file", False) and level >= handler.level:
            record = root_logger.makeRecord(
                root_logger.name,
                level,
                fn="",
                lno=0,
                msg=message,
                args=(),
                exc_info=None,
            )
            handler.handle(record)


def log_timing(step, elapsed_s, processed=None, unit=None):
    processed_str = ""
    if processed is not None and unit is not None:
        plural = {"batch": "batches", "ontology-batch": "ontology-batches"}
        suffix = unit if processed == 1 else plural.get(unit, f"{unit}s")
        processed_str = f" | {processed:>5} {suffix}"
    logger.info(f"{step:<28} | {elapsed_s:>8.3f}s{processed_str}")


# ====================
# Tensor preprocessing
# ====================

def process_distogram(distogram, sigma_dist):
    sigma = sigma_dist  # or e.g., np.nanmedian(dist_matrix)
    # Gaussian kernel similarity transformation
    sim_matrix = torch.exp(-torch.square(distogram) / (2 * sigma**2))
    # Handle missing or infinite values
    sim_matrix.masked_fill_(torch.isinf(distogram) | torch.isnan(distogram), 0)
    return sim_matrix


def pad_embedding(embedding: torch.Tensor, max_len: int, emb_size: int) -> torch.Tensor:
    padded = torch.zeros(max_len, emb_size)
    seq_len = min(embedding.shape[0], max_len)
    padded[:seq_len, :].copy_(embedding[:seq_len, :])
    return padded


def pad_distogram(distogram: torch.Tensor, max_len: int) -> torch.Tensor:
    padded = torch.full((max_len, max_len), float("inf"))
    seq_len = min(distogram.shape[0], max_len)
    padded[:seq_len, :seq_len].copy_(distogram[:seq_len, :seq_len])
    return padded


# ===============================
# Structure parsing (.cif / .pdb)
# ===============================

def _as_list(value):
    return value if isinstance(value, list) else [value]


def _clean_seqres(seq):
    return "".join(seq.split())


def _aa1(code):
    if len(code) == 1:
        return code
    code = substitutions.get(code, code)
    return aa_dict.get(code, "X")


def _format_positions(positions, max_ranges=10):
    positions = sorted(set(positions))
    if not positions:
        return ""
    ranges = []
    start = prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
        else:
            ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = pos
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    if len(ranges) > max_ranges:
        return f"{','.join(ranges[:max_ranges])},... [{len(positions)} pos]"
    suffix = f" [{len(positions)} pos]"
    return f"{','.join(ranges)}{suffix}"


def _seqres_from_pdb_tokens(tokens):
    return "".join(_aa1(substitutions.get(token, token)) for token in tokens) if tokens else None


def _seqres_from_cif_dict(cif):
    for key in ("_entity_poly.pdbx_seq_one_letter_code_can", "_entity_poly.pdbx_seq_one_letter_code"):
        raw = cif.get(key)
        if raw is not None:
            entries = _as_list(raw)
            assert len(entries) == 1, (
                "Multi-chain/model CIF/PDB files are not supported. Please split it into parts."
            )
            seq = _clean_seqres(entries[0])
            if seq:
                return seq
    mon_ids = cif.get("_entity_poly_seq.mon_id")
    if mon_ids is not None:
        return "".join(_aa1(substitutions.get(mon_id, mon_id)) for mon_id in _as_list(mon_ids))
    return None


def _atoms_from_pdb(pdb_file, atom_name):
    coords_by_id = {}
    aa_by_id = {}
    chain = None
    seqres_tokens = []

    with open(pdb_file, "r") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "SEQRES":
                # PDB SEQRES residue names occupy columns 20-22, 24-26, ... (1-based), i.e.
                # 0-based index 19 onward in 4-char steps (3-char name + 1 space).
                for i in range(19, min(len(line), 70), 4):
                    name = line[i : i + 3].strip()
                    if name:
                        seqres_tokens.append(name)
                continue
            if record != "ATOM":
                continue
            if line[16].strip() not in ("", "A"):
                continue
            residue = line[17:20].strip()
            if residue in unwanted_residues:
                continue
            atom = line[12:16].strip()
            residue_id = int(line[22:26])
            chain = chain or (line[21].strip() or "A")
            if residue_id not in aa_by_id:
                aa_by_id[residue_id] = _aa1(residue)
            if atom == atom_name and residue_id not in coords_by_id:
                coords_by_id[residue_id] = (
                    np.float32(line[30:38]),
                    np.float32(line[38:46]),
                    np.float32(line[46:54]),
                )

    if not coords_by_id:
        raise ValueError(f"No {atom_name} atoms found in {Path(pdb_file).name}")
    return coords_by_id, aa_by_id, chain, _seqres_from_pdb_tokens(seqres_tokens)


def _atoms_from_cif(cif_file, atom_name):
    cif = MMCIF2Dict(str(cif_file))
    atom_ids = _as_list(cif.get("_atom_site.auth_atom_id", cif["_atom_site.label_atom_id"]))
    residues = _as_list(cif.get("_atom_site.auth_comp_id", cif["_atom_site.label_comp_id"]))
    chains = _as_list(cif.get("_atom_site.auth_asym_id", cif["_atom_site.label_asym_id"]))
    residue_ids = _as_list(cif.get("_atom_site.auth_seq_id", cif["_atom_site.label_seq_id"]))
    xs = _as_list(cif["_atom_site.Cartn_x"])
    ys = _as_list(cif["_atom_site.Cartn_y"])
    zs = _as_list(cif["_atom_site.Cartn_z"])
    groups = _as_list(cif.get("_atom_site.group_PDB", ["ATOM"] * len(atom_ids)))

    coords_by_id = {}
    aa_by_id = {}
    chain = None
    for group, atom, residue, chain_id, residue_id, x, y, z in zip(
        groups, atom_ids, residues, chains, residue_ids, xs, ys, zs
    ):
        if group != "ATOM" or residue in unwanted_residues:
            continue
        residue_id = int(residue_id)
        chain = chain or str(chain_id)
        if residue_id not in aa_by_id:
            aa_by_id[residue_id] = _aa1(residue)
        if atom == atom_name and residue_id not in coords_by_id:
            coords_by_id[residue_id] = (np.float32(x), np.float32(y), np.float32(z))

    if not coords_by_id:
        raise ValueError(f"No {atom_name} atoms found in {Path(cif_file).name}")

    strand_entries = _as_list(cif.get("_entity_poly.pdbx_strand_id", []))
    strands = [part.strip() for entry in strand_entries for part in str(entry).split(",") if part.strip()]
    if strands:
        assert len(strands) == 1, (
            "Multi-chain/model CIF/PDB files are not supported. Please split it into parts."
        )
    return coords_by_id, aa_by_id, chain, _seqres_from_cif_dict(cif)


def _finalize_structure(coords_by_id, aa_by_id, chain, seqres, protein_id, atom_name):
    residue_ids = set(coords_by_id) | set(aa_by_id)
    max_residue_id = max(residue_ids)

    if seqres is not None:
        seq_len = len(seqres)
        sequence = seqres
    else:
        seq_len = max_residue_id
        sequence = "".join(aa_by_id.get(residue_id, "X") for residue_id in range(1, seq_len + 1))

    coords = np.full((seq_len, 3), np.nan, dtype=np.float32)
    for residue_id, (x, y, z) in coords_by_id.items():
        if 1 <= residue_id <= seq_len:
            coords[residue_id - 1] = (x, y, z)

    span = range(1, seq_len + 1)
    missing_dist = [residue_id for residue_id in span if residue_id not in coords_by_id]
    if missing_dist:
        log_file_only(
            f"{protein_id}: missing {atom_name} coords at {_format_positions(missing_dist)}",
            level=logging.WARNING,
        )

    if seqres is not None:
        mismatches = [
            residue_id
            for residue_id in span
            if residue_id in aa_by_id and aa_by_id[residue_id] != seqres[residue_id - 1]
        ]
        if mismatches:
            examples = ", ".join(
                f"{pos}:{seqres[pos - 1]}/{aa_by_id[pos]}"
                for pos in mismatches[:3]
            )
            suffix = f" e.g. {examples}" if examples else ""
            log_file_only(
                f"{protein_id}: SEQRES/ATOM mismatch at {_format_positions(mismatches)}{suffix}",
                level=logging.WARNING,
            )
        extra = sorted(residue_id for residue_id in coords_by_id if residue_id < 1 or residue_id > seq_len)
        if extra:
            log_file_only(
                f"{protein_id}: ignoring {atom_name} outside SEQRES at {_format_positions(extra)}",
                level=logging.WARNING,
            )
    else:
        missing_aa = [residue_id for residue_id in span if residue_id not in aa_by_id]
        if missing_aa:
            log_file_only(
                f"{protein_id}: no ATOM for residue name at {_format_positions(missing_aa)} (using X)",
                level=logging.WARNING,
            )

    residue_df = pd.DataFrame(
        {
            "residue_id": np.arange(1, seq_len + 1, dtype=np.int32),
            "residue_aa": list(sequence),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "chain": chain,
        }
    )
    return sequence, coords, residue_df


def _list_structure_files(struct_path):
    struct_path = Path(struct_path)
    return sorted(list(struct_path.glob("*.cif")) + list(struct_path.glob("*.pdb")))


def _dedupe_structure_paths(paths):
    """Keep one file per protein id (stem), preferring .cif over .pdb.

    An explicit ``.pdb`` selection is unaffected upstream: only paths already selected
    reach this step, so if a stem has only its ``.pdb`` here, the ``.pdb`` is kept.
    """
    deduped = {}
    for p in paths:
        if p.stem not in deduped or p.suffix.lower() == ".cif":
            deduped[p.stem] = p
    return [deduped[stem] for stem in sorted(deduped)]


def parse_cif_file(cif_file, atom_name):
    cif_file = Path(cif_file)
    if cif_file.suffix.lower() == ".pdb":
        coords_by_id, aa_by_id, chain, seqres = _atoms_from_pdb(cif_file, atom_name)
    else:
        coords_by_id, aa_by_id, chain, seqres = _atoms_from_cif(cif_file, atom_name)
    sequence, coords, residue_df = _finalize_structure(
        coords_by_id, aa_by_id, chain, seqres, cif_file.stem, atom_name
    )
    return cif_file.stem, sequence, coords, residue_df


def try_parse_cif_file(cif_file, atom_name):
    try:
        return parse_cif_file(cif_file, atom_name)
    except UnicodeDecodeError as exc:
        log_file_only(f"Skipping broken structure file {Path(cif_file).name}: {exc}", level=logging.WARNING)
        return None


def extract_coordinates_from_cif(cif_file, atom_name):
    _, _, _, residue_df = parse_cif_file(cif_file, atom_name)
    return residue_df


# ===========================
# Distograms & ESM embeddings
# ===========================

def generate_distograms_for_list(coords, show_progress=True, log_start=True):
    """
    coords: list of per-protein coordinate arrays, shape (n_residues, 3).

    Returns distograms in the same order as `coords`. Serial loop only; revisit
    parallelization when batch processing is in place.
    """
    distograms = []
    if log_start:
        logger.info("Generating distograms...")
    iterator = tqdm.tqdm(coords, disable=not show_progress)
    for coord in iterator:
        distances = pdist(coord, metric="euclidean")
        distances = distances.astype(np.float32)
        distogram = squareform(distances).astype(np.float32)
        np.fill_diagonal(distogram, 0.0)
        distograms.append(distogram)
    return distograms


def generate_embeddings_for_list(sequences, tokenizer, model, device, show_progress=True, log_start=True):
    """
    sequences: list of sequences, e.g. ["NIGIVSGDVTTL", "MEANK", ...].
    tokenizer / model / device: the ESM-2 tokenizer, model and torch device.
    Returns a list of embeddings, one per sequence (same order).
    """
    embeddings = []
    with torch.inference_mode():
        if log_start:
            logger.info("Generating embeddings...")
        iterator = tqdm.tqdm(sequences, disable=not show_progress)
        for seq in iterator:
            input_ = tokenizer(seq, return_tensors="pt")
            input_ = {k: v.to(device) for k, v in input_.items()}
            with torch.inference_mode():
                output = model(**input_, output_hidden_states=True)
            embedding = output.hidden_states[-1]
            embedding = embedding[0, :].to("cpu").to(torch.float32).numpy()
            embeddings.append(embedding)
            del input_, output
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return embeddings


def generate_embeddings_for_batch(sequences, tokenizer, model, device, max_seq_len):
    """
    sequences: list of sequences, e.g. ["NIGIVSGDVTTL", "MEANK", ...].
    tokenizer / model / device: the ESM-2 tokenizer, model and torch device.
    Returns padded embeddings.
    """
    with torch.inference_mode():
        logger.info("Generating embeddings...")
        input_ = tokenizer(sequences, return_tensors="pt",
                           padding="max_length", truncation=True,
                           max_length=max_seq_len)
        input_ = {k: v.to(device) for k, v in input_.items()}
        output = model(**input_, output_hidden_states=True)
        output = output.hidden_states[-1]  # (B, max_tok_len, 1280)
        embeddings = output.to('cpu', dtype=torch.float32).numpy()
        del input_, output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return embeddings


# =============================
# Batch preparation & inference
# =============================

def fill_gap_offdiagonals(sim, valid_len, eps=1e-6):
    """Restore the backbone-adjacency line that missing residues zero out in the similarity matrix.

    A residue with missing coordinates has its whole distogram row/column set to 0, which breaks
    the +/-1 off-diagonal (the "consecutive residues are neighbours" line, normally ~0.93). That
    zero-cross is out of distribution for the structural prober, which was trained on gapless
    structures. As a rough repair we set only the immediate +/-1 off-diagonals to 1 wherever a gap
    zeroed them out (resolved neighbours are well above ``eps``, so they are left untouched). We do
    not fill farther off-diagonals: their true similarity decays with distance, so forcing 1 there
    would inject a stronger artificial bias. Operates in place over the valid residue range.
    """
    if valid_len <= 1:
        return sim
    i = torch.arange(valid_len - 1)
    gap = sim[i, i + 1] < eps  # gap-induced zeros only; resolved neighbours are ~0.93
    gi = i[gap]
    sim[gi, gi + 1] = 1.0
    sim[gi + 1, gi] = 1.0
    return sim


def preprocess_data(embedding, distogram, max_seq_len, emb_size, sigma_dist) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embedding = torch.as_tensor(embedding, dtype=torch.float32) if embedding is not None else None
    distogram = torch.as_tensor(distogram, dtype=torch.float32) if distogram is not None else None

    if embedding is None and distogram is None:
        raise ValueError("At least one of embedding or distogram must be provided.")

    # Match `DeepFRIDataset`: keep <cls>, drop <eof>, and build the mask in residue space.
    if embedding is not None and embedding.shape[0] > 0:
        embedding = embedding[:-1]

    embedding_seq_len = embedding.shape[0] if embedding is not None else None
    distogram_seq_len = distogram.shape[0] if distogram is not None else None
    mask_seq_len = (
        distogram_seq_len
        if distogram_seq_len is not None
        else max(0, embedding_seq_len - 1)
    )
    valid_len = min(mask_seq_len, max_seq_len)

    padded_embedding = (
        pad_embedding(embedding, max_seq_len, emb_size)
        if embedding is not None
        else torch.zeros(max_seq_len, emb_size, dtype=torch.float32)
    )

    if distogram is not None:
        padded_distogram = pad_distogram(distogram, max_seq_len)
        distogram_processed = process_distogram(padded_distogram, sigma_dist)
        # Repair the +/-1 backbone band where missing residues zeroed it out (structural gaps).
        fill_gap_offdiagonals(distogram_processed, valid_len)
    else:
        distogram_processed = torch.zeros(max_seq_len, max_seq_len, dtype=torch.float32)

    mask = torch.zeros(max_seq_len, dtype=torch.float32)
    mask[:valid_len] = 1

    return padded_embedding, distogram_processed, mask


def prepare_batches_for_inference(
    struct_path,
    tokenizer,
    model,
    device,
    atom_name,
    max_seq_len,
    emb_size,
    sigma_dist,
    batch_size=32,
    file_names=None,
    preprocessing=True,
    return_struct_info=False,
):
    """Parse structures, compute distograms + ESM embeddings, and yield batches.

    Args:
        struct_path: directory of ``.cif`` / ``.pdb`` structure files.
        tokenizer / model / device: ESM-2 tokenizer, model and torch device.
        atom_name: atom used for distances (e.g. ``"CA"``).
        max_seq_len / emb_size / sigma_dist: preprocessing constants (see ``preprocess_data``).
        batch_size: number of proteins per yielded batch.
        file_names: optional explicit list of structures to run, each a path relative to
            ``struct_path`` (bare name or subfolder path) or an absolute path; non-existent entries
            are skipped. When None, every top-level structure in ``struct_path`` is used.
        preprocessing: if True, pad/process into stacked tensors; otherwise yield raw arrays.
        return_struct_info: if True, also yield per-protein structure metadata.

    Yields:
        ``(batch_ids, embeddings, distograms, masks)`` tuples (plus struct info if requested).
    """
    logger.info("Preparing inputs for inference...")

    struct_path = Path(struct_path)

    # With an explicit selection, resolve each entry against struct_path (absolute entries are
    # used as-is), keeping only those that exist; supports bare names and relative subfolder paths.
    # Otherwise scan the top level of struct_path.
    if file_names is not None:
        paths = []
        for name in file_names:
            p = Path(name)
            full = p if p.is_absolute() else struct_path / p
            if full.is_file():
                paths.append(full)
    else:
        paths = _list_structure_files(struct_path)

    # Keep one file per protein id, preferring .cif over .pdb (see _dedupe_structure_paths).
    paths = _dedupe_structure_paths(paths)

    max_workers = min(len(paths), max(1, min(8, os.cpu_count() or 1)))
    skipped_total = 0
    parsed_total = 0
    yielded_batches = 0
    parse_elapsed = 0.0
    distogram_elapsed = 0.0
    embedding_elapsed = 0.0
    batching_elapsed = 0.0

    for chunk_idx, chunk_start in enumerate(range(0, len(paths), batch_size), start=1):
        path_chunk = paths[chunk_start : chunk_start + batch_size]

        parse_start = time.perf_counter()
        if max_workers <= 1:
            parsed_chunk = [try_parse_cif_file(filename, atom_name) for filename in path_chunk]
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(path_chunk))) as pool:
                parsed_chunk = list(pool.map(lambda filename: try_parse_cif_file(filename, atom_name), path_chunk))
        parse_batch_elapsed = time.perf_counter() - parse_start
        parse_elapsed += parse_batch_elapsed

        parsed_items = []
        for structure_file, parsed in zip(path_chunk, parsed_chunk):
            if parsed is None:
                skipped_total += 1
                continue
            protein_id, sequence, coords, residue_df = parsed
            parsed_items.append(
                {
                    'protein_id': protein_id,
                    'structure_file': structure_file,
                    'residue_df': residue_df,
                    'sequence': sequence,
                    'coords': coords,
                }
            )

        if not parsed_items:
            log_timing(f"Parsing batch {chunk_idx:>3}", parse_batch_elapsed, 0, "file")
            continue

        parsed_total += len(parsed_items)
        log_timing(f"Parsing batch {chunk_idx:>3}", parse_batch_elapsed, len(parsed_items), "file")
        coords = [item['coords'] for item in parsed_items]
        sequences_in_order = [item['sequence'] for item in parsed_items]

        distogram_start = time.perf_counter()
        distograms = generate_distograms_for_list(coords, show_progress=False, log_start=False)
        distogram_batch_elapsed = time.perf_counter() - distogram_start
        distogram_elapsed += distogram_batch_elapsed
        log_timing(f"Distograms batch {chunk_idx:>3}", distogram_batch_elapsed, len(parsed_items), "protein")

        embedding_start = time.perf_counter()
        embeddings = generate_embeddings_for_list(
            sequences_in_order, tokenizer, model, device, show_progress=False, log_start=False
        )
        embedding_batch_elapsed = time.perf_counter() - embedding_start
        embedding_elapsed += embedding_batch_elapsed
        log_timing(f"Embeddings batch {chunk_idx:>3}", embedding_batch_elapsed, len(parsed_items), "protein")

        assert len(distograms) == len(embeddings) == len(parsed_items)
        for item, distogram, embedding in zip(parsed_items, distograms, embeddings):
            assert distogram.shape[0] == embedding.shape[0] - 2, (
                f"Length mismatch for {item['protein_id']}: distogram {distogram.shape[0]} vs "
                f"embedding L={embedding.shape[0]} (expect L-2 == n residues)"
            )

        batching_start = time.perf_counter()
        batch_embeddings = []
        batch_distograms = []
        batch_masks = []
        batch_ids = [item['protein_id'] for item in parsed_items]

        if preprocessing:
            for embedding, distogram in zip(embeddings, distograms):
                emb, dist, mask = preprocess_data(embedding, distogram, max_seq_len, emb_size, sigma_dist)
                batch_embeddings.append(emb)
                batch_distograms.append(dist)
                batch_masks.append(mask)

            batch = (
                batch_ids,
                torch.stack(batch_embeddings),
                torch.stack(batch_distograms),
                torch.stack(batch_masks),
            )
        else:
            for embedding, distogram in zip(embeddings, distograms):
                batch_embeddings.append(embedding)
                batch_distograms.append(distogram)
                batch_masks.append(None)

            batch = (
                batch_ids,
                batch_embeddings,
                batch_distograms,
                batch_masks,
            )

        batching_batch_elapsed = time.perf_counter() - batching_start
        batching_elapsed += batching_batch_elapsed
        yielded_batches += 1
        log_timing(f"Batch assembly {chunk_idx:>3}", batching_batch_elapsed, len(parsed_items), "protein")

        if return_struct_info:
            batch_struct_info = [
                {
                    'protein_id': item['protein_id'],
                    'cif_path': str(item['structure_file']),
                    'sequence': item['sequence'],
                    'residue_df': item['residue_df'],
                    'coords': item['coords'],
                }
                for item in parsed_items
            ]
            yield batch + (batch_struct_info,)
        else:
            yield batch

    if skipped_total:
        logger.warning(f"Skipped {skipped_total} broken structure file(s) during parsing.")
    preparation_compute_total = parse_elapsed + distogram_elapsed + embedding_elapsed + batching_elapsed
    log_timing("Structure files parsing", parse_elapsed, parsed_total, "file")
    log_timing("Distograms generation", distogram_elapsed, parsed_total, "protein")
    log_timing("Embeddings generation", embedding_elapsed, parsed_total, "protein")
    log_timing("Batch assembly total", batching_elapsed, yielded_batches, "batch")
    log_timing("Preparation compute total", preparation_compute_total, parsed_total, "protein")


def inference_for_batch(
    model_,
    embeddings_batch: torch.Tensor,
    distograms_batch: torch.Tensor,
    masks_batch: torch.Tensor,
    batch_ids=None,
    batch_idx=None,
    ontology=None,
    return_attr=True,
):
    start = time.perf_counter()
    logits, logits_struct, logits_esm, gate, _, _, _ = model_(
        embeddings_batch, distograms_batch, masks_batch, return_attr=return_attr
    )
    preds = logits.sigmoid().detach().cpu().numpy()
    preds_struct = logits_struct.sigmoid().detach().cpu().numpy()
    preds_seq = logits_esm.sigmoid().detach().cpu().numpy()
    gate_np = gate.detach().cpu().numpy()
    elapsed = time.perf_counter() - start

    ont_prefix = f"{ontology} " if ontology is not None else ""
    batch_label = f"{ont_prefix}Inference batch {batch_idx:>3}" if batch_idx is not None else f"{ont_prefix}Inference batch"
    batch_size = len(batch_ids) if batch_ids is not None else embeddings_batch.shape[0]
    log_timing(batch_label, elapsed, batch_size, "protein")
    return preds, preds_struct, preds_seq, gate_np


def extract_go_terms_from_preds(mapping: dict[str, int],
                                preds: list[float],
                                threshold=0.5) -> list[str]:
    """
    Extract GO terms from the predictions.
    If threshold is None, return all GO terms.
    """
    if threshold is None:
        return [mapping[i] for i in range(len(preds))]

    go_terms = []
    for i in range(len(preds)):
        if preds[i] > threshold:
            go_terms.append(mapping[i])
    return go_terms


# ============
# GO hierarchy
# ============

# Ontology root terms (molecular_function / cellular_component / biological_process).
# They are trivially predicted for every protein, so they are excluded from the summary.
ROOT_GO_IDS = frozenset({"GO:0003674", "GO:0005575", "GO:0008150"})


def build_go_descendant_indices(go_terms_mapping, go_graph_fast):
    """Map each label index to the indices of its GO descendants (incl. itself).

    Args:
        go_terms_mapping: dict ``{label_index -> GO id}`` for one ontology.
        go_graph_fast: a parsed GO graph (``fastobo.load(...)``).

    Returns:
        dict ``{label_index -> np.ndarray of descendant label indices}``, used to
        propagate predictions up the GO hierarchy.
    """
    children_by_go = {}
    for frame in go_graph_fast:
        if not isinstance(frame, fastobo.term.TermFrame):
            continue
        child_go = str(frame.id)
        for clause in frame:
            if isinstance(clause, fastobo.term.IsAClause):
                parent_go = str(clause.term)
                children_by_go.setdefault(parent_go, set()).add(child_go)

    idx_to_go = {int(idx): str(go_id) for idx, go_id in go_terms_mapping.items()}
    go_to_idx = {go_id: idx for idx, go_id in idx_to_go.items()}
    memo = {}

    def collect_descendants(go_id):
        if go_id in memo:
            return memo[go_id]
        descendants = {go_id}
        for child_go in children_by_go.get(go_id, set()):
            descendants.update(collect_descendants(child_go))
        memo[go_id] = descendants
        return descendants

    descendant_indices = {}
    for idx, go_id in idx_to_go.items():
        descendant_indices[idx] = np.array(
            [go_to_idx[descendant_go] for descendant_go in collect_descendants(go_id) if descendant_go in go_to_idx],
            dtype=np.int64,
        )
    return descendant_indices


def propagate_prediction_record(record, descendant_indices):
    """Propagate per-label probabilities up the GO hierarchy (max over descendants).

    For each prediction branch in ``record`` (fusion / structure / sequence), replaces
    each label's score with the maximum score among its GO descendants and records the
    source index the propagated value came from.
    """
    propagated_record = dict(record)
    branch_specs = [
        ('pred_proba', 'prop_source_idx'),
        ('pred_proba_struct', 'struct_prop_source_idx'),
        ('pred_proba_seq', 'seq_prop_source_idx'),
    ]
    for value_key, source_key in branch_specs:
        values = record.get(value_key)
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float32)
        propagated_values = values.copy()
        source_indices = np.arange(len(values), dtype=np.int64)
        for idx, descendant_idxs in descendant_indices.items():
            if descendant_idxs.size:
                best_local = int(np.argmax(values[descendant_idxs]))
                best_idx = int(descendant_idxs[best_local])
                propagated_values[idx] = float(values[best_idx])
                source_indices[idx] = best_idx
        propagated_record[value_key] = propagated_values
        propagated_record[source_key] = source_indices
    return propagated_record


# =================
# Prediction tables
# =================

def _resolve_go_term(go_terms_mapping, term_idx):
    """Map a label index to its GO id string."""
    idx = int(term_idx)
    if isinstance(go_terms_mapping, dict):
        raw = go_terms_mapping.get(idx, go_terms_mapping.get(str(idx), idx))
    else:
        raw = go_terms_mapping[idx]
    return str(raw)


def _resolve_go_name(go_term, go_name_map):
    """Map a GO id to its human-readable name (empty string if unknown)."""
    if go_name_map is None:
        return ""
    value = go_name_map.get(go_term, "")
    return "" if value is None or (isinstance(value, float) and not np.isfinite(value)) else str(value)


def build_all_prediction_table(protein_records, go_terms_mapping):
    """Per-GO-term table of raw probabilities for the fusion / structure / sequence branches.

    Columns: go_id, pred_prob (fusion), struct_prob, seq_prob (sequence), gate.
    """
    rows = []
    for record in protein_records:
        probs = np.asarray(record["pred_proba"], dtype=np.float32)
        struct_probs = np.asarray(record["pred_proba_struct"], dtype=np.float32)
        seq_probs = np.asarray(record["pred_proba_seq"], dtype=np.float32)
        gate_probs = None if record.get("pred_gate") is None else np.asarray(record["pred_gate"], dtype=np.float32)

        for idx in range(len(probs)):
            go_term = _resolve_go_term(go_terms_mapping, idx)
            rows.append(
                {
                    "go_id": go_term,
                    "pred_prob": float(probs[idx]),
                    "struct_prob": float(struct_probs[idx]),
                    "seq_prob": float(seq_probs[idx]),
                    "gate": float(gate_probs[idx]) if gate_probs is not None else None,
                }
            )
    return pd.DataFrame(rows)


def build_propagated_prediction_table(protein_records, go_terms_mapping):
    """Per-GO-term table with hierarchically-propagated source terms and raw probabilities.

    Columns: go_id, prop_go_id, struct_prop_go_id, seq_prop_go_id, pred_prob, struct_prob, seq_prob.
    """
    rows = []
    for record in protein_records:
        probs = np.asarray(record["pred_proba"], dtype=np.float32)
        struct_probs = np.asarray(record["pred_proba_struct"], dtype=np.float32)
        seq_probs = np.asarray(record["pred_proba_seq"], dtype=np.float32)
        prop_source_idx = np.asarray(record["prop_source_idx"], dtype=np.int64)
        struct_prop_source_idx = np.asarray(record["struct_prop_source_idx"], dtype=np.int64)
        seq_prop_source_idx = np.asarray(record["seq_prop_source_idx"], dtype=np.int64)

        for idx in range(len(probs)):
            go_term = _resolve_go_term(go_terms_mapping, idx)
            rows.append(
                {
                    "go_id": go_term,
                    "prop_go_id": _resolve_go_term(go_terms_mapping, prop_source_idx[idx]),
                    "struct_prop_go_id": _resolve_go_term(go_terms_mapping, struct_prop_source_idx[idx]),
                    "seq_prop_go_id": _resolve_go_term(go_terms_mapping, seq_prop_source_idx[idx]),
                    "pred_prob": float(probs[idx]),
                    "struct_prob": float(struct_probs[idx]),
                    "seq_prob": float(seq_probs[idx]),
                }
            )
    return pd.DataFrame(rows)


def _threshold_pair(threshold):
    """Normalize a threshold spec to a ``(fusion_and_sequence, structure)`` float pair.

    Accepts a single float (applied to all branches) or a 1/2-length tuple/list. The two-value
    form applies the first threshold to the fusion and sequence branches and the second to the
    structure branch (the structural prober is trained with a different loss and tends to output
    higher probabilities on average).
    """
    if isinstance(threshold, (tuple, list)):
        if len(threshold) == 1:
            return float(threshold[0]), float(threshold[0])
        return float(threshold[0]), float(threshold[1])
    return float(threshold), float(threshold)


def _branch_mask(probs, t):
    """Boolean keep-mask for one branch. ``t <= 0`` keeps everything, ``t >= 1`` keeps nothing."""
    if t <= 0.0:
        return np.ones(len(probs), dtype=bool)
    if t >= 1.0:
        return np.zeros(len(probs), dtype=bool)
    return probs >= t


def build_prediction_summary(protein_records, go_terms_mapping, threshold=0.5, top_k=10,
                             go_name_map=None, propagated_records=None):
    """Thresholded / top-k prediction summary across branches (fusion, structure, sequence).

    ``threshold`` is either a single float (applied to all branches) or a ``(fusion_and_sequence,
    structure)`` pair. For each protein, a GO term is kept when *any* branch probability passes
    its threshold. A threshold of 0 keeps every term, and a threshold of 1 keeps none. Selected
    terms are sorted by fusion probability, then sequence, then structure (all descending).
    Reports raw probabilities per branch. When ``propagated_records`` is provided, the propagated
    columns (prop_go_id, struct_prop_go_id, seq_prop_go_id, pred_prop_prob, struct_prop_prob,
    seq_prop_prob) are added as well; when it is None those columns are omitted entirely. Returns
    the summary DataFrame and ``{protein_id: [selected GO ids]}``.
    """
    rows = []
    goterms = {}
    propagated_lookup = {}
    if propagated_records is not None:
        propagated_lookup = {str(record["protein_id"]): record for record in propagated_records}
    for record in protein_records:
        probs = np.asarray(record["pred_proba"], dtype=np.float32)
        struct_probs = np.asarray(record["pred_proba_struct"], dtype=np.float32)
        seq_probs = np.asarray(record["pred_proba_seq"], dtype=np.float32)
        gate_probs = None if record.get("pred_gate") is None else np.asarray(record["pred_gate"], dtype=np.float32)
        propagated_record = propagated_lookup.get(str(record["protein_id"]))
        prop_probs = None if propagated_record is None else np.asarray(propagated_record["pred_proba"], dtype=np.float32)
        struct_prop_probs = None if propagated_record is None else np.asarray(propagated_record["pred_proba_struct"], dtype=np.float32)
        seq_prop_probs = None if propagated_record is None else np.asarray(propagated_record["pred_proba_seq"], dtype=np.float32)
        prop_source_idx = None if propagated_record is None or propagated_record.get("prop_source_idx") is None else np.asarray(propagated_record["prop_source_idx"], dtype=np.int64)
        struct_prop_source_idx = None if propagated_record is None or propagated_record.get("struct_prop_source_idx") is None else np.asarray(propagated_record["struct_prop_source_idx"], dtype=np.int64)
        seq_prop_source_idx = None if propagated_record is None or propagated_record.get("seq_prop_source_idx") is None else np.asarray(propagated_record["seq_prop_source_idx"], dtype=np.int64)

        # Keep a term if any branch passes its threshold. Fusion and sequence share the first
        # threshold; structure uses the second (see _threshold_pair).
        if threshold is None:
            idxs = np.arange(len(probs), dtype=np.int64)
        else:
            t_fs, t_struct = _threshold_pair(threshold)
            mask = _branch_mask(probs, t_fs) | _branch_mask(seq_probs, t_fs) | _branch_mask(struct_probs, t_struct)
            idxs = np.where(mask)[0]

        # Drop ontology root terms (molecular_function / cellular_component / biological_process):
        # they are trivially predicted and should not occupy slots in the summary.
        idxs = np.array(
            [i for i in idxs if _resolve_go_term(go_terms_mapping, int(i)) not in ROOT_GO_IDS],
            dtype=np.int64,
        )

        # Sort selected terms by fusion probability, then sequence, then structure (all desc).
        if idxs.size:
            order = np.lexsort((struct_probs[idxs], seq_probs[idxs], probs[idxs]))[::-1]
            idxs = idxs[order]

        if top_k is not None:
            idxs = idxs[: int(top_k)]

        goterms[record["protein_id"]] = [_resolve_go_term(go_terms_mapping, int(i)) for i in idxs]
        for rank, idx in enumerate(idxs, start=1):
            go_term = _resolve_go_term(go_terms_mapping, int(idx))
            row = {
                "protein_id": record["protein_id"],
                "rank": rank,
                # "selection": selection,
                "go_term": go_term,
                "go_term_name": _resolve_go_name(go_term, go_name_map),
                # "term_idx": int(idx),
                "pred_prob": float(probs[int(idx)]),
                "struct_prob": float(struct_probs[int(idx)]),
                "seq_prob": float(seq_probs[int(idx)]),
                "gate": float(gate_probs[int(idx)]) if gate_probs is not None else None,
            }
            # Propagated columns are included only when propagation data is available.
            if propagated_record is not None:
                row.update(
                    {
                        "prop_go_id": None if prop_source_idx is None else _resolve_go_term(go_terms_mapping, int(prop_source_idx[int(idx)])),
                        "struct_prop_go_id": None if struct_prop_source_idx is None else _resolve_go_term(go_terms_mapping, int(struct_prop_source_idx[int(idx)])),
                        "seq_prop_go_id": None if seq_prop_source_idx is None else _resolve_go_term(go_terms_mapping, int(seq_prop_source_idx[int(idx)])),
                        "pred_prop_prob": None if prop_probs is None else float(prop_probs[int(idx)]),
                        "struct_prop_prob": None if struct_prop_probs is None else float(struct_prop_probs[int(idx)]),
                        "seq_prop_prob": None if seq_prop_probs is None else float(seq_prop_probs[int(idx)]),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows), goterms
