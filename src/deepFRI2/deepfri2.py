#!/usr/bin/env python
"""deepFRI2 inference CLI.

Predict Gene Ontology (GO) terms for proteins with deepFRI2.

Usage
-----
    python src/deepFRI2/deepfri2.py --input_dir /path/to/structures
    python src/deepFRI2/deepfri2.py -i structures/ -o results/run1 -f ids.txt

Inputs
------
    --input_dir  / -i : folder with protein structures (``.cif`` / ``.pdb``).       [required]
    --output_dir / -o : folder for results; defaults to ``<repo>/results`` if omitted.
    --ids_file   / -f : text file with one entry per line naming the structures to
                        run (e.g. ``abCD.cif`` or just ``abCD``); if omitted, every
                        structure in ``--input_dir`` is processed.
    --batch_size / -b : proteins per inference batch (default: 32).
    --threshold  / -t : keep GO terms with score >= this in the summary (default: 0.1).
    --top_k      / -k : max GO terms per protein in the summary (default: all selected).
    --prop       / -p : propagate scores up the GO hierarchy (default: off). When set, also
                        writes the preds_propagated/ folder and adds the propagated columns
                        (prop_go_id, struct_prop_go_id, seq_prop_go_id, pred_prop_prob,
                        struct_prop_prob, seq_prop_prob) to the summary.
    --verbose    / -v : enable debug logging (e.g. per-checkpoint weight-load report).

Output (under <output_dir>; same as inference.ipynb)
------
    preds/<protein>__<ontology>.csv            : all GO terms with raw fusion / structure /
                                                 sequence probabilities and the gate.
    preds_propagated/<protein>__<ontology>.csv : all GO terms with hierarchically-propagated
                                                 source terms and probabilities (only with --prop).
    prediction_summary.csv                     : thresholded / top-k predictions per protein
                                                 across all ontologies (raw, plus propagated
                                                 columns only with --prop).
    log.txt                                    : run log.

ESM weights must be present locally (run ``src/deepFRI2/download_esm.py`` once); deepFRI2
head checkpoints are expected under ``params/<ontology>/``.
"""

import argparse
import json
import time
from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path

from config import (
    DIST_TYPE,
    EMBED_MODEL_NAME,
    ESM_DIM,
    GO_VERSION,
    M_ANTI,
    M_DIAG,
    MAX_SEQ_LEN,
    MODEL_NAMES,
    NUM_ANTI,
    NUM_DIAG,
    ONTOLOGIES,
    SIGMA_DIST,
)
from version import config_version

logger = getLogger("inference")

# Repository paths. deepfri2.py lives at <repo>/src/deepFRI2/, so the repo root is two
# levels up and the model weights / checkpoints live under <repo>/params/.
REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_DIR = REPO_ROOT / "params"

# Defaults for the tunable inference options (overridable on the command line).
DEFAULT_BATCH_SIZE = 32
DEFAULT_THRESHOLD = 0.1
DEFAULT_TOP_K = None

BANNER = r"""
      _                 _____ ____  ___ ____
   __| | ___  ___ _ __ |  ___|  _ \|_ _|___ \
  / _` |/ _ \/ _ \ '_ \| |_  | |_) || |  __) |
 | (_| |  __/  __/ |_) |  _| |  _ < | | / __/
  \__,_|\___|\___| .__/|_|   |_| \_\___|_____|
                  |_|
   Predicting protein function
   Across sequence and structure,
   Scalable. Interpretable. By design.
"""


# =============
# Setup helpers
# =============

def load_go_terms_mappings(params_dir):
    """Return ``{ontology: {label_index: GO id}}`` from the per-ontology label files."""
    mappings = {}
    for ontology, names in MODEL_NAMES.items():
        labels_path = Path(params_dir) / ontology / f"labels_{names['fusion']}.json"
        with open(labels_path) as f:
            mappings[ontology] = {int(k): v for k, v in json.load(f).items()}
    return mappings


def load_go_name_map(obo_path):
    """Return ``{GO id: human-readable name}`` parsed from a GO ``.obo`` file."""
    import fastobo

    go_graph = fastobo.load(str(obo_path))
    go_name_map = {}
    for frame in go_graph:
        if isinstance(frame, fastobo.term.TermFrame):
            term_id = str(frame.id)
            for clause in frame:
                if isinstance(clause, fastobo.term.NameClause):
                    go_name_map[term_id] = str(clause.name)
                    break
    return go_graph, go_name_map


def load_esm(device, params_dir):
    """Load the local ESM tokenizer and model in eval mode on ``device``."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    esm_dir = str(Path(params_dir) / EMBED_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(esm_dir, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(esm_dir, local_files_only=True)
    model = model.to(device).to(dtype=torch.float32).eval()
    logger.info(f"ESM embedding model loaded from {esm_dir}")
    return tokenizer, model


def load_models(device, num_labels_by_ontology, params_dir):
    """Build and load a deepFRI2 fusion model for every ontology."""
    from model import build_deepfri2_model

    models = {
        ontology: build_deepfri2_model(
            ontology,
            run_names=MODEL_NAMES[ontology],
            num_labels=num_labels_by_ontology[ontology],
            device=device,
            params_dir=str(params_dir),
            esm_dim=ESM_DIM,
            m_diag=M_DIAG,
            m_anti=M_ANTI,
            num_diag=NUM_DIAG,
            num_anti=NUM_ANTI,
        )
        for ontology in ONTOLOGIES
    }
    logger.info(f"deepFRI2 models loaded: {', '.join(ONTOLOGIES)}")
    return models


def resolve_file_names(input_dir, ids_file):
    """Turn an ids file into a list of structure file names present in ``input_dir``.

    Each line may be a file name (``abCD.cif``) or a bare id (``abCD``). Returns None
    when ``ids_file`` is None (meaning: process every structure in the folder).
    """
    if ids_file is None:
        return None
    entries = [line.strip() for line in Path(ids_file).read_text().splitlines() if line.strip()]
    files = sorted(Path(input_dir).glob("*.cif")) + sorted(Path(input_dir).glob("*.pdb"))
    by_name = {f.name: f.name for f in files}
    by_stem = {}
    for f in files:
        by_stem.setdefault(f.stem, f.name)

    resolved, missing = [], []
    for entry in entries:
        if entry in by_name:
            resolved.append(by_name[entry])
        elif Path(entry).stem in by_stem:
            resolved.append(by_stem[Path(entry).stem])
        else:
            missing.append(entry)
    if missing:
        preview = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        logger.warning(f"{len(missing)} id(s) from {ids_file} not found in {input_dir}: {preview}")
    return resolved


# =========
# Inference
# =========

def run_inference(input_dir, output_dir, file_names, models, tokenizer, esm_model,
                  device, go_terms_mappings, descendant_indices_by_ontology, go_name_map,
                  batch_size, threshold, top_k, prop=False):
    """Run all-ontology inference and write the same outputs as inference.ipynb.

    Produces, under ``output_dir``:
    - ``preds/<protein>__<ontology>.csv``            (raw fusion / structure / sequence probs + gate)
    - ``prediction_summary.csv``                     (thresholded / top-k summary across ontologies)

    When ``prop`` is True, GO-hierarchy propagation is run as well, which additionally produces:
    - ``preds_propagated/<protein>__<ontology>.csv`` (hierarchically-propagated terms + probs)
    - the propagated columns (prop_go_id, struct_prop_go_id, seq_prop_go_id, pred_prop_prob,
      struct_prop_prob, seq_prop_prob) in ``prediction_summary.csv``.
    """
    from utils import (
        build_all_prediction_table,
        build_prediction_summary,
        build_propagated_prediction_table,
        inference_for_batch,
        log_timing,
        prepare_batches_for_inference,
        propagate_prediction_record,
    )

    output_dir = Path(output_dir)
    preds_dir = output_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_propagated_dir = output_dir / "preds_propagated"
    if prop:
        preds_propagated_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "prediction_summary.csv"
    summary_path.unlink(missing_ok=True)
    header_written = False

    total_proteins = 0
    inference_elapsed = 0.0  # model forward-pass time only (excludes parsing/embedding/IO)
    start = time.perf_counter()

    batch_iterator = prepare_batches_for_inference(
        input_dir,
        tokenizer=tokenizer,
        model=esm_model,
        device=device,
        atom_name=DIST_TYPE,
        max_seq_len=MAX_SEQ_LEN,
        emb_size=ESM_DIM,
        sigma_dist=SIGMA_DIST,
        batch_size=batch_size,
        file_names=file_names,
    )

    for batch_idx, batch in enumerate(batch_iterator, start=1):
        batch_ids, embeddings_batch, distograms_batch, masks_batch = batch
        embeddings_batch = embeddings_batch.to(device)
        distograms_batch = distograms_batch.to(device)
        masks_batch = masks_batch.to(device)
        total_proteins += len(batch_ids)

        for ontology in ONTOLOGIES:
            mapping = go_terms_mappings[ontology]
            descendant_indices = descendant_indices_by_ontology[ontology] if prop else None
            inference_start = time.perf_counter()
            preds, preds_struct, preds_seq, gate = inference_for_batch(
                models[ontology],
                embeddings_batch,
                distograms_batch,
                masks_batch,
                batch_ids=batch_ids,
                batch_idx=batch_idx,
                ontology=ontology,
            )
            inference_elapsed += time.perf_counter() - inference_start

            batch_records = [
                {
                    "protein_id": protein_id,
                    "ontology": ontology,
                    "pred_proba": probs,
                    "pred_proba_struct": probs_struct,
                    "pred_proba_seq": probs_seq,
                    "pred_gate": gate_vals,
                }
                for protein_id, probs, probs_struct, probs_seq, gate_vals in zip(
                    batch_ids, preds, preds_struct, preds_seq, gate
                )
            ]
            # GO-hierarchy propagation is only computed when requested (--prop).
            propagated_batch_records = (
                [propagate_prediction_record(record, descendant_indices) for record in batch_records]
                if prop
                else None
            )

            # Per-protein raw table (fusion / structure / sequence branches), plus the
            # propagated table when propagation is enabled.
            for i, record in enumerate(batch_records):
                build_all_prediction_table([record], mapping).to_csv(
                    preds_dir / f"{record['protein_id']}__{ontology}.csv", index=False
                )
                if prop:
                    build_propagated_prediction_table([propagated_batch_records[i]], mapping).to_csv(
                        preds_propagated_dir / f"{record['protein_id']}__{ontology}.csv", index=False
                    )

            # Thresholded / top-k summary, appended across ontologies and batches.
            summary, _ = build_prediction_summary(
                batch_records,
                mapping,
                threshold=threshold,
                top_k=top_k,
                go_name_map=go_name_map,
                propagated_records=propagated_batch_records,
            )
            summary.insert(0, "ontology", ontology)
            summary.to_csv(summary_path, mode="a", header=not header_written, index=False)
            header_written = True

    log_timing("Inference time", inference_elapsed, total_proteins, "protein")
    log_timing("Total time", time.perf_counter() - start, total_proteins, "protein")
    if prop:
        logger.info(f"Wrote per-protein tables to {preds_dir} and {preds_propagated_dir}")
    else:
        logger.info(f"Wrote per-protein tables to {preds_dir}")
    logger.info(f"Wrote summary to {summary_path} for {total_proteins} protein(s)")
    return summary_path


# ===
# CLI
# ===

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="deepFRI2 inference: predict GO terms for protein structures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", "-i", type=Path, required=True,
                        help="Folder with protein structures (.cif / .pdb).")
    parser.add_argument("--output_dir", "-o", type=Path, default=None,
                        help="Folder for results (default: <repo>/results).")
    parser.add_argument("--ids_file", "-f", type=Path, default=None,
                        help="Text file listing structures to run (one per line, "
                             "e.g. 'abCD.cif' or 'abCD'); default: all files in input_dir.")
    parser.add_argument("--batch_size", "-b", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Proteins per inference batch (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Keep GO terms with score >= this in the summary "
                             f"(default: {DEFAULT_THRESHOLD}).")
    parser.add_argument("--top_k", "-k", type=int, default=DEFAULT_TOP_K,
                        help="Max GO terms per protein in the summary (default: all selected).")
    parser.add_argument("--prop", "-p", action="store_true",
                        help="Propagate scores up the GO hierarchy (default: off); also writes "
                             "the preds_propagated/ folder and the propagated summary columns.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging (e.g. per-checkpoint weight-load report).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Terminal-only intro (printed, never logged to the run file). flush=True so it
    # appears before the logging output even when stdout is piped/redirected.
    print(BANNER, flush=True)

    # Heavy dependencies are imported only now, after argument parsing, so `--help`
    # and argument errors return without paying the torch/transformers import cost.
    import torch
    from utils import build_go_descendant_indices, configure_log_file

    # Console logging; --verbose lowers the level to DEBUG. Per-run file logging is
    # attached below (the file handler is at INFO, so it stays clean of debug noise).
    basicConfig(
        level=DEBUG if args.verbose else INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir is not None else REPO_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_log_file(output_dir / "log.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Report the full run configuration up front for transparency.
    ids_file = args.ids_file if args.ids_file is not None else "None (all structures in input dir)"
    top_k = args.top_k if args.top_k is not None else "all"
    logger.info(f"deepFRI2 version : {config_version()}")
    logger.info(f"Input dir       : {input_dir}")
    logger.info(f"Output dir      : {output_dir}")
    logger.info(f"IDs file        : {ids_file}")
    logger.info(f"Batch size      : {args.batch_size}")
    logger.info(f"Threshold       : {args.threshold}")
    logger.info(f"Top k           : {top_k}")
    logger.info(f"Propagate       : {args.prop}")
    logger.info(f"Verbose         : {args.verbose}")
    logger.info(f"Device          : {device}")

    go_terms_mappings = load_go_terms_mappings(PARAMS_DIR)
    num_labels_by_ontology = {ont: len(m) for ont, m in go_terms_mappings.items()}

    tokenizer, esm_model = load_esm(device, PARAMS_DIR)
    models = load_models(device, num_labels_by_ontology, PARAMS_DIR)

    go_graph, go_name_map = load_go_name_map(PARAMS_DIR / f"go_{GO_VERSION}.obo")
    # GO-hierarchy descendant indices are only needed for propagation (--prop).
    descendant_indices_by_ontology = (
        {
            ontology: build_go_descendant_indices(mapping, go_graph)
            for ontology, mapping in go_terms_mappings.items()
        }
        if args.prop
        else None
    )

    file_names = resolve_file_names(input_dir, args.ids_file)

    run_inference(
        input_dir, output_dir, file_names, models, tokenizer, esm_model, device,
        go_terms_mappings, descendant_indices_by_ontology, go_name_map,
        batch_size=args.batch_size, threshold=args.threshold, top_k=args.top_k,
        prop=args.prop,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
