#!/usr/bin/env python
"""deepFRI2 interpretability CLI.

Predict GO terms for protein structures and, for the selected terms, write one interpretability
report per protein/GO-term pair: which residues and which residue-residue contacts drove the
score, and which contiguous segments look like candidate functional-activity sites (binding
sites, catalytic sites, interaction interfaces -- the report does not assume which).

Reports always come from the fusion model: the explanation is built from both branches
(ESM embeddings and distogram kernels) and weighted by the model's own gate, so a
sequence-driven term is explained mostly by sequence signals and a structure-driven term by
structural ones.

Usage
-----
    python src/deepFRI2/interpret.py --input /path/to/structures
    python src/deepFRI2/interpret.py -i structures/ -o results/reports -f ids.txt -a mf
    python src/deepFRI2/interpret.py -i structures/ --go_terms pairs.tsv --true_residues sites.csv

Inputs
------
    --input      / -i : folder with protein structures (``.cif`` / ``.pdb``).          [required]
                        Unlike ``deepfri2.py`` a FASTA input is not accepted: the structural part
                        of the explanation needs coordinates.
    --output_dir / -o : folder for predictions and reports; defaults to ``<repo>/results``.
    --ids_file   / -f : text file with one entry per line naming the structures to run (same
                        format as ``deepfri2.py --ids_file``). Default: every top-level structure.
    --aspect     / -a : comma-separated GO aspects (ontologies) to run: any of MF, CC, BP
                        (case-insensitive). Default: mf,cc,bp.
    --batch_size / -b : proteins per inference batch (default: 32).
    --threshold  / -t : one float (all branches) or two comma-separated floats (fusion/sequence,
                        structure) for keeping GO terms in the summary (default: 0.1,0.2). The
                        first value also selects which terms get a report.
    --top_k      / -k : max GO terms per protein in the prediction summary (default: all selected).
    --prop       / -p : propagate scores up the GO hierarchy (default: off).
    --summary    / -s : skip the per-protein ``preds/`` folders (default: off). Interpretability
                        reports are written either way.
    --verbose    / -v : debug logging, plus per-step timings for every analyzed term.

Term selection
--------------
    A protein gets a report for every GO term scoring >= the first ``--threshold`` value, capped at
    ``--max_terms`` per protein and aspect; if no term passes, the ``--top_k_fallback`` best terms
    are used instead. With ``--go_terms`` the listed terms are analyzed instead (one pass, terms
    resolved to whichever requested aspect defines them), which is the mode to use for
    benchmarking against known annotations.

Interpretability options (rarely changed; defaults match the published reports)
------
    --max_terms       : max reported GO terms per protein and aspect (default: 11).
    --top_k_fallback  : reported terms when none passes the threshold (default: 4).
    --smooth_window   : moving-average window for the per-residue curves (default: 1, no smoothing).
    --top_windows     : kernel windows kept per report (default: 12).
    --top_residues    : residues kept in ``top_residues.csv`` (default: 20).
    --save_workers    : threads saving a report while the next term is analyzed (0 = save inline,
                        default: 1).
    --go_terms        : CSV/TSV with ``protein_id`` and ``go_term`` columns listing the terms to
                        analyze per protein. A ``go_term`` cell may hold one GO id, a
                        comma-separated list, or a Python-style list.
    --true_residues   : CSV/TSV with ``protein_id``, ``go_term`` and an ``activity_positions``
                        (or ``binding_positions`` / ``positions``) column holding reference
                        (1-based) residue indices; they are drawn on the report plots for
                        comparison and otherwise do not affect the analysis.

Output (under <output_dir>)
------
    <protein>/<GO_id>/sequence_analysis.png        : per-residue attribution curves (both branches).
    <protein>/<GO_id>/kernel_distmap.png           : distogram-level attribution and top kernel windows.
    <protein>/<GO_id>/structure_analysis.html      : interactive 3D viewer coloured by attribution.
    <protein>/<GO_id>/residues.csv                 : every residue with all attribution signals.
    <protein>/<GO_id>/top_residues.csv             : the highest-scoring residues.
    <protein>/<GO_id>/activity_site_candidates.csv : contiguous candidate activity segments.
    <protein>/<GO_id>/top_kernel_windows.csv       : the kernel windows that drove the structure branch.
    <protein>/<GO_id>/kernel_gradinput.npy         : residue-residue grad x input matrix.
    <protein>/<GO_id>/summary.json                 : the report's headline numbers and file paths.
    interpretability_summary.csv                   : one row per analyzed protein/GO-term pair.
    prediction_summary.csv, preds/, preds_propagated/, log.txt : as in ``deepfri2.py``.

ESM weights must be present locally (run ``src/deepFRI2/download_esm.py`` once); deepFRI2
head checkpoints are expected under ``params/<ontology>/``.
"""

import argparse
import shlex
import sys
import time
from logging import DEBUG, INFO, basicConfig, getLogger
from pathlib import Path

from config import (
    DIST_TYPE,
    ESM_DIM,
    GO_VERSION,
    MAX_SEQ_LEN,
    ONTOLOGIES,
    SIGMA_DIST,
)

# The interpretability CLI shares the whole setup path with the inference CLI (paths,
# model/GO loading, input validation, --aspect/--threshold parsing); only the run loop differs.
from deepfri2 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    PARAMS_DIR,
    REPO_ROOT,
    _fatal,
    load_esm,
    load_go_name_map,
    load_go_terms_mappings,
    load_models,
    parse_aspect,
    parse_threshold,
    resolve_file_names,
    validate_input_dir,
)
from version import config_version

logger = getLogger("inference")

# Defaults for the interpretability options. These reproduce the reports shipped with the paper;
# they trade runtime for report depth.
DEFAULT_MAX_TERMS = 11
DEFAULT_TOP_K_FALLBACK = 4
DEFAULT_SMOOTH_WINDOW = 1
DEFAULT_TOP_WINDOWS = 12
DEFAULT_TOP_RESIDUES = 20
DEFAULT_SAVE_WORKERS = 1


# ==================
# Benchmark inputs
# ==================

def _read_table(path):
    """Read a CSV/TSV table (tab-delimited for ``.tsv`` / ``.tab``, comma otherwise)."""
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        _fatal(f"File does not exist: {path}")
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    return pd.read_csv(path, delimiter=delimiter)


def _require_columns(path, frame, columns):
    """Terminate the run when a benchmark table lacks a required column."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        _fatal(f"{path}: missing required column(s): {', '.join(missing)}. "
               f"Found: {', '.join(map(str, frame.columns))}")


def _as_term_list(value):
    """Parse a GO-term cell into a list of GO ids.

    Accepts a single id (``GO:0008270``), a comma-separated list (``GO:0008270,GO:0043167``) or a
    Python-style list (``['GO:0008270', 'GO:0043167']``); blanks and NaNs yield an empty list.
    """
    from ast import literal_eval

    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text[0] in "[(":
        try:
            return [str(item).strip() for item in literal_eval(text)]
        except (ValueError, SyntaxError):
            return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _as_int_list(value):
    """Parse a residue-position cell into a list of ints (same input forms as ``_as_term_list``)."""
    positions = []
    for item in _as_term_list(value):
        try:
            positions.append(int(float(item)))
        except ValueError:
            continue
    return positions


def load_custom_go_terms(path):
    """Return ``{protein_id: [GO id, ...]}`` from a table with ``protein_id`` / ``go_term`` columns.

    Rows of the same protein are merged (duplicates dropped, order kept), so both one-row-per-term
    and one-row-per-protein-with-a-list layouts work.
    """
    frame = _read_table(path)
    _require_columns(path, frame, ("protein_id", "go_term"))
    terms_by_protein = {}
    for protein_id, raw_terms in zip(frame["protein_id"].astype(str), frame["go_term"]):
        for go_term in _as_term_list(raw_terms):
            bucket = terms_by_protein.setdefault(protein_id, [])
            if go_term not in bucket:
                bucket.append(go_term)
    if not terms_by_protein:
        _fatal(f"{path}: no usable protein_id / go_term pairs found")
    logger.info(f"Custom GO terms  : {sum(map(len, terms_by_protein.values()))} term(s) "
                f"for {len(terms_by_protein)} protein(s) from {path}")
    return terms_by_protein


def load_true_residues(path):
    """Return ``{(protein_id, GO id): [residue index, ...]}`` from a benchmark annotation table.

    Requires ``protein_id``, ``go_term`` and a positions column: ``activity_positions``,
    ``binding_positions`` or ``positions`` (the reference residues need not be a binding site --
    a catalytic site or an interaction interface works the same way). A row whose ``go_term`` holds
    several ids contributes its positions to each of them, as one annotated ligand site does.
    Positions are 1-based residue indices; out-of-range ones are dropped later, per protein.
    """
    frame = _read_table(path)
    _require_columns(path, frame, ("protein_id", "go_term"))
    positions_column = next(
        (name for name in ("activity_positions", "binding_positions", "positions")
         if name in frame.columns),
        None,
    )
    if positions_column is None:
        _fatal(f"{path}: needs a positions column (activity_positions, binding_positions or "
               f"positions). Found: {', '.join(map(str, frame.columns))}")
    residues_by_pair = {}
    for protein_id, raw_terms, raw_positions in zip(
        frame["protein_id"].astype(str), frame["go_term"], frame[positions_column]
    ):
        positions = _as_int_list(raw_positions)
        if not positions:
            continue
        for go_term in _as_term_list(raw_terms):
            bucket = residues_by_pair.setdefault((protein_id, go_term), [])
            bucket.extend(position for position in positions if position not in bucket)
    if not residues_by_pair:
        _fatal(f"{path}: no usable protein_id / go_term / {positions_column} rows found")
    logger.info(f"True residues    : {sum(map(len, residues_by_pair.values()))} residue(s) "
                f"for {len(residues_by_pair)} protein/GO pair(s) from {path}")
    return residues_by_pair


# ================
# Interpretability
# ================

def run_interpretability(input_dir, output_dir, file_names, models, tokenizer, esm_model, device,
                         go_terms_mappings, descendant_indices_by_ontology, go_name_map,
                         batch_size, threshold, top_k, aspects, options,
                         prop=False, summary_only=False, log_runtime=False,
                         custom_terms_by_protein=None, custom_true_residues=None):
    """Run inference and write interpretability reports, one inference batch at a time.

    Each batch is parsed, embedded and predicted once; its reports are written before the next
    batch is read, so peak memory stays at one batch and both summary CSVs grow incrementally.

    ``options`` holds the interpretability knobs (see ``parse_args``). ``threshold`` is the
    ``(fusion_and_sequence, structure)`` pair used for the prediction summary; its first value also
    decides which terms are reported. Returns ``(prediction_summary_path, interpretability_summary_path)``.
    """
    import matplotlib
    import pandas as pd
    import torch

    # Report plots are written to file, never shown: pin the non-interactive backend before
    # interpret_utils imports pyplot (matplotlib would pick a GUI backend on a machine with a
    # display, which can only slow the run down or fail).
    matplotlib.use("Agg")

    from interpret_utils import analyze_records_with_interpretability, build_interpretability_records
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
    preds_propagated_dir = output_dir / "preds_propagated"
    if not summary_only:
        preds_dir.mkdir(parents=True, exist_ok=True)
        if prop:
            preds_propagated_dir.mkdir(parents=True, exist_ok=True)

    # Both summaries are appended batch by batch (never held whole in memory).
    prediction_summary_path = output_dir / "prediction_summary.csv"
    interpretability_summary_path = output_dir / "interpretability_summary.csv"
    prediction_summary_path.unlink(missing_ok=True)
    interpretability_summary_path.unlink(missing_ok=True)
    prediction_header_written = False
    interpretability_header_written = False

    threshold_fs = threshold[0]
    # Restricted to the requested aspects: a term of a non-loaded aspect has no model to explain it.
    mappings_by_aspect = {ontology: go_terms_mappings[ontology] for ontology in aspects}
    # Report rows are labelled with the aspect that defines the term (--go_terms may cross aspects).
    aspect_by_go_term = {}
    for ontology in aspects:
        for go_term in mappings_by_aspect[ontology].values():
            aspect_by_go_term.setdefault(str(go_term), ontology)

    total_proteins = 0
    total_reports = 0
    inference_elapsed = 0.0
    interpretability_elapsed = 0.0
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
        return_struct_info=True,
        model_type="fusion",
    )

    for batch_idx, batch in enumerate(batch_iterator, start=1):
        batch_ids, embeddings_batch, distograms_batch, masks_batch, batch_struct_info = batch
        # Reports need the model's own inputs, so the records are built from the batch tensors
        # (copied to the CPU) before those are moved to the device.
        records = build_interpretability_records(
            batch_ids, embeddings_batch, distograms_batch, masks_batch, batch_struct_info,
            max_seq_len=MAX_SEQ_LEN,
        )
        truncated = [record["protein_id"] for record in records if record["was_truncated"]]
        if truncated:
            preview = ", ".join(truncated[:5]) + (", ..." if len(truncated) > 5 else "")
            logger.warning(f"{len(truncated)} protein(s) longer than {MAX_SEQ_LEN} aa; reports cover "
                           f"the first {MAX_SEQ_LEN} residues only: {preview}")

        embeddings_batch = embeddings_batch.to(device)
        distograms_batch = distograms_batch.to(device)
        masks_batch = masks_batch.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_proteins += len(batch_ids)

        # --- predictions (all aspects) -------------------------------------------------------
        probabilities_by_aspect = {}
        batch_summaries = []
        for ontology in aspects:
            mapping = go_terms_mappings[ontology]
            inference_start = time.perf_counter()
            preds, preds_struct, preds_seq, gate = inference_for_batch(
                models[ontology],
                embeddings_batch,
                distograms_batch,
                masks_batch,
                batch_ids=batch_ids,
                batch_idx=batch_idx,
                ontology=ontology,
                model_type="fusion",
            )
            inference_elapsed += time.perf_counter() - inference_start

            batch_records = [
                {
                    "protein_id": protein_id,
                    "ontology": ontology,
                    "pred_proba": preds[i],
                    "pred_proba_struct": preds_struct[i],
                    "pred_proba_seq": preds_seq[i],
                    "pred_gate": gate[i],
                }
                for i, protein_id in enumerate(batch_ids)
            ]
            probabilities_by_aspect[ontology] = {
                record["protein_id"]: record["pred_proba"] for record in batch_records
            }
            propagated_batch_records = (
                [propagate_prediction_record(record, descendant_indices_by_ontology[ontology])
                 for record in batch_records]
                if prop
                else None
            )

            if not summary_only:
                for i, record in enumerate(batch_records):
                    build_all_prediction_table([record], mapping).to_csv(
                        preds_dir / f"{record['protein_id']}__{ontology}.csv", index=False
                    )
                    if prop:
                        build_propagated_prediction_table([propagated_batch_records[i]], mapping).to_csv(
                            preds_propagated_dir / f"{record['protein_id']}__{ontology}.csv", index=False
                        )

            summary, _ = build_prediction_summary(
                batch_records,
                mapping,
                threshold=threshold,
                top_k=top_k,
                go_name_map=go_name_map,
                propagated_records=propagated_batch_records,
            )
            summary.insert(0, "ontology", ontology)
            batch_summaries.append(summary)

        batch_summaries = [frame for frame in batch_summaries if not frame.empty]
        if batch_summaries:
            batch_summary = pd.concat(batch_summaries, ignore_index=True)
            batch_summary.to_csv(prediction_summary_path, mode="a",
                                 header=not prediction_header_written, index=False)
            prediction_header_written = True

        # --- interpretability reports --------------------------------------------------------
        # With an explicit term list one pass suffices: every listed term is resolved to the aspect
        # that defines it (and analyzed with that aspect's model). Otherwise each aspect selects and
        # explains its own above-threshold terms.
        passes = [(aspects[0], custom_terms_by_protein)] if custom_terms_by_protein else [
            (ontology, None) for ontology in aspects
        ]
        for primary, custom_terms in passes:
            probabilities = probabilities_by_aspect[primary]
            records_with_predictions = [
                {**record, "pred_proba": probabilities[record["protein_id"]]} for record in records
            ]
            interpretability_start = time.perf_counter()
            report_summary, _ = analyze_records_with_interpretability(
                models[primary],
                records_with_predictions,
                mappings_by_aspect[primary],
                output_dir=output_dir,
                threshold=threshold_fs,
                max_terms_per_protein=options.max_terms,
                top_k_fallback=options.top_k_fallback,
                smooth_window=options.smooth_window,
                top_windows=options.top_windows,
                top_residues=options.top_residues,
                keep_in_memory=False,
                go_name_map=go_name_map,
                custom_terms_by_protein=custom_terms,
                custom_true_residues=custom_true_residues,
                log_runtime=log_runtime,
                save_in_background=options.save_workers > 0,
                save_workers=max(1, options.save_workers),
                models_by_ontology=models,
                go_terms_mappings_by_ontology=mappings_by_aspect,
                write_summary=False,
            )
            interpretability_elapsed += time.perf_counter() - interpretability_start

            if not report_summary.empty:
                report_summary.insert(
                    0, "ontology",
                    report_summary["go_term"].map(aspect_by_go_term).fillna(primary),
                )
                report_summary.to_csv(interpretability_summary_path, mode="a",
                                      header=not interpretability_header_written, index=False)
                interpretability_header_written = True
                total_reports += len(report_summary)

    # Keep both files present even when nothing passed selection, so the log's paths always resolve.
    if not prediction_header_written:
        prediction_summary_path.touch()
    if not interpretability_header_written:
        interpretability_summary_path.touch()

    log_timing("Inference time", inference_elapsed, total_proteins, "protein")
    log_timing("Interpretability time", interpretability_elapsed, total_reports, "report")
    log_timing("Total time", time.perf_counter() - start, total_proteins, "protein")
    if summary_only:
        logger.info("Summary-only mode: per-protein prediction tables not written")
    elif prop:
        logger.info(f"Wrote per-protein tables to {preds_dir} and {preds_propagated_dir}")
    else:
        logger.info(f"Wrote per-protein tables to {preds_dir}")
    logger.info(f"Wrote predictions summary to {prediction_summary_path} for {total_proteins} protein(s)")
    logger.info(f"Wrote {total_reports} report(s) to {output_dir}, indexed in {interpretability_summary_path}")
    return prediction_summary_path, interpretability_summary_path


# ===
# CLI
# ===

def _positive_int(value):
    """argparse type for a strictly positive int."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got: {value!r}")
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got: {number}")
    return number


def _non_negative_int(value):
    """argparse type for an int >= 0."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got: {value!r}")
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got: {number}")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="deepFRI2 interpretability: explain predicted GO terms residue by residue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, required=True, dest="input",
                        help="Folder with protein structures (.cif / .pdb). A FASTA input is not "
                             "supported: the structural part of the report needs coordinates.")
    parser.add_argument("--output_dir", "-o", type=Path, default=None,
                        help="Folder for predictions and reports (default: <repo>/results).")
    parser.add_argument("--ids_file", "-f", type=Path, default=None,
                        help="Text file listing structures to run (one per line, "
                             "e.g. 'abCD.cif' or 'abCD'); default: all files in input_dir.")
    parser.add_argument("--aspect", "-a", type=parse_aspect, default=list(ONTOLOGIES),
                        metavar="MF,CC,BP",
                        help="Comma-separated GO aspects (ontologies) to run: any of MF, CC, BP "
                             "(case-insensitive). Default: mf,cc,bp.")
    parser.add_argument("--batch_size", "-b", type=_positive_int, default=DEFAULT_BATCH_SIZE,
                        help=f"Proteins per inference batch (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--threshold", "-t", type=parse_threshold, default=DEFAULT_THRESHOLD,
                        metavar="T | T_fs,T_struct",
                        help="Keep a GO term if any branch score >= threshold. One float applies to "
                             "all branches; two comma-separated floats apply to fusion/sequence and "
                             "structure respectively. The first value also selects which terms get a "
                             "report (default: 0.1,0.2).")
    parser.add_argument("--top_k", "-k", type=_positive_int, default=DEFAULT_TOP_K,
                        help="Max GO terms per protein in the prediction summary (default: all "
                             "selected). Reported terms are capped by --max_terms instead.")
    parser.add_argument("--prop", "-p", action="store_true",
                        help="Propagate scores up the GO hierarchy (default: off); also writes "
                             "the preds_propagated/ folder and the propagated summary columns.")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="Skip the per-protein preds/ (and preds_propagated/) folders "
                             "(default: off). Interpretability reports are written either way.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging and per-step timings for every analyzed term.")

    group = parser.add_argument_group(
        "interpretability options",
        "Rarely changed; the defaults reproduce the reports shipped with the paper.",
    )
    group.add_argument("--max_terms", type=_positive_int, default=DEFAULT_MAX_TERMS,
                       help=f"Max reported GO terms per protein and aspect (default: {DEFAULT_MAX_TERMS}).")
    group.add_argument("--top_k_fallback", type=_positive_int, default=DEFAULT_TOP_K_FALLBACK,
                       help="Reported terms when no term passes the threshold "
                            f"(default: {DEFAULT_TOP_K_FALLBACK}).")
    group.add_argument("--smooth_window", type=_positive_int, default=DEFAULT_SMOOTH_WINDOW,
                       help="Moving-average window for the per-residue curves; 1 means no "
                            f"smoothing (default: {DEFAULT_SMOOTH_WINDOW}).")
    group.add_argument("--top_windows", type=_positive_int, default=DEFAULT_TOP_WINDOWS,
                       help=f"Kernel windows kept per report (default: {DEFAULT_TOP_WINDOWS}).")
    group.add_argument("--top_residues", type=_positive_int, default=DEFAULT_TOP_RESIDUES,
                       help=f"Residues kept in top_residues.csv (default: {DEFAULT_TOP_RESIDUES}).")
    group.add_argument("--save_workers", type=_non_negative_int, default=DEFAULT_SAVE_WORKERS,
                       help="Threads saving a report while the next term is analyzed; 0 saves "
                            f"inline (default: {DEFAULT_SAVE_WORKERS}).")
    group.add_argument("--go_terms", type=Path, default=None,
                       help="CSV/TSV with protein_id and go_term columns listing the terms to "
                            "analyze per protein (instead of selecting them by threshold).")
    group.add_argument("--true_residues", type=Path, default=None,
                       help="CSV/TSV with protein_id, go_term and activity_positions (or "
                            "binding_positions / positions) columns; the "
                            "reference residues are drawn on the report plots for comparison.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from utils import build_go_descendant_indices, configure_log_file

    log_level = DEBUG if args.verbose else INFO
    basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    input_dir = args.input
    output_dir = args.output_dir if args.output_dir is not None else REPO_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_log_file(output_dir / "log.txt", level=log_level)

    # Record the exact command that launched this run, so log.txt is self-contained.
    logger.info(f"Command          : {shlex.join([sys.executable, *sys.argv])}")

    if not input_dir.exists():
        _fatal(f"Input does not exist: {input_dir}")
    if input_dir.is_file():
        _fatal(f"--input must be a directory of structures, got a file: {input_dir}. "
               f"Interpretability needs coordinates, so a FASTA input is not supported.")

    logger.info("Loading torch...")
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    top_k = args.top_k if args.top_k is not None else "all"
    threshold_fs, threshold_struct = args.threshold
    aspects = args.aspect
    logger.info(f"Model version    : {config_version()}")
    logger.info(f"Input            : {input_dir}")
    logger.info(f"Output dir       : {output_dir}")
    logger.info(f"IDs file         : {args.ids_file if args.ids_file is not None else 'None (all structures in input dir)'}")
    logger.info("Model            : fusion (interpretability explains both branches)")
    logger.info(f"Threshold        : fusion/seq={threshold_fs}, struct={threshold_struct}")
    logger.info(f"Aspects          : {', '.join(aspects)}")
    logger.info(f"Batch size       : {args.batch_size}")
    logger.info(f"Top k            : {top_k}")
    logger.info(f"Propagate        : {args.prop}")
    logger.info(f"Summary only     : {args.summary}")
    logger.info(f"Verbose          : {args.verbose}")
    logger.info(f"Device           : {device}")
    logger.info(f"Max terms        : {args.max_terms} per protein and aspect "
                f"(fallback: {args.top_k_fallback} best terms)")
    logger.info(f"Report detail    : smooth_window={args.smooth_window}, "
                f"top_windows={args.top_windows}, top_residues={args.top_residues}")
    logger.info(f"Save workers     : {args.save_workers}")

    validate_input_dir(input_dir, require_structures=args.ids_file is None)
    file_names = resolve_file_names(input_dir, args.ids_file)

    # Benchmark inputs are read before any heavy work so a malformed table fails fast.
    custom_terms_by_protein = load_custom_go_terms(args.go_terms) if args.go_terms else None
    custom_true_residues = load_true_residues(args.true_residues) if args.true_residues else None
    if custom_true_residues and not custom_terms_by_protein:
        logger.warning("--true_residues without --go_terms: reference residues are only drawn for "
                       "terms that happen to be selected by threshold.")

    go_terms_mappings = load_go_terms_mappings(PARAMS_DIR)
    num_labels_by_ontology = {ontology: len(mapping) for ontology, mapping in go_terms_mappings.items()}

    tokenizer, esm_model = load_esm(device, PARAMS_DIR)
    models = load_models(device, num_labels_by_ontology, PARAMS_DIR, ontologies=aspects,
                         model_type="fusion")

    go_graph, go_name_map = load_go_name_map(PARAMS_DIR / f"go_{GO_VERSION}.obo")
    descendant_indices_by_ontology = (
        {ontology: build_go_descendant_indices(go_terms_mappings[ontology], go_graph)
         for ontology in aspects}
        if args.prop
        else None
    )

    run_interpretability(
        input_dir, output_dir, file_names, models, tokenizer, esm_model, device,
        go_terms_mappings, descendant_indices_by_ontology, go_name_map,
        batch_size=args.batch_size, threshold=args.threshold, top_k=args.top_k,
        aspects=aspects, options=args, prop=args.prop, summary_only=args.summary,
        # Per-step timings are one line per attribution stage per term, so they follow -v; the same
        # numbers always land in interpretability_summary.csv and each report's summary.json.
        log_runtime=args.verbose,
        custom_terms_by_protein=custom_terms_by_protein,
        custom_true_residues=custom_true_residues,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
