"""Per-GO-term interpretability for deepFRI2 predictions.

Given one protein and one predicted GO term, ``analyze_go_term`` attributes the fusion model's
score back to individual residues and residue–residue contacts, combining several signals:

- sequence branch: pooling attention, grad x input and integrated gradients on the ESM embeddings;
- structure branch: grad x input and integrated gradients on the distogram (full map, plus the
  diagonal-band and off-diagonal kernel groups separately), and the per-window kernel scores
  weighted by their gradients;
- the two branches are merged with the model's own gate, so a term dominated by the sequence
  branch is explained mostly by sequence signals and vice versa.

The per-residue curves are then reduced to a activity-site propensity from which contiguous
candidate sites are extracted. ``analyze_records_with_interpretability`` drives this over a list
of protein records and writes one report bundle per protein/GO-term pair
(plots, residue tables, candidate sites, a 3D structure viewer and a summary JSON) via
``save_analysis_bundle``; see ``interpret.py`` for the CLI that feeds it.

Attribution is computed in float32 with AMP disabled (see ``analyze_go_term``), so numbers are
stable but the per-term cost is dominated by the integrated-gradients steps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch, Rectangle

from utils import ROOT_GO_IDS, _resolve_go_name, _resolve_go_term

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency, only used as a 3D-viewer fallback
    go = None


EPS = 1e-8
LOGGER = logging.getLogger("inference")
_COOL_START = np.array([43, 166, 154], dtype=np.float32)
_COOL_END = np.array([103, 93, 163], dtype=np.float32)
_HOT_LOW = np.array([255, 196, 61], dtype=np.float32)
_HOT_HIGH = np.array([179, 35, 23], dtype=np.float32)

# ------------------------------------------------------------------------------------------------
# Weights of the aggregate measures
#
# Each ingredient is normalized to its own maximum before it is weighted (the raw signals live on
# unrelated scales), so within every group the weights are shares of 1 and can be read directly as
# "how much this signal counts". Across the two *branches* a weighted sum is the wrong rule
# altogether -- the two curves have incomparable shape, not just scale -- so the gate merges them
# multiplicatively; see the geometric merge in `analyze_go_term`.
#
# The values come from redistributing the weights of the retired integrated-gradients terms over
# the signals that remain, preserving the relative ordering the original weighting expressed:
#   * within a branch combo, the surviving weights were renormalized to sum to 1
#     (sequence 0.25 : 0.15 -> 0.65 : 0.35; structure 0.35 : 0.35 : 0.15 -> 0.40 : 0.40 : 0.20);
#   * in the activity score, each integrated-gradients term was replaced by its grad x input
#     counterpart at the same weight -- both attribute the same logit to the same input, so the
#     eight-signal balance is unchanged.
# ------------------------------------------------------------------------------------------------

# How much a near-zero branch can veto the other in the geometric merge. Each branch combo is in
# [0, 1], so log(x + floor) bottoms out at log(floor): a smaller floor makes the merge more strictly
# "both branches must agree", a larger one closer to an average. 1e-4 was the value benchmarked.
BRANCH_MERGE_FLOOR = 1e-4

# Sequence branch: grad x input on the ESM embeddings, plus the pooling attention the model reports.
# Attention is not an attribution (it says where the model looked, not what changed the score), so
# it stays the minority term.
SEQ_COMBO_WEIGHTS = {"grad_x_input": 0.65, "attention": 0.35}

# Structure branch: pair-level grad x input on the distogram, window-level kernel score x gradient,
# and the model's own per-residue kernel attributions.
STRUCT_COMBO_WEIGHTS = {"grad_x_input": 0.40, "kernel_score_x_grad": 0.40, "kernel_attr": 0.20}

# Signed structure combination: the pair-level signal is sharper than the window-level one, which is
# smeared over a whole window span, hence the larger share.
STRUCT_SIGNED_WEIGHTS = {"grad_x_input": 0.60, "kernel_score_x_grad": 0.40}

# Functional-activity score: the multi-signal heuristic ranking residues as candidate activity
# sites. It deliberately re-uses signals already inside `combined_abs` (as the original did), so a
# residue supported by several independent signals outranks one carried by a single strong curve.
ACTIVITY_SCORE_WEIGHTS = {
    "combined_abs": 0.30,          # the gate-weighted fusion of both branches
    "combined_signed_positive": 0.18,  # only score-increasing evidence
    "seq_grad_abs": 0.14,          # sequence branch, residue level
    "struct_grad_abs": 0.14,       # structure branch, residue level
    "kernel_abs_total": 0.10,      # structure branch, window level
    "kernel_overlap_weight": 0.08,  # how strongly top kernel windows cover the residue
    "kernel_overlap_count": 0.04,  # how many of them do
    "seq_attention": 0.02,         # weakest evidence, kept as a tie-breaker
}


class HtmlFigure:
    def __init__(self, html: str):
        self.html = str(html)

    def to_html(self, *args, **kwargs) -> str:
        return self.html

    def write_html(self, path: str | Path, *args, **kwargs) -> None:
        Path(path).write_text(self.html, encoding="utf-8")

    def _repr_html_(self) -> str:
        return self.html


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def _rgb_to_hex(rgb: np.ndarray) -> str:
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _lerp_color(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    return start + (end - start) * t


def _log_runtime_step(
    enabled: bool,
    protein_id: str,
    go_term: str,
    step: str,
    elapsed_s: float,
) -> None:
    if not enabled:
        return
    LOGGER.info(
        "Interpretability | %s | %s | %-20s | %7.3fs",
        protein_id,
        go_term,
        step,
        float(elapsed_s),
    )


def _colorize_residue_cartoon(scores: np.ndarray) -> tuple[list[str], list[str]]:
    scores = _normalize_positive(scores)
    residue_count = int(scores.shape[0])
    if residue_count <= 0:
        return [], []

    cartoon_colors: list[str] = []
    hotspot_colors: list[str] = []
    denom = max(1, residue_count - 1)
    for idx, score in enumerate(scores.tolist()):
        pos = idx / float(denom)
        base_rgb = _lerp_color(_COOL_START, _COOL_END, pos)
        hot_rgb = _lerp_color(_HOT_LOW, _HOT_HIGH, score)
        blend = np.clip((float(score) - 0.42) / 0.58, 0.0, 1.0) ** 0.9
        cartoon_rgb = _lerp_color(base_rgb, hot_rgb, 0.8 * blend)
        cartoon_colors.append(_rgb_to_hex(cartoon_rgb))
        hotspot_colors.append(_rgb_to_hex(hot_rgb))
    return cartoon_colors, hotspot_colors


def _smooth1d(values: np.ndarray, window: int) -> np.ndarray:
    if window is None or int(window) <= 1:
        return np.asarray(values, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    vmax = float(np.max(values)) if values.size else 0.0
    if vmax <= 0.0 or not np.isfinite(vmax):
        return np.zeros_like(values, dtype=np.float32)
    return values / vmax


def _normalize_signed(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    vmax = float(np.max(np.abs(values))) if values.size else 0.0
    if vmax <= 0.0 or not np.isfinite(vmax):
        return np.zeros_like(values, dtype=np.float32)
    return values / vmax


def _close_small_gaps(mask: np.ndarray, max_gap: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if mask.size == 0 or int(max_gap) <= 0:
        return mask

    idx = 0
    length = int(mask.size)
    while idx < length:
        if mask[idx]:
            idx += 1
            continue
        gap_start = idx
        while idx < length and not mask[idx]:
            idx += 1
        gap_end = idx
        if (
            gap_start > 0
            and gap_end < length
            and mask[gap_start - 1]
            and mask[gap_end]
            and (gap_end - gap_start) <= int(max_gap)
        ):
            mask[gap_start:gap_end] = True
    return mask


def _iter_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    segments: list[tuple[int, int]] = []
    idx = 0
    length = int(mask.size)
    while idx < length:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < length and mask[idx]:
            idx += 1
        segments.append((start, idx))
    return segments


def _segment_sequence_string(residue_df: pd.DataFrame, start: int, end: int) -> str:
    if "residue_aa" not in residue_df.columns:
        return ""
    aas = residue_df["residue_aa"].astype(str).tolist()[int(start):int(end)]
    return "".join(aa if len(aa) == 1 else "X" for aa in aas)


def _build_activity_site_candidates(
    analysis: dict[str, Any],
    *,
    max_segments: int = 5,
    gap_tolerance: int = 1,
    min_segment_len: int = 2,
) -> tuple[np.ndarray, np.ndarray, float, pd.DataFrame, np.ndarray]:
    """Build the activity score and call contiguous candidate sites from it.

    Returns ``(additive_score, multiplicative_score, threshold, candidates, segment_labels)``.

    The **additive** score -- the weighted sum of the normalized ingredients -- is what calls the
    segments, and is the score the reports have always used. The **multiplicative** score is the same
    ingredients under the same weights combined as a weighted geometric mean, i.e. an AND-like rule:
    a residue needs support from several signals rather than one strong one. It is computed and
    plotted alongside for comparison because the same substitution at the *branch* merge measurably
    improved localization (see ``benchmark.md`` §6-§7); whether it also helps here is open, so it does
    not yet drive the segments.
    """
    residue_count = int(analysis["residue_count"])
    if residue_count <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, 0.0, pd.DataFrame(), np.array([], dtype=object)

    # Every ingredient normalized to its own maximum, then weighted by ACTIVITY_SCORE_WEIGHTS.
    components = {
        "combined_abs": analysis["combined_abs"],
        "combined_signed_positive": np.clip(
            np.asarray(analysis["combined_signed"], dtype=np.float32), 0.0, None
        ),
        "seq_grad_abs": analysis["seq_grad_abs"],
        "struct_grad_abs": analysis["kernel_grad_x_input_full_abs"],
        "kernel_abs_total": analysis["kernel_abs_total"],
        "kernel_overlap_weight": analysis["kernel_overlap_weight"],
        "kernel_overlap_count": analysis["kernel_overlap_count"],
        "seq_attention": analysis["seq_attention"],
    }
    normalized = {name: _normalize_positive(values) for name, values in components.items()}
    raw_score = np.zeros(residue_count, dtype=np.float32)
    log_score = np.zeros(residue_count, dtype=np.float32)
    for name, weight in ACTIVITY_SCORE_WEIGHTS.items():
        raw_score = raw_score + np.float32(weight) * normalized[name]
        log_score = log_score + np.float32(weight) * np.log(normalized[name] + BRANCH_MERGE_FLOOR)
    smooth_window = 3 if residue_count < 80 else 5
    if residue_count >= 240:
        smooth_window = 7
    smoothed_score = _normalize_positive(_smooth1d(raw_score, smooth_window))
    multiplicative_score = _normalize_positive(_smooth1d(np.exp(log_score), smooth_window))

    peak_value = float(np.max(smoothed_score)) if smoothed_score.size else 0.0
    if peak_value <= 0.0 or not np.isfinite(peak_value):
        return (smoothed_score, multiplicative_score, 0.0, pd.DataFrame(),
                np.array([""] * residue_count, dtype=object))

    quantile_level = 0.75 if residue_count < 25 else 0.82
    score_threshold = max(
        0.55 * peak_value,
        float(np.quantile(smoothed_score, quantile_level)),
        float(np.mean(smoothed_score) + 0.50 * np.std(smoothed_score)),
    )
    score_threshold = min(score_threshold, peak_value)

    mask = _close_small_gaps(smoothed_score >= score_threshold, max_gap=gap_tolerance)
    segments = _iter_true_segments(mask)
    if not segments:
        peak_idx = int(np.argmax(smoothed_score))
        local_floor = 0.70 * peak_value
        start = peak_idx
        end = peak_idx + 1
        while start > 0 and float(smoothed_score[start - 1]) >= local_floor:
            start -= 1
        while end < residue_count and float(smoothed_score[end]) >= local_floor:
            end += 1
        segments = [(start, end)]

    filtered_segments: list[tuple[int, int]] = []
    for start, end in segments:
        segment_peak = float(np.max(smoothed_score[start:end]))
        if (end - start) >= int(min_segment_len) or segment_peak >= max(0.82 * peak_value, score_threshold * 1.10):
            filtered_segments.append((start, end))
    if not filtered_segments:
        filtered_segments = [max(segments, key=lambda seg: float(np.max(smoothed_score[seg[0]:seg[1]])))]

    residue_df = analysis["residue_df"].reset_index(drop=True)
    rows = []
    for start, end in filtered_segments:
        segment_scores = smoothed_score[start:end]
        peak_offset = int(np.argmax(segment_scores))
        peak_idx = start + peak_offset
        peak_row = residue_df.iloc[peak_idx]
        rows.append(
            {
                "start_index_1based": int(start + 1),
                "end_index_1based": int(end),
                "length": int(end - start),
                "peak_residue_index_1based": int(peak_idx + 1),
                "peak_residue_id": int(peak_row.get("residue_id", peak_idx + 1)),
                "peak_residue_aa": str(peak_row.get("residue_aa", "?")),
                "peak_activity_score": float(smoothed_score[peak_idx]),
                "mean_activity_score": float(np.mean(segment_scores)),
                "sum_activity_score": float(np.sum(segment_scores)),
                "mean_overlap_weight": float(np.mean(np.asarray(analysis["kernel_overlap_weight"], dtype=np.float32)[start:end])),
                "max_overlap_count": float(np.max(np.asarray(analysis["kernel_overlap_count"], dtype=np.float32)[start:end])),
                "mean_seq_grad_abs": float(np.mean(np.asarray(analysis["seq_grad_abs"], dtype=np.float32)[start:end])),
                "mean_struct_grad_abs": float(np.mean(np.asarray(analysis["kernel_grad_x_input_full_abs"], dtype=np.float32)[start:end])),
                "mean_kernel_abs": float(np.mean(np.asarray(analysis["kernel_abs_total"], dtype=np.float32)[start:end])),
                "segment_score": float(0.60 * np.max(segment_scores) + 0.40 * np.mean(segment_scores)),
                "sequence": _segment_sequence_string(residue_df, start, end),
            }
        )

    if rows:
        candidates = pd.DataFrame(rows).sort_values(
            ["segment_score", "peak_activity_score", "sum_activity_score"],
            ascending=False,
        ).head(int(max_segments)).reset_index(drop=True)
        candidates["rank"] = np.arange(1, len(candidates) + 1, dtype=int)
        candidates["segment_label"] = [f"AS{i}" for i in candidates["rank"].tolist()]
        ordered_cols = [
            "rank",
            "segment_label",
            "start_index_1based",
            "end_index_1based",
            "length",
            "peak_residue_index_1based",
            "peak_residue_id",
            "peak_residue_aa",
            "peak_activity_score",
            "mean_activity_score",
            "sum_activity_score",
            "mean_overlap_weight",
            "max_overlap_count",
            "mean_seq_grad_abs",
            "mean_struct_grad_abs",
            "mean_kernel_abs",
            "segment_score",
            "sequence",
        ]
        candidates = candidates[ordered_cols]
    else:
        candidates = pd.DataFrame(
            columns=[
                "rank",
                "segment_label",
                "start_index_1based",
                "end_index_1based",
                "length",
                "peak_residue_index_1based",
                "peak_residue_id",
                "peak_residue_aa",
                "peak_activity_score",
                "mean_activity_score",
                "sum_activity_score",
                "mean_overlap_weight",
                "max_overlap_count",
                "mean_seq_grad_abs",
                "mean_struct_grad_abs",
                "mean_kernel_abs",
                "segment_score",
                "sequence",
            ]
        )

    segment_labels = np.array([""] * residue_count, dtype=object)
    for _, row in candidates.iterrows():
        start = int(row["start_index_1based"]) - 1
        end = int(row["end_index_1based"])
        segment_labels[start:end] = str(row["segment_label"])

    return (smoothed_score, multiplicative_score, float(score_threshold), candidates,
            segment_labels)


def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.unsqueeze(0) if tensor.ndim in (1, 2) else tensor


def _valid_token_mask(
    mask1: torch.Tensor | None,
    token_count: int,
    details: dict[str, Any] | None = None,
) -> np.ndarray:
    if details is not None:
        token_mask = details.get("token_mask")
        if token_mask is not None:
            return _as_numpy(token_mask[0]).astype(bool)
    if mask1 is None:
        return np.ones(token_count, dtype=bool)
    residue_mask = _as_numpy(mask1[0]).astype(bool)
    residue_count = int(residue_mask.sum())
    token_mask = np.zeros(int(token_count), dtype=bool)
    if token_mask.size > 0:
        token_mask[0] = True
        upper = min(token_mask.size, residue_count + 1)
        token_mask[1:upper] = True
    return token_mask


def _infer_token_slice(valid_token_count: int, residue_count: int) -> slice:
    if valid_token_count >= residue_count + 1:
        return slice(1, 1 + residue_count)
    return slice(0, residue_count)


def _trim_token_curve(values: np.ndarray, valid_tokens: np.ndarray, residue_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid_tokens = np.asarray(valid_tokens, dtype=bool)
    if valid_tokens.size == values.shape[0]:
        trimmed = values[valid_tokens]
        if trimmed.shape[0] > residue_count:
            trimmed = trimmed[-residue_count:]
    else:
        valid_count = int(valid_tokens.sum()) if valid_tokens.size else len(values)
        token_slice = _infer_token_slice(valid_count, residue_count)
        trimmed = values[token_slice]
    if trimmed.shape[0] >= residue_count:
        return trimmed[:residue_count].astype(np.float32, copy=False)
    out = np.zeros(residue_count, dtype=np.float32)
    out[: trimmed.shape[0]] = trimmed.astype(np.float32, copy=False)
    return out


def _collapse_pair_matrix_to_residues(matrix: np.ndarray, residue_count: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)[:residue_count, :residue_count]
    signed = 0.5 * (matrix.sum(axis=1) + matrix.sum(axis=0))
    absv = 0.5 * (np.abs(matrix).sum(axis=1) + np.abs(matrix).sum(axis=0))
    return signed.astype(np.float32), absv.astype(np.float32)


def _kernel_visibility_masks(
    model,
    dist1: torch.Tensor,
    mask1: torch.Tensor | None,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    kernel_model = getattr(model, "kernel", model)
    device = dist1.device
    dtype = dist1.dtype
    N = int(dist1.shape[-1])

    if mask1 is not None:
        rv_np = _as_numpy(mask1[0]).astype(bool)
    else:
        rv_np = np.ones(N, dtype=bool)
    rv_t = torch.as_tensor(rv_np, device=device, dtype=dtype)
    pair_valid = rv_t[:, None] * rv_t[None, :]

    diag_m = int(getattr(kernel_model, "canonical_diag_ms", 1))
    diag_bandwidth = getattr(kernel_model, "diag_bandwidth", None)
    if diag_bandwidth is None:
        diag_bandwidth = 2 * (diag_m - 1) + 1
    half_w = (int(diag_bandwidth) - 1) // 2
    idx = torch.arange(N, device=device)
    ii, jj = torch.meshgrid(idx, idx, indexing="ij")
    Wdiag = ((jj - ii).abs() <= half_w).to(dtype=dtype)
    Wanti = (jj > ii).to(dtype=dtype)
    return rv_np, pair_valid, Wdiag, Wanti


def _aggregate_window_values_to_residues(
    values: np.ndarray,
    starts: np.ndarray,
    window_size: int,
    residue_count: int,
) -> np.ndarray:
    out = np.zeros(residue_count, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    for value, (r0, c0) in zip(values, starts, strict=False):
        if not np.isfinite(value) or abs(float(value)) <= 1e-12:
            continue
        r0 = int(r0)
        c0 = int(c0)
        r1 = min(residue_count, r0 + int(window_size))
        c1 = min(residue_count, c0 + int(window_size))
        if r0 < residue_count:
            out[r0:r1] += float(value) / max(1, r1 - r0)
        if c0 < residue_count:
            out[c0:c1] += float(value) / max(1, c1 - c0)
    return out


def _select_target_logits(model_out: Any, target_branch: str = "fused") -> torch.Tensor:
    if not isinstance(model_out, tuple):
        return model_out

    logits = model_out[0]
    details = model_out[1] if len(model_out) > 1 and isinstance(model_out[1], dict) else None
    if target_branch == "struct" and details is not None and isinstance(details.get("logits_struct"), torch.Tensor):
        return details["logits_struct"]
    if target_branch == "esm" and details is not None and isinstance(details.get("logits_esm"), torch.Tensor):
        return details["logits_esm"]
    return logits


def _normalized_patch_kernel_corr(
    dist_square: np.ndarray,
    kernel_square: np.ndarray,
    start: tuple[int, int],
    window_size: int,
    residue_count: int,
) -> float:
    r0, c0 = int(start[0]), int(start[1])
    r1 = min(residue_count, r0 + int(window_size))
    c1 = min(residue_count, c0 + int(window_size))
    patch = np.zeros((window_size, window_size), dtype=np.float32)
    patch[: r1 - r0, : c1 - c0] = dist_square[r0:r1, c0:c1]

    patch_flat = patch.reshape(-1).astype(np.float32)
    kernel_flat = np.asarray(kernel_square, dtype=np.float32).reshape(-1)
    patch_flat = patch_flat - patch_flat.mean()
    kernel_flat = kernel_flat - kernel_flat.mean()
    patch_std = float(patch_flat.std())
    kernel_std = float(kernel_flat.std())
    if patch_std <= EPS or kernel_std <= EPS:
        return 0.0
    patch_flat = patch_flat / patch_std
    kernel_flat = kernel_flat / kernel_std
    return float(np.dot(patch_flat, kernel_flat) / patch_flat.size)


def _kernel_window_dataframe(
    branch_name: str,
    branch_details: dict[str, Any] | None,
    branch_grad: torch.Tensor | None,
    dist_square: np.ndarray,
    residue_count: int,
    top_windows: int,
) -> pd.DataFrame:
    if not branch_details or branch_details.get("disabled"):
        return pd.DataFrame()
    if branch_grad is None:
        return pd.DataFrame()

    scores = _as_numpy(branch_details["scores"])[0]
    grads = _as_numpy(branch_grad)[0]
    starts = _as_numpy(branch_details["starts"])
    kernel_bank = _as_numpy(branch_details["kernel_bank"])[:, 0]
    arch_ids = list(branch_details["arch_ids"])
    window_size = int(branch_details["window_size"])

    contrib = scores * grads
    flat = np.abs(contrib).reshape(-1)
    if flat.size == 0:
        return pd.DataFrame()

    n_take = min(int(top_windows), flat.size)
    idxs = np.argpartition(flat, -n_take)[-n_take:]
    idxs = idxs[np.argsort(flat[idxs])[::-1]]

    rows = []
    for rank, flat_idx in enumerate(idxs, start=1):
        kernel_idx, window_idx = np.unravel_index(int(flat_idx), contrib.shape)
        start = starts[window_idx]
        score = float(scores[kernel_idx, window_idx])
        grad = float(grads[kernel_idx, window_idx])
        value = float(contrib[kernel_idx, window_idx])
        corr = _normalized_patch_kernel_corr(
            dist_square, kernel_bank[kernel_idx], start, window_size, residue_count
        )
        r0, c0 = int(start[0]), int(start[1])
        rows.append(
            {
                "rank": rank,
                "branch": branch_name,
                "kernel_idx": int(kernel_idx),
                "kernel_name": arch_ids[kernel_idx],
                "window_idx": int(window_idx),
                "row_start_1based": r0 + 1,
                "row_end_1based": min(r0 + window_size, residue_count),
                "col_start_1based": c0 + 1,
                "col_end_1based": min(c0 + window_size, residue_count),
                "score": score,
                "grad": grad,
                "grad_x_score": value,
                "abs_grad_x_score": abs(value),
                "patch_kernel_corr": corr,
            }
        )
    return pd.DataFrame(rows)


def _top_residue_table(
    analysis: dict[str, Any],
    top_residues: int,
) -> pd.DataFrame:
    residue_df = analysis["residue_table"].copy()
    residue_df["seq_grad_abs"] = analysis["seq_grad_abs"]
    residue_df["struct_grad_abs"] = analysis["kernel_grad_x_input_full_abs"]
    residue_df["kernel_contrib_abs"] = analysis["kernel_abs_total"]
    residue_df["kernel_overlap_weight"] = analysis["kernel_overlap_weight"]
    residue_df["kernel_overlap_count"] = analysis["kernel_overlap_count"]
    residue_df["attention"] = analysis["seq_attention"]
    cols = [
        "rank",
        "residue_index_1based",
        "residue_id",
        "residue_aa",
        "combined_abs",
        "combined_signed",
        "seq_grad_abs",
        "struct_grad_abs",
        "kernel_contrib_abs",
        "kernel_overlap_weight",
        "kernel_overlap_count",
        "activity_score",
        "activity_segment_label",
        "attention",
    ]
    if "chain" in residue_df.columns:
        cols.insert(2, "chain")
    return residue_df.sort_values("combined_abs", ascending=False).head(int(top_residues))[cols]


def _iter_kernel_window_rectangles(
    kernel_windows: pd.DataFrame | None,
    residue_count: int,
    *,
    mirror: bool = True,
) -> list[dict[str, Any]]:
    if kernel_windows is None or len(kernel_windows) == 0:
        return []

    rects: list[dict[str, Any]] = []
    for row in kernel_windows.itertuples(index=False):
        r0 = max(0, int(getattr(row, "row_start_1based")) - 1)
        r1 = min(int(residue_count), int(getattr(row, "row_end_1based")))
        c0 = max(0, int(getattr(row, "col_start_1based")) - 1)
        c1 = min(int(residue_count), int(getattr(row, "col_end_1based")))
        if r0 >= r1 or c0 >= c1:
            continue

        rank = int(getattr(row, "rank"))
        branch = str(getattr(row, "branch"))
        label = f"{'D' if branch == 'diag' else 'A'}{rank}"
        rect = {
            "branch": branch,
            "rank": rank,
            "label": label,
            "r0": r0,
            "r1": r1,
            "c0": c0,
            "c1": c1,
            "grad_x_score": float(getattr(row, "grad_x_score")),
            "abs_grad_x_score": float(getattr(row, "abs_grad_x_score")),
            "patch_kernel_corr": float(getattr(row, "patch_kernel_corr")),
            "kernel_name": str(getattr(row, "kernel_name")),
            "mirrored": False,
        }
        rects.append(rect)

        if mirror and (r0 != c0 or r1 != c1):
            mirror_rect = dict(rect)
            mirror_rect["r0"] = c0
            mirror_rect["r1"] = c1
            mirror_rect["c0"] = r0
            mirror_rect["c1"] = r1
            mirror_rect["mirrored"] = True
            rects.append(mirror_rect)
    return rects


def _build_kernel_overlap_maps(
    kernel_windows: pd.DataFrame | None,
    residue_count: int,
    *,
    mirror: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count_map = np.zeros((residue_count, residue_count), dtype=np.float32)
    weight_map = np.zeros((residue_count, residue_count), dtype=np.float32)
    signed_map = np.zeros((residue_count, residue_count), dtype=np.float32)
    if kernel_windows is None or len(kernel_windows) == 0:
        return count_map, weight_map, signed_map

    for rect in _iter_kernel_window_rectangles(kernel_windows, residue_count, mirror=mirror):
        r0, r1 = int(rect["r0"]), int(rect["r1"])
        c0, c1 = int(rect["c0"]), int(rect["c1"])
        count_map[r0:r1, c0:c1] += 1.0
        weight_map[r0:r1, c0:c1] += float(rect["abs_grad_x_score"])
        signed_map[r0:r1, c0:c1] += float(rect["grad_x_score"])
    return count_map, weight_map, signed_map


def _branch_dist_gradxinput_curves(
    model,
    embed1: torch.Tensor,
    dist1: torch.Tensor,
    mask1: torch.Tensor | None,
    term_idx: int,
    residue_count: int,
    *,
    use_diag: bool | None = None,
    use_anti: bool | None = None,
    W: torch.Tensor | None = None,
    target_branch: str = "fused",
) -> tuple[np.ndarray, np.ndarray]:
    with torch.enable_grad():
        dist_var = dist1.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        try:
            model_out = model(
                embed1,
                dist_var,
                mask1,
                return_attr=False,
                return_details=True,
                use_diag=use_diag,
                use_anti=use_anti,
            )
            logits = _select_target_logits(model_out, target_branch=target_branch)
        except TypeError:
            logits = model(embed1, dist_var, mask1, return_attr=False)
        target = logits[0, int(term_idx)]
        grad = torch.autograd.grad(target, dist_var, retain_graph=False, create_graph=False, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(dist_var)
    pair_gx = dist_var[0] * grad[0]
    if W is not None:
        pair_gx = pair_gx * W.to(dtype=pair_gx.dtype, device=pair_gx.device)
    pair_gx = _as_numpy(pair_gx)
    return _collapse_pair_matrix_to_residues(pair_gx, residue_count)

def _residue_report_table(analysis: dict[str, Any]) -> pd.DataFrame:
    residue_df = analysis["residue_df"].copy().reset_index(drop=True)
    residue_df["residue_index_1based"] = np.arange(1, len(residue_df) + 1)
    residue_df["esm_attention"] = analysis["seq_attention"]
    residue_df["esm_grad_x_input_signed"] = analysis["seq_grad_signed"]
    residue_df["esm_grad_x_input_abs"] = analysis["seq_grad_abs"]
    residue_df["kernel_attr_diag"] = analysis["kernel_attr_diag"]
    residue_df["kernel_attr_anti"] = analysis["kernel_attr_anti"]
    residue_df["kernel_attr_sum"] = analysis["kernel_attr_sum"]
    residue_df["kernel_grad_x_input_full_signed"] = analysis["kernel_grad_x_input_full_signed"]
    residue_df["kernel_grad_x_input_diag_signed"] = analysis["kernel_grad_x_input_diag_signed"]
    residue_df["kernel_grad_x_input_anti_signed"] = analysis["kernel_grad_x_input_anti_signed"]
    residue_df["kernel_grad_x_input_full_abs"] = analysis["kernel_grad_x_input_full_abs"]
    residue_df["kernel_grad_x_input_diag_abs"] = analysis["kernel_grad_x_input_diag_abs"]
    residue_df["kernel_grad_x_input_anti_abs"] = analysis["kernel_grad_x_input_anti_abs"]
    residue_df["kernel_signed_diag"] = analysis["kernel_signed_diag"]
    residue_df["kernel_signed_anti"] = analysis["kernel_signed_anti"]
    residue_df["kernel_signed_total"] = analysis["kernel_signed_total"]
    residue_df["kernel_abs_diag"] = analysis["kernel_abs_diag"]
    residue_df["kernel_abs_anti"] = analysis["kernel_abs_anti"]
    residue_df["kernel_abs_total"] = analysis["kernel_abs_total"]
    residue_df["kernel_overlap_weight"] = analysis["kernel_overlap_weight"]
    residue_df["kernel_overlap_count"] = analysis["kernel_overlap_count"]
    residue_df["combined_abs"] = analysis["combined_abs"]
    residue_df["combined_signed"] = analysis["combined_signed"]
    residue_df["activity_score"] = analysis.get("activity_score", np.zeros(len(residue_df), dtype=np.float32))
    residue_df["activity_score_multiplicative"] = analysis.get(
        "activity_score_multiplicative", np.zeros(len(residue_df), dtype=np.float32)
    )
    residue_df["activity_threshold"] = float(analysis.get("activity_threshold", 0.0))
    residue_df["activity_segment_label"] = analysis.get("activity_segment_labels", np.array([""] * len(residue_df), dtype=object))
    residue_df["rank"] = residue_df["combined_abs"].rank(method="first", ascending=False).astype(int)
    preferred_cols = [
        "rank",
        "residue_index_1based",
        "chain",
        "residue_id",
        "residue_aa",
        "esm_attention",
        "esm_grad_x_input_signed",
        "esm_grad_x_input_abs",
        "kernel_attr_diag",
        "kernel_attr_anti",
        "kernel_attr_sum",
        "kernel_grad_x_input_full_signed",
        "kernel_grad_x_input_diag_signed",
        "kernel_grad_x_input_anti_signed",
        "kernel_grad_x_input_full_abs",
        "kernel_grad_x_input_diag_abs",
        "kernel_grad_x_input_anti_abs",
        "kernel_signed_diag",
        "kernel_signed_anti",
        "kernel_signed_total",
        "kernel_abs_diag",
        "kernel_abs_anti",
        "kernel_abs_total",
        "kernel_overlap_weight",
        "kernel_overlap_count",
        "combined_abs",
        "combined_signed",
        "activity_score",
        "activity_score_multiplicative",
        "activity_threshold",
        "activity_segment_label",
    ]
    cols = [col for col in preferred_cols if col in residue_df.columns]
    other_cols = [col for col in residue_df.columns if col not in cols]
    return residue_df[cols + other_cols]


def analyze_go_term(
    model,
    record: dict[str, Any],
    term_idx: int,
    go_term: str,
    *,
    smooth_window: int = 1,
    top_windows: int = 12,
    top_residues: int = 20,
    log_runtime: bool = False,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    protein_id = str(record["protein_id"])
    sequence = str(record["sequence"])
    residue_count = int(record["residue_count"])
    residue_df = record["residue_df"].iloc[:residue_count].copy().reset_index(drop=True)
    timings_s: dict[str, float] = {}
    total_start = time.perf_counter()

    embed1 = _ensure_batch(record["padded_embedding"]).to(device=device, dtype=torch.float32)
    dist1 = _ensure_batch(record["padded_distogram"]).to(device=device, dtype=torch.float32)
    mask1 = _ensure_batch(record["mask"]).to(device=device, dtype=torch.float32) if record.get("mask") is not None else None

    original_amp = getattr(model, "amp_dtype", None)
    kernel_model = getattr(model, "kernel", None)
    original_kernel_amp = getattr(kernel_model, "amp_dtype", None) if kernel_model is not None else None
    if hasattr(model, "amp_dtype"):
        model.amp_dtype = None
    if kernel_model is not None and hasattr(kernel_model, "amp_dtype"):
        kernel_model.amp_dtype = None
    model.eval()

    try:
        step_start = time.perf_counter()
        with torch.enable_grad():
            embed_var = embed1.detach().clone().requires_grad_(True)
            dist_var = dist1.detach().clone().requires_grad_(True)

            logits, logits_struct, logits_esm, attr_diag, attr_anti, details = model(
                embed_var,
                dist_var,
                mask1,
                return_attr=True,
                return_details=True,
            )
            term_logit = logits[0, int(term_idx)]
            struct_term_logit = logits_struct[0, int(term_idx)]
            grad_embed = torch.autograd.grad(
                term_logit,
                embed_var,
                retain_graph=True,
                allow_unused=True,
            )[0]
            grad_dist = torch.autograd.grad(
                struct_term_logit,
                dist_var,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad_embed is None:
                grad_embed = torch.zeros_like(embed_var)
            if grad_dist is None:
                grad_dist = torch.zeros_like(dist_var)

            diag_scores = None if not details.get("diag") else details["diag"].get("scores")
            anti_scores = None if not details.get("anti") else details["anti"].get("scores")
            diag_grad = anti_grad = None
            grad_targets = []
            if diag_scores is not None:
                grad_targets.append(diag_scores)
            if anti_scores is not None:
                grad_targets.append(anti_scores)
            if grad_targets:
                extra_grads = torch.autograd.grad(
                    struct_term_logit,
                    grad_targets,
                    retain_graph=True,
                    allow_unused=True,
                )
                idx = 0
                if diag_scores is not None:
                    diag_grad = extra_grads[idx]
                    idx += 1
                if anti_scores is not None:
                    anti_grad = extra_grads[idx]
        timings_s["forward_and_grad"] = time.perf_counter() - step_start
        _log_runtime_step(log_runtime, protein_id, go_term, "forward_and_grad", timings_s["forward_and_grad"])

        valid_tokens = _valid_token_mask(mask1, embed1.shape[1], details=details)
        attn_weights = details.get("attn_weights")
        if attn_weights is None:
            seq_attention = np.zeros(residue_count, dtype=np.float32)
        else:
            seq_attention = _trim_token_curve(_as_numpy(attn_weights[0]), valid_tokens, residue_count)

        grad_embed_np = _as_numpy(grad_embed[0])
        embed_np = _as_numpy(embed1[0])
        seq_grad_signed = _trim_token_curve((embed_np * grad_embed_np).sum(axis=-1), valid_tokens, residue_count)
        seq_grad_abs = _trim_token_curve(np.abs(embed_np * grad_embed_np).sum(axis=-1), valid_tokens, residue_count)

        step_start = time.perf_counter()
        rv_np, pair_valid, Wdiag, Wanti = _kernel_visibility_masks(model, dist1, mask1)

        pair_gx = dist_var[0] * grad_dist[0]
        pair_gx = _as_numpy(pair_gx * pair_valid.to(dtype=pair_gx.dtype, device=pair_gx.device))
        kernel_gradinput_matrix = pair_gx[:residue_count, :residue_count].astype(np.float32)
        struct_grad_signed, struct_grad_abs = _collapse_pair_matrix_to_residues(pair_gx, residue_count)
        branch_grad_diag_signed, branch_grad_diag_abs = _branch_dist_gradxinput_curves(
            model,
            embed1,
            dist1,
            mask1,
            int(term_idx),
            residue_count,
            use_diag=True,
            use_anti=False,
            W=pair_valid * Wdiag,
            target_branch="struct",
        )
        branch_grad_anti_signed, branch_grad_anti_abs = _branch_dist_gradxinput_curves(
            model,
            embed1,
            dist1,
            mask1,
            int(term_idx),
            residue_count,
            use_diag=False,
            use_anti=True,
            W=pair_valid * Wanti,
            target_branch="struct",
        )
        timings_s["kernel_grad_x_input"] = time.perf_counter() - step_start
        _log_runtime_step(log_runtime, protein_id, go_term, "kernel_grad_x_input", timings_s["kernel_grad_x_input"])

        step_start = time.perf_counter()
        attr_diag_np = _as_numpy(attr_diag[0])[:residue_count].astype(np.float32, copy=False)
        attr_anti_np = _as_numpy(attr_anti[0])[:residue_count].astype(np.float32, copy=False)
        attr_sum_np = attr_diag_np + attr_anti_np

        dist_square = _as_numpy(dist1[0])[:residue_count, :residue_count].astype(np.float32, copy=False)

        kernel_signed_diag = np.zeros(residue_count, dtype=np.float32)
        kernel_abs_diag = np.zeros(residue_count, dtype=np.float32)
        if details.get("diag") and not details["diag"].get("disabled") and diag_grad is not None:
            diag_scores_np = _as_numpy(details["diag"]["scores"])[0]
            diag_grad_np = _as_numpy(diag_grad)[0]
            diag_signed_per_window = (diag_scores_np * diag_grad_np).sum(axis=0)
            diag_abs_per_window = np.abs(diag_scores_np * diag_grad_np).sum(axis=0)
            kernel_signed_diag = _aggregate_window_values_to_residues(
                diag_signed_per_window,
                _as_numpy(details["diag"]["starts"]),
                int(details["diag"]["window_size"]),
                residue_count,
            )
            kernel_abs_diag = _aggregate_window_values_to_residues(
                diag_abs_per_window,
                _as_numpy(details["diag"]["starts"]),
                int(details["diag"]["window_size"]),
                residue_count,
            )

        kernel_signed_anti = np.zeros(residue_count, dtype=np.float32)
        kernel_abs_anti = np.zeros(residue_count, dtype=np.float32)
        if details.get("anti") and not details["anti"].get("disabled") and anti_grad is not None:
            anti_scores_np = _as_numpy(details["anti"]["scores"])[0]
            anti_grad_np = _as_numpy(anti_grad)[0]
            anti_signed_per_window = (anti_scores_np * anti_grad_np).sum(axis=0)
            anti_abs_per_window = np.abs(anti_scores_np * anti_grad_np).sum(axis=0)
            kernel_signed_anti = _aggregate_window_values_to_residues(
                anti_signed_per_window,
                _as_numpy(details["anti"]["starts"]),
                int(details["anti"]["window_size"]),
                residue_count,
            )
            kernel_abs_anti = _aggregate_window_values_to_residues(
                anti_abs_per_window,
                _as_numpy(details["anti"]["starts"]),
                int(details["anti"]["window_size"]),
                residue_count,
            )

        kernel_signed_total = kernel_signed_diag + kernel_signed_anti
        kernel_abs_total = kernel_abs_diag + kernel_abs_anti

        struct_term_logit = float(logits_struct[0, int(term_idx)].detach().cpu())
        esm_term_logit = float(logits_esm[0, int(term_idx)].detach().cpu())
        gate_term = float(details["gate"][0, int(term_idx)].detach().cpu())
        seq_branch_weight = abs(gate_term * esm_term_logit)
        struct_branch_weight = abs(struct_term_logit)
        branch_total = seq_branch_weight + struct_branch_weight + EPS
        seq_weight = seq_branch_weight / branch_total
        struct_weight = struct_branch_weight / branch_total

        # Per-branch combination of the surviving attribution signals. Every ingredient is
        # normalized to its own maximum *before* weighting: the signals live on unrelated scales
        # (a summed grad x input over 1280 embedding channels vs. an attention mass vs. a kernel
        # score x gradient), so only after normalization do the weights below mean what they say.
        # Weights within a branch sum to 1 and keep the relative ordering the signals had when
        # integrated gradients still anchored these sums (see COMBINED_WEIGHTS).
        seq_combo = _normalize_positive(
            SEQ_COMBO_WEIGHTS["grad_x_input"] * _normalize_positive(seq_grad_abs)
            + SEQ_COMBO_WEIGHTS["attention"] * _normalize_positive(seq_attention)
        )
        struct_combo = _normalize_positive(
            STRUCT_COMBO_WEIGHTS["grad_x_input"] * _normalize_positive(struct_grad_abs)
            + STRUCT_COMBO_WEIGHTS["kernel_score_x_grad"] * _normalize_positive(kernel_abs_total)
            + STRUCT_COMBO_WEIGHTS["kernel_attr"] * _normalize_positive(np.abs(attr_sum_np))
        )
        # The gate decides how much each branch explains this term, so it also weights the merge --
        # but as a weighted *geometric* mean (a weighted sum in log space), not an arithmetic one.
        #
        # Why: the structure branch's per-residue curve is broad and high-baseline (it barely varies
        # between a site and the bulk), while the sequence branch is near-zero except for sharp peaks.
        # Added arithmetically, the structural curve contributes a flat pedestal that compresses the
        # dynamic range the ranking depends on, and it does so in proportion to its weight. Multiplied,
        # it *modulates* the sequence evidence instead: it can sharpen or veto a peak, and reorder
        # residues the sequence branch scores alike, without flattening the result.
        #
        # Measured on the benchmarks (range-level average precision, held-out halves):
        #   GO:0005488, structure branch confident   arithmetic 0.446  geometric 0.505  sequence 0.503
        #   GO:0020037, structure branch weak        arithmetic 0.530  geometric 0.775  sequence 0.776
        # So the geometric rule matches the sequence branch where the structure branch has nothing to
        # add, and overtakes it where the structure branch is confident -- which the arithmetic rule
        # never does at any weight.
        combined_abs = _normalize_positive(
            np.exp(
                seq_weight * np.log(_normalize_positive(seq_combo) + BRANCH_MERGE_FLOOR)
                + struct_weight * np.log(_normalize_positive(struct_combo) + BRANCH_MERGE_FLOOR)
            )
        )
        # Signed counterpart: the sequence branch has one signed signal left, the structure branch
        # two (pair-level grad x input and window-level kernel score x gradient).
        combined_signed = (
            seq_weight * _normalize_signed(seq_grad_signed)
            + struct_weight * _normalize_signed(
                STRUCT_SIGNED_WEIGHTS["grad_x_input"] * _normalize_signed(struct_grad_signed)
                + STRUCT_SIGNED_WEIGHTS["kernel_score_x_grad"] * _normalize_signed(kernel_signed_total)
            )
        )

        kernel_windows = pd.concat(
            [
                _kernel_window_dataframe(
                    "diag",
                    details.get("diag"),
                    diag_grad,
                    dist_square,
                    residue_count,
                    top_windows=top_windows,
                ),
                _kernel_window_dataframe(
                    "anti",
                    details.get("anti"),
                    anti_grad,
                    dist_square,
                    residue_count,
                    top_windows=top_windows,
                ),
            ],
            ignore_index=True,
        )
        if not kernel_windows.empty:
            kernel_windows = kernel_windows.sort_values("abs_grad_x_score", ascending=False).head(int(top_windows))

        kernel_overlap_count_map, kernel_overlap_weight_map, _ = _build_kernel_overlap_maps(
            kernel_windows,
            residue_count,
            mirror=False,
        )
        _, kernel_overlap_weight = _collapse_pair_matrix_to_residues(kernel_overlap_weight_map, residue_count)
        _, kernel_overlap_count = _collapse_pair_matrix_to_residues(kernel_overlap_count_map, residue_count)
        (activity_score, activity_score_multiplicative, activity_threshold,
         activity_site_candidates, activity_segment_labels) = _build_activity_site_candidates(
            {
                "residue_count": residue_count,
                "residue_df": residue_df,
                "combined_abs": combined_abs,
                "combined_signed": combined_signed,
                "seq_grad_abs": seq_grad_abs,
                "kernel_grad_x_input_full_abs": struct_grad_abs,
                "kernel_abs_total": kernel_abs_total,
                "kernel_overlap_weight": kernel_overlap_weight,
                "kernel_overlap_count": kernel_overlap_count,
                "seq_attention": seq_attention,
            }
        )
        timings_s["postprocess_and_reports"] = time.perf_counter() - step_start
        _log_runtime_step(log_runtime, protein_id, go_term, "postprocess_and_reports", timings_s["postprocess_and_reports"])

        analysis = {
            "protein_id": protein_id,
            "go_term": go_term,
            "term_idx": int(term_idx),
            "cif_path": str(record["cif_path"]) if record.get("cif_path") else "",
            "selected_chain": str(
                record.get("selected_chain")
                or (residue_df["chain"].iloc[0] if "chain" in residue_df.columns and not residue_df.empty else "")
            ),
            "sequence": sequence[:residue_count],
            "residue_df": residue_df,
            "residue_count": residue_count,
            "coords": residue_df[["x", "y", "z"]].to_numpy(dtype=np.float32),
            "dist_square": dist_square,
            "pred_prob": float(torch.sigmoid(term_logit.detach()).cpu()),
            "struct_prob": float(torch.sigmoid(logits_struct[0, int(term_idx)].detach()).cpu()),
            "esm_prob": float(torch.sigmoid(logits_esm[0, int(term_idx)].detach()).cpu()),
            "final_logit": float(term_logit.detach().cpu()),
            "struct_logit": struct_term_logit,
            "esm_logit": esm_term_logit,
            "gate": gate_term,
            "seq_branch_weight": seq_weight,
            "struct_branch_weight": struct_weight,
            "esm_attention": _smooth1d(seq_attention, smooth_window),
            "seq_attention": _smooth1d(seq_attention, smooth_window),
            "seq_grad_signed": _smooth1d(seq_grad_signed, smooth_window),
            "seq_grad_abs": _smooth1d(seq_grad_abs, smooth_window),
            "kernel_attr_diag": _smooth1d(attr_diag_np, smooth_window),
            "kernel_attr_anti": _smooth1d(attr_anti_np, smooth_window),
            "kernel_attr_sum": _smooth1d(attr_sum_np, smooth_window),
            "kernel_grad_x_input_full_signed": _smooth1d(struct_grad_signed, smooth_window),
            "kernel_grad_x_input_diag_signed": _smooth1d(branch_grad_diag_signed, smooth_window),
            "kernel_grad_x_input_anti_signed": _smooth1d(branch_grad_anti_signed, smooth_window),
            "kernel_grad_x_input_full_abs": _smooth1d(struct_grad_abs, smooth_window),
            "kernel_grad_x_input_diag_abs": _smooth1d(branch_grad_diag_abs, smooth_window),
            "kernel_grad_x_input_anti_abs": _smooth1d(branch_grad_anti_abs, smooth_window),
            "kernel_signed_diag": _smooth1d(kernel_signed_diag, smooth_window),
            "kernel_signed_anti": _smooth1d(kernel_signed_anti, smooth_window),
            "kernel_signed_total": _smooth1d(kernel_signed_total, smooth_window),
            "kernel_abs_diag": _smooth1d(kernel_abs_diag, smooth_window),
            "kernel_abs_anti": _smooth1d(kernel_abs_anti, smooth_window),
            "kernel_abs_total": _smooth1d(kernel_abs_total, smooth_window),
            "kernel_overlap_weight": _smooth1d(kernel_overlap_weight, smooth_window),
            "kernel_overlap_count": _smooth1d(kernel_overlap_count, smooth_window),
            "activity_score": activity_score,
            "activity_score_multiplicative": activity_score_multiplicative,
            "activity_threshold": activity_threshold,
            "activity_site_candidates": activity_site_candidates,
            "activity_segment_labels": activity_segment_labels,
            "combined_abs": _smooth1d(combined_abs, smooth_window),
            "combined_signed": _smooth1d(combined_signed, smooth_window),
            "kernel_windows": kernel_windows,
            "kernel_gradinput_matrix": kernel_gradinput_matrix,
            "timings_s": timings_s,
        }
        analysis["residue_table"] = _residue_report_table(analysis)
        analysis["top_residues"] = _top_residue_table(analysis, top_residues=top_residues)
        timings_s["total"] = time.perf_counter() - total_start
        return analysis
    finally:
        if hasattr(model, "amp_dtype"):
            model.amp_dtype = original_amp
        if kernel_model is not None and hasattr(kernel_model, "amp_dtype"):
            kernel_model.amp_dtype = original_kernel_amp


def plot_sequence_analysis(analysis: dict[str, Any], save_path: Path | None = None):
    x = np.arange(1, int(analysis["residue_count"]) + 1)
    top_residues = analysis["top_residues"]["residue_index_1based"].tolist()
    activity_sites = analysis.get("activity_site_candidates", pd.DataFrame())
    activity_threshold = float(analysis.get("activity_threshold", 0.0))
    true_residues = [int(v) for v in analysis.get("true_residue_indices_1based", [])]
    residue_count = int(analysis["residue_count"])
    go_label = _go_term_display(analysis["go_term"], analysis.get("go_term_name"))

    # Residue tracks stacked and sharing the residue axis: sequence branch, then the structure
    # branch from coarse to fine, then the aggregate. The two pair-level maps sit side by side in
    # their own full-width row (they are L x L, so they are drawn square rather than stretched to
    # the track width, which means they do not share the tracks' x axis).
    #
    # Heights are in inches (they sum to the figure height) rather than arbitrary ratios, so the maps
    # row can be made as tall as a map is wide -- MAP_SIZE below -- which is what makes the maps come
    # out square. They are drawn with aspect="auto" and fill their cell exactly; a fixed "equal"
    # aspect would letterbox them instead.
    #
    # tight_layout, not constrained_layout: the latter shrinks these part-width map axes to ~2.8"
    # when the figure also holds full-width tracks, which flattens the maps.
    MAP_SIZE = 6.0            # inches, per map -- both maps get exactly this, so both are square
    panel_heights = [2.2, 2.2, 2.2, MAP_SIZE, 2.2, 3.0]
    fig = plt.figure(figsize=(16, sum(panel_heights)))
    grid = fig.add_gridspec(len(panel_heights), 1, height_ratios=panel_heights)

    track_rows = (0, 1, 2, 4, 5)
    axes = []
    for row in track_rows:
        shared = axes[0] if axes else None
        axes.append(fig.add_subplot(grid[row], sharex=shared))
    # Keep panel numbering readable below: axes[0..2] curves, axes[3] maps row, axes[4:] curves.
    axes = axes[:3] + [None] + axes[3:]
    for ax in axes[:3]:
        ax.tick_params(labelbottom=False)
    axes[4].tick_params(labelbottom=False)

    # The maps row gets its own column layout with the colorbars as *separate* axes. Attaching them
    # to the map axes instead would steal width from the map -- and steal different amounts from each
    # (two colorbars on the left map, one on the right), which is exactly what left them different
    # sizes. Fixed columns keep both maps at MAP_SIZE. Widths in inches:
    # Column widths in inches. Tick labels and axis labels are drawn *outside* an axes box, so every
    # gap here has to be wide enough for whatever leans into it -- column 5 in particular carries both
    # the outer bar's right-hand labels and the right map's y axis.
    #   map | flipped bar's labels | bar | gap | bar | labels + right map's y axis | map | gap | bar | slack
    map_widths = [MAP_SIZE, 0.62, 0.16, 0.08, 0.16, 1.15, MAP_SIZE, 0.10, 0.16, 0.50]
    map_grid = grid[3].subgridspec(1, len(map_widths), width_ratios=map_widths, wspace=0.0)
    map_axes = (fig.add_subplot(map_grid[0]), fig.add_subplot(map_grid[6]))
    # Inner bar first in the list, so the flipped one sits next to the left map.
    left_caxes = (fig.add_subplot(map_grid[2]), fig.add_subplot(map_grid[4]))
    right_cax = fig.add_subplot(map_grid[8])

    # Attention sums to 1 over the sequence while |grad x input| is a sum over 1280 channels, so a
    # shared y-axis flattens attention to a line at zero: give it its own axis.
    axes[0].plot(x, analysis["seq_grad_abs"], label="Local: |grad x input|", lw=1.0, color="#f18f01")
    axes[0].set_ylabel("|grad x input|")
    attention_ax = axes[0].twinx()
    attention_ax.plot(x, analysis["seq_attention"], label="Global: attention", lw=1.2, color="#305f72")
    attention_ax.set_ylabel("Attention", color="#305f72")
    attention_ax.set_title("Sequence signal")
    attention_ax.tick_params(axis="y", labelcolor="#305f72")
    handles = axes[0].get_legend_handles_labels()
    twin_handles = attention_ax.get_legend_handles_labels()
    axes[0].legend(handles[0] + twin_handles[0], handles[1] + twin_handles[1], loc="upper right")

    axes[1].plot(x, analysis["kernel_attr_diag"], label="Diagonal", lw=1.0)
    axes[1].plot(x, analysis["kernel_attr_anti"], label="Anti-diagonal", lw=1.0)
    axes[1].plot(x, analysis["kernel_attr_sum"], label="Summed", lw=1.2)
    axes[1].axhline(0.0, color="black", lw=0.8, alpha=0.4)
    axes[1].set_ylabel("Attribution")
    axes[1].set_title("Structure global signal")
    axes[1].legend(loc="upper right")

    axes[2].plot(x, analysis["kernel_grad_x_input_diag_abs"], label="Diagonal", lw=1.0)
    axes[2].plot(x, analysis["kernel_grad_x_input_anti_abs"], label="Anti-diagonal", lw=1.0)
    axes[2].plot(x, analysis["kernel_grad_x_input_full_abs"], label="Summed", lw=1.0)
    axes[2].set_ylabel("|grad x input|")
    axes[2].set_title("Structure local signal")
    axes[2].legend(loc="upper right")

    # Pair-level view of the same structure branch: where the attribution sits on the distance map,
    # and where the top kernel windows pile up. Square, side by side.
    map_left, map_right = map_axes
    draw_structure_vs_attribution_map(fig, map_left, analysis, aspect="auto", caxes=left_caxes)
    map_left.set_title("Structure (lower) vs residue-residue |grad x input| signed (upper)", fontsize=10)
    draw_kernel_overlap_map(fig, map_right, analysis, aspect="auto", cax=right_cax)
    map_right.set_title("Structure window overlap (native search space)", fontsize=10)
    for ax in map_axes:
        ax.set_xlabel("Residue index", fontsize=9)
        ax.set_ylabel("Residue index", fontsize=9)
        ax.tick_params(labelsize=8)

    axes[4].plot(x, analysis["kernel_signed_diag"], label="Diagonal", lw=1.0)
    axes[4].plot(x, analysis["kernel_signed_anti"], label="Anti-diagonal", lw=1.0)
    axes[4].plot(x, analysis["kernel_abs_total"], label="|diag| + |anti|", lw=1.0, ls="--")
    axes[4].plot(
        x,
        _normalize_positive(analysis["kernel_overlap_weight"]),
        label="Overlap (sum)",
        lw=1.0,
        color="#7a5195",
    )
    axes[4].plot(
        x,
        _normalize_positive(analysis["kernel_overlap_count"]),
        label="Overlap (count)",
        lw=1.0,
        ls=":",
        color="#ef5675",
    )
    axes[4].axhline(0.0, color="black", lw=0.8, alpha=0.4)
    axes[4].set_ylabel("Top kernels overlap")
    axes[4].legend(loc="upper right")

    summary_ax = axes[5]
    summary_ax.bar(x, analysis["combined_abs"], color="#305f72", alpha=0.85, label="Combined influence")
    summary_ax.plot(x, _normalize_positive(np.abs(analysis["combined_signed"])), color="#f18f01", lw=1.0, label="|Combined signed|")
    summary_ax.plot(x, analysis["activity_score"], color="#7c3aed", lw=1.2, label="Activity score (add.)")
    if "activity_score_multiplicative" in analysis:
        # Same ingredients and weights as the additive score, combined as a weighted geometric mean:
        # a residue needs support from several signals rather than one strong one.
        summary_ax.plot(x, analysis["activity_score_multiplicative"], color="#d60dbd", lw=1.2,
                        ls="--", label="Activity score (multi.)")
    if activity_threshold > 0.0:
        # The cut-off the segments below were called at. Drawn (it explains their extent) but kept
        # out of the legend: it is an internal segmentation threshold, not a signal to read off.
        summary_ax.axhline(activity_threshold, color="#7c3aed", lw=1.0, ls=":", alpha=0.8)
    for residue_idx in top_residues[:10]:
        summary_ax.axvline(residue_idx, color="#8f2d56", lw=0.8, alpha=0.35)
    band_colors = ["#fde68a", "#fed7aa", "#bfdbfe", "#c7d2fe", "#bbf7d0"]
    for idx, (_, row) in enumerate(activity_sites.head(5).iterrows()):
        start = int(row["start_index_1based"])
        end = int(row["end_index_1based"])
        color = band_colors[idx % len(band_colors)]
        # Bands span the residue tracks only; the maps have their own axes and would be obscured.
        for ax in axes:
            if ax is not None:
                ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.12, zorder=0)
        summary_ax.text(
            0.5 * (start + end),
            0.96 - 0.07 * idx,
            f"{row['segment_label']} {start}-{end}",
            transform=summary_ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="#4b5563",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "alpha": 0.92},
            clip_on=False,
        )
    if true_residues:
        summary_ax.vlines(
            true_residues,
            ymin=0.83,
            ymax=0.995,
            transform=summary_ax.get_xaxis_transform(),
            color="#dc2626",
            lw=1.6,
            alpha=0.95,
            zorder=5,
            label="GT residues",
        )
    summary_ax.set_ylabel("Normalized scores")
    summary_ax.set_title("Combined signal")
    summary_ax.set_xlabel("Residue index (1-based)")
    # Fixed legend order, independent of the order the artists were drawn in.
    legend_order = ["|Combined signed|", "Combined influence", "Activity score (add.)",
                    "Activity score (multi.)", "GT residues"]
    handles, labels = summary_ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered = [(by_label[name], name) for name in legend_order if name in by_label]
    summary_ax.legend([h for h, _ in ordered], [n for _, n in ordered], loc="upper right")

    fig.suptitle(
        (
            f"{analysis['protein_id']} | {go_label} \n"
            f"fusion={analysis['pred_prob']:.3f} | "
            f"sequence={analysis['esm_prob']:.3f} | "
            f"structure={analysis['struct_prob']:.3f} | "
            f"gate={analysis['gate']:.3f}"
        ),
        y=0.997,
    )
    fig.tight_layout(h_pad=0.9, rect=(0, 0, 1, 0.985))
    if save_path is not None:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    return fig


_BRANCH_COLORS = {"diag": "#0f766e", "anti": "#b45309"}


def _draw_kernel_windows(ax, kernel_windows, residue_count, *, label_windows=True):
    """Outline the top kernel windows on a pair-level map, in the model's native orientation."""
    max_abs = 0.0
    if kernel_windows is not None and len(kernel_windows) > 0:
        max_abs = float(np.max(np.asarray(kernel_windows["abs_grad_x_score"], dtype=np.float32)))
    max_abs = max(max_abs, EPS)

    for rect in _iter_kernel_window_rectangles(kernel_windows, residue_count, mirror=False):
        # Opacity tracks the window's contribution, so the strongest windows read first.
        alpha = 0.25 + 0.55 * min(1.0, float(rect["abs_grad_x_score"]) / max_abs)
        ax.add_patch(
            Rectangle(
                (int(rect["c0"]) - 0.5, int(rect["r0"]) - 0.5),
                int(rect["c1"]) - int(rect["c0"]),
                int(rect["r1"]) - int(rect["r0"]),
                fill=False,
                lw=1.8,
                ec=_BRANCH_COLORS.get(str(rect["branch"]), "#8f2d56"),
                alpha=alpha,
            )
        )
        if label_windows:
            ax.text(
                int(rect["c0"]) + 1.0,
                int(rect["r1"]) - 2.0,
                str(rect["label"]),
                color=_BRANCH_COLORS.get(str(rect["branch"]), "#8f2d56"),
                fontsize=8,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.72},
            )


def draw_structure_vs_attribution_map(fig, ax, analysis, *, aspect="equal", label_windows=True,
                                      caxes=None):
    """Draw the distogram and the pair-level attribution as two halves of one square map.

    The distogram is symmetric and the model only ever searches the upper triangle, so the two
    signals can share one axis without overlapping: **lower triangle = the input structure**
    (distance feature), **upper triangle = the signed residue-residue Grad x Input** that the
    structure branch derived from it (the matrix also saved as ``kernel_gradinput.npy``). Reading
    across the diagonal therefore compares "what the structure looks like here" with "what the
    model made of it", and the top kernel windows are outlined where they natively live.

    Colour limits for the attribution are symmetric around zero at the 99th percentile of
    ``|attribution|``, so a handful of extreme pairs cannot wash the map out.
    """
    residue_count = int(analysis["residue_count"])
    dist_square = np.asarray(analysis["dist_square"], dtype=np.float32)[:residue_count, :residue_count]
    attribution = np.asarray(analysis["kernel_gradinput_matrix"], dtype=np.float32)
    attribution = attribution[:residue_count, :residue_count]

    lower = np.tri(residue_count, k=-1, dtype=bool)
    structure_half = np.ma.masked_where(~lower, dist_square)
    # The attribution matrix is symmetric by construction; show it once, in the upper triangle.
    attribution_half = np.ma.masked_where(lower, attribution)

    finite = np.abs(attribution[np.isfinite(attribution)])
    limit = float(np.percentile(finite, 99)) if finite.size else 0.0
    limit = max(limit, EPS)

    structure_im = ax.imshow(structure_half, cmap="cividis", origin="upper", aspect=aspect)
    attribution_im = ax.imshow(
        attribution_half, cmap="RdBu_r", origin="upper", aspect=aspect, vmin=-limit, vmax=limit
    )
    _draw_kernel_windows(ax, analysis.get("kernel_windows"), residue_count, label_windows=label_windows)
    ax.plot([-0.5, residue_count - 0.5], [-0.5, residue_count - 0.5], color="black", lw=0.8, alpha=0.45)

    # Two colorbars share the strip right of the map. The inner one is flipped -- ticks and label on
    # its left -- so the two sets of tick labels face away from each other instead of colliding in
    # the gap between the bars. `caxes` places them in dedicated axes (see plot_sequence_analysis,
    # where stealing width from the map would make the two maps different sizes); without it they are
    # attached to the map axes, which is fine for a stand-alone figure.
    if caxes is None:
        attribution_bar = fig.colorbar(attribution_im, ax=ax, fraction=0.040, pad=0.16)
        structure_bar = fig.colorbar(structure_im, ax=ax, fraction=0.040, pad=0.04)
    else:
        attribution_bar = fig.colorbar(attribution_im, cax=caxes[0])
        structure_bar = fig.colorbar(structure_im, cax=caxes[1])
    # attribution_bar.set_label("GxI signed (upper)", fontsize=8)
    attribution_bar.ax.yaxis.set_ticks_position("left")
    # attribution_bar.ax.yaxis.set_label_position("left")
    # attribution_bar.ax.tick_params(labelsize=7)
    # structure_bar.set_label("distance (lower)", fontsize=8)
    # structure_bar.ax.tick_params(labelsize=7)

    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor=_BRANCH_COLORS["diag"], label="Diagonal window", linewidth=1.8),
            Patch(facecolor="none", edgecolor=_BRANCH_COLORS["anti"], label="Off-diagonal window", linewidth=1.8),
        ],
        loc="lower right",
        fontsize=8,
        frameon=True,
    )
    return structure_im, attribution_im


def draw_kernel_overlap_map(fig, ax, analysis, *, aspect="equal", cax=None):
    """Draw where the top kernel windows pile up: summed |score x grad|, with overlap contours."""
    residue_count = int(analysis["residue_count"])
    dist_square = np.asarray(analysis["dist_square"], dtype=np.float32)[:residue_count, :residue_count]
    count_map, weight_map, _ = _build_kernel_overlap_maps(
        analysis.get("kernel_windows"), residue_count, mirror=False
    )
    overlap_mask = np.ma.masked_where(weight_map <= 0.0, weight_map)

    ax.imshow(dist_square, cmap="gray_r", origin="upper", alpha=0.32, aspect=aspect)
    overlap_im = ax.imshow(overlap_mask, cmap="magma", origin="upper", alpha=0.92, aspect=aspect)
    if np.any(count_map >= 2):
        ax.contour(count_map, levels=[2], colors="#38bdf8", linewidths=1.0, origin="upper")
        if np.max(count_map) >= 3:
            ax.contour(count_map, levels=[3], colors="#0f172a", linewidths=0.9, origin="upper")
    ax.plot([-0.5, residue_count - 0.5], [-0.5, residue_count - 0.5], color="white", lw=0.8, alpha=0.35)

    overlap_bar = (fig.colorbar(overlap_im, ax=ax, fraction=0.046, pad=0.01) if cax is None
                   else fig.colorbar(overlap_im, cax=cax))
    overlap_bar.set_label("Sum |grad x score|", fontsize=8)
    overlap_bar.ax.tick_params(labelsize=7)
    return overlap_im


def plot_kernel_distmap_analysis(analysis: dict[str, Any], save_path: Path | None = None):
    """Stand-alone pair-level figure: structure vs attribution, and kernel-window overlap.

    Same two maps that ``plot_sequence_analysis`` embeds in its panel stack, drawn here at true
    square aspect (the embedded copies are stretched to align with the residue tracks).
    """
    residue_count = int(analysis["residue_count"])
    top_residues = analysis["top_residues"]["residue_index_1based"].astype(int).tolist()[:10]
    go_label = _go_term_display(analysis["go_term"], analysis.get("go_term_name"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=140)
    ax0, ax1 = axes

    draw_structure_vs_attribution_map(fig, ax0, analysis)
    for residue_idx in top_residues:
        pos = int(residue_idx) - 1
        ax0.axhline(pos, color="#ffffff", lw=0.5, alpha=0.22)
        ax0.axvline(pos, color="#ffffff", lw=0.5, alpha=0.22)
    ax0.set_title("Structure (lower) vs residue-residue grad x input (upper)")
    ax0.set_xlabel("Residue index")
    ax0.set_ylabel("Residue index")

    draw_kernel_overlap_map(fig, ax1, analysis)
    for residue_idx in top_residues:
        pos = int(residue_idx) - 1
        ax1.axhline(pos, color="#f8fafc", lw=0.45, alpha=0.15)
        ax1.axvline(pos, color="#f8fafc", lw=0.45, alpha=0.15)
    ax1.set_title("Structure window overlap (native search space)")
    ax1.set_xlabel("Residue index")
    ax1.set_ylabel("Residue index")
    ax1.text(
        0.01,
        1.02,
        "Native model view: upper triangle only | Cyan: >=2 overlaps | Black: >=3 overlaps",
        transform=ax1.transAxes,
        fontsize=8,
        color="#334155",
    )

    fig.suptitle(
        (
            f"{analysis['protein_id']} | structure windows on distogram | "
            f"fusion={analysis['pred_prob']:.3f}\n"
            f"{go_label}"
        ),
        y=0.995,
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    return fig


def _build_structure_plotly_figure(analysis: dict[str, Any]):
    if go is None:
        return None

    coords = np.asarray(analysis["coords"], dtype=np.float32)
    scores = np.asarray(analysis["combined_abs"], dtype=np.float32)
    top_indices = set((analysis["top_residues"]["residue_index_1based"].astype(int) - 1).tolist())
    go_label = _go_term_display(analysis["go_term"], analysis.get("go_term_name"))

    hover_text = []
    for idx, row in analysis["residue_df"].iterrows():
        hover_text.append(
            "<br>".join(
                [
                    f"Residue #{idx + 1}",
                    f"AA: {row.get('residue_aa', '?')}",
                    f"Residue ID: {row.get('residue_id', '?')}",
                    f"Combined: {scores[idx]:.4f}",
                    f"Sequence GxI abs: {analysis['seq_grad_abs'][idx]:.4f}",
                    f"Structure GxI abs: {analysis['kernel_grad_x_input_full_abs'][idx]:.4f}",
                    f"Structure abs: {analysis['kernel_abs_total'][idx]:.4f}",
                    f"Overlap weight: {analysis['kernel_overlap_weight'][idx]:.4f}",
                    f"Overlap count: {analysis['kernel_overlap_count'][idx]:.4f}",
                    f"Activity score: {analysis['activity_score'][idx]:.4f}",
                    f"Activity segment: {analysis['activity_segment_labels'][idx] or '-'}",
                ]
            )
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="lines",
            line={"color": "rgba(60,60,60,0.35)", "width": 6},
            hoverinfo="skip",
            name="Backbone",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker={
                "size": 5,
                "color": scores,
                "colorscale": "Turbo",
                "opacity": 0.95,
                "colorbar": {"title": "Combined influence"},
            },
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Residues",
        )
    )

    if top_indices:
        top_coords = coords[sorted(top_indices)]
        fig.add_trace(
            go.Scatter3d(
                x=top_coords[:, 0],
                y=top_coords[:, 1],
                z=top_coords[:, 2],
                mode="markers",
                marker={"size": 8, "color": "#8f2d56", "opacity": 0.95},
                name="Top residues",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=(
            f"{analysis['protein_id']} | p={analysis['pred_prob']:.3f}"
            f"<br><sup>{go_label}</sup>"
        ),
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def _build_structure_3d_html(analysis: dict[str, Any]) -> str | None:
    cif_path_value = analysis.get("cif_path")
    if not cif_path_value:
        return None

    cif_path = Path(str(cif_path_value))
    if not cif_path.exists():
        return None
    structure_format = "pdb" if cif_path.suffix.lower() == ".pdb" else "cif"

    residue_df = analysis["residue_df"].copy().reset_index(drop=True)
    if residue_df.empty:
        return None

    selected_chain = str(
        analysis.get("selected_chain")
        or (residue_df["chain"].iloc[0] if "chain" in residue_df.columns else "")
        or "A"
    )

    scores = np.asarray(analysis["combined_abs"], dtype=np.float32)
    signed_scores = np.asarray(analysis["combined_signed"], dtype=np.float32)
    seq_attention = np.asarray(analysis["seq_attention"], dtype=np.float32)
    kernel_overlap_weight = np.asarray(analysis["kernel_overlap_weight"], dtype=np.float32)
    kernel_overlap_count = np.asarray(analysis["kernel_overlap_count"], dtype=np.float32)
    activity_score = np.asarray(analysis.get("activity_score", np.zeros_like(scores)), dtype=np.float32)
    activity_score_multiplicative = np.asarray(
        analysis.get("activity_score_multiplicative", np.zeros_like(scores)), dtype=np.float32
    )
    activity_segment_labels = np.asarray(
        analysis.get("activity_segment_labels", np.array([""] * len(scores), dtype=object)),
        dtype=object,
    )
    activity_candidates = analysis.get("activity_site_candidates", pd.DataFrame()).copy().reset_index(drop=True)
    cartoon_colors, hotspot_colors = _colorize_residue_cartoon(scores)

    # Scores the viewer can colour residues by, in dropdown order. Each is normalized to [0, 1] so a
    # single palette works for all of them and 1.0 always means "the strongest residue in this
    # protein for this score".
    color_score_options = [
        ("Sequence local", "seq_grad_abs"),
        ("Structure local (all)", "kernel_grad_x_input_full_abs"),
        ("Structure local (top)", "kernel_overlap_weight"),
        ("Activity score (add.)", "activity_score"),
        ("Activity score (multi.)", "activity_score_multiplicative"),
    ]
    color_scores = {}
    for label, key in color_score_options:
        values = np.asarray(analysis.get(key, np.zeros_like(scores)), dtype=np.float32)
        color_scores[label] = [
            round(float(v), 5) for v in _normalize_positive(np.abs(values))[:len(residue_df)]
        ]
    color_score_labels = [label for label, _ in color_score_options]
    default_color_score = "Activity score (add.)"
    # One line per option, naming the panel of sequence_analysis.png that plots the same curve, so the
    # 3D colouring can be read against the 1-D figure without guessing.
    color_score_help = {
        "Sequence local":
            "|Grad x Input| on the ESM embeddings. Plotted in <b>Sequence signal</b> "
            "(orange, \u201cLocal: |Grad x Input|\u201d). Sharp, residue-specific peaks.",
        "Structure local (all)":
            "|Grad x Input| on the distogram, diagonal + off-diagonal. Plotted in "
            "<b>Structure local signal</b> (green, \u201cSummed\u201d). Broad, varies over tens of "
            "residues.",
        "Structure local (top)":
            "Overlap (sum): how strongly the top structure windows cover this residue. Plotted in "
            "<b>Top kernels overlap</b> (purple, \u201cOverlap (sum)\u201d). Blocky \u2014 it changes "
            "at window edges.",
        "Activity score (add.)":
            "Weighted <b>sum</b> of all signals. Plotted in <b>Combined signal</b> (solid purple). "
            "This is the curve the AS candidate sites are called from.",
        "Activity score (multi.)":
            "Same ingredients and weights as a weighted <b>geometric mean</b>. Plotted in "
            "<b>Combined signal</b> (dashed magenta). Stricter: several signals must agree, so it drops "
            "lower between peaks.",
    }
    top_table = analysis["top_residues"].copy().reset_index(drop=True)
    top_table = top_table.head(min(20, len(top_table)))
    rank_by_index = np.empty_like(scores, dtype=np.int32)
    if scores.size:
        rank_by_index[np.argsort(scores)[::-1]] = np.arange(1, scores.size + 1, dtype=np.int32)

    residue_entries: list[dict[str, Any]] = []
    for idx, row in residue_df.iterrows():
        chain_value = str(row.get("chain", selected_chain) or selected_chain)
        resi_value = int(row.get("residue_id", idx + 1))
        residue_entries.append(
            {
                "key": f"{chain_value}:{resi_value}",
                "resi": resi_value,
                "chain": chain_value,
                "index1": int(idx + 1),
                "aa": str(row.get("residue_aa", "?")),
                "score": float(scores[idx]),
                "combined_signed": float(signed_scores[idx]),
                "seq_grad": float(analysis["seq_grad_abs"][idx]),
                "struct_grad": float(analysis["kernel_grad_x_input_full_abs"][idx]),
                "kernel": float(analysis["kernel_abs_total"][idx]),
                "overlap_weight": float(kernel_overlap_weight[idx]),
                "overlap_count": float(kernel_overlap_count[idx]),
                "activity_score": float(activity_score[idx]),
                "activity_score_multiplicative": float(activity_score_multiplicative[idx]),
                "activity_segment": str(activity_segment_labels[idx] or ""),
                "attention": float(seq_attention[idx]),
                "rank": int(rank_by_index[idx]),
                "cartoon_color": cartoon_colors[idx],
                "hotspot_color": hotspot_colors[idx],
                "x": float(row.get("x", 0.0)),
                "y": float(row.get("y", 0.0)),
                "z": float(row.get("z", 0.0)),
            }
        )

    top_entries: list[dict[str, Any]] = []
    for _, row in top_table.iterrows():
        residue_index = int(row["residue_index_1based"]) - 1
        if residue_index < 0 or residue_index >= len(residue_entries):
            continue
        meta = residue_entries[residue_index]
        top_entries.append(
            {
                "key": meta["key"],
                "resi": int(meta["resi"]),
                "chain": meta["chain"],
                "index1": meta["index1"],
                "aa": meta["aa"],
                "rank": meta["rank"],
                "label": (
                    f"#{meta['index1']} {meta['aa']} (resi {meta['resi']})"
                    f" | combined {meta['score']:.3f}"
                ),
                "color": meta["hotspot_color"],
                "score": meta["score"],
                "combined_signed": meta["combined_signed"],
                "seq_grad": meta["seq_grad"],
                "struct_grad": meta["struct_grad"],
                "kernel": meta["kernel"],
                "overlap_weight": meta["overlap_weight"],
                "overlap_count": meta["overlap_count"],
                "activity_score": meta["activity_score"],
                "activity_segment": meta["activity_segment"],
                "attention": meta["attention"],
            }
        )

    activity_entries: list[dict[str, Any]] = []
    for _, row in activity_candidates.head(5).iterrows():
        peak_index = int(row["peak_residue_index_1based"]) - 1
        if peak_index < 0 or peak_index >= len(residue_entries):
            continue
        peak_meta = residue_entries[peak_index]
        activity_entries.append(
            {
                "segment_label": str(row["segment_label"]),
                "start": int(row["start_index_1based"]),
                "end": int(row["end_index_1based"]),
                "length": int(row["length"]),
                "peak_key": peak_meta["key"],
                "peak_index1": int(row["peak_residue_index_1based"]),
                "peak_aa": str(row["peak_residue_aa"]),
                "peak_score": float(row["peak_activity_score"]),
                "mean_score": float(row["mean_activity_score"]),
                "mean_overlap_weight": float(row["mean_overlap_weight"]),
            }
        )

    # Residue keys in sequence order, so the sequence track can be laid out left to right.
    residue_order = [entry["key"] for entry in residue_entries]
    residue_meta_by_key = {
        entry["key"]: {
            "key": entry["key"],
            "chain": entry["chain"],
            "resi": entry["resi"],
            "index1": entry["index1"],
            "aa": entry["aa"],
            "rank": entry["rank"],
            "score": entry["score"],
            "combined_signed": entry["combined_signed"],
            "seq_grad": entry["seq_grad"],
            "struct_grad": entry["struct_grad"],
            "kernel": entry["kernel"],
            "overlap_weight": entry["overlap_weight"],
            "overlap_count": entry["overlap_count"],
            "activity_score": entry["activity_score"],
            "activity_score_multiplicative": entry["activity_score_multiplicative"],
            "activity_segment": entry["activity_segment"],
            "attention": entry["attention"],
            "cartoon_color": entry["cartoon_color"],
            "hotspot_color": entry["hotspot_color"],
            "x": entry["x"],
            "y": entry["y"],
            "z": entry["z"],
        }
        for entry in residue_entries
    }

    key_seed = (
        f"{analysis['protein_id']}|{analysis['go_term']}|{selected_chain}|"
        f"{analysis['term_idx']}|{analysis['pred_prob']:.6f}"
    )
    viewer_id = f"mol_{hashlib.md5(key_seed.encode('utf-8')).hexdigest()[:12]}"
    root_id = f"{viewer_id}_root"
    selected_panel_id = f"{viewer_id}_selected"
    clear_button_id = f"{viewer_id}_clear"
    color_select_id = f"{viewer_id}_colorby"
    color_legend_id = f"{viewer_id}_colorlegend"
    color_help_id = f"{viewer_id}_colorhelp"
    sequence_track_id = f"{viewer_id}_seqtrack"

    color_options_html = "".join(
        f"<option value=\"{label}\"{' selected' if label == default_color_score else ''}>{label}</option>"
        for label in color_score_labels
    )
    top_rows_html = "".join(
        (
            f"<tr data-role='top-residue-row' data-key='{entry['key']}' "
            "style='cursor:pointer; border-top: 1px solid #edf2f7;'>"
            "<td style='padding: 8px 0 8px 0;'>"
            f"<span style='display:inline-block; width:10px; height:10px; border-radius:999px; "
            f"background:{entry['color']}; margin-right:8px; vertical-align:middle;'></span>"
            f"{entry['label']}</td>"
            f"<td style='padding: 8px 0;'>{entry['score']:.3f}</td>"
            f"<td style='padding: 8px 0;'>{entry['seq_grad']:.3f}</td>"
            f"<td style='padding: 8px 0;'>{entry['struct_grad']:.3f}</td>"
            f"<td style='padding: 8px 0;'>{entry['kernel']:.3f}<br>"
            f"<span style='font-size:11px;color:#64748b;'>ov {entry['overlap_weight']:.2f}</span></td>"
            "</tr>"
        )
        for entry in top_entries
    )
    activity_rows_html = "".join(
        (
            f"<tr data-role='activity-row' data-key='{entry['peak_key']}' "
            "style='cursor:pointer; border-top: 1px solid #edf2f7;'>"
            f"<td style='padding: 7px 0;'>{entry['segment_label']}</td>"
            f"<td style='padding: 7px 0;'>{entry['start']}-{entry['end']}</td>"
            f"<td style='padding: 7px 0;'>#{entry['peak_index1']} {entry['peak_aa']}</td>"
            f"<td style='padding: 7px 0;'>{entry['peak_score']:.3f}</td>"
            "</tr>"
        )
        for entry in activity_entries
    )
    if not activity_rows_html:
        activity_rows_html = (
            "<tr><td colspan='4' style='padding: 8px 0; color: #64748b;'>"
            "No contiguous activity segment was detected for this GO term.</td></tr>"
        )

    summary_bits = [
        f"<strong>{analysis['protein_id']}</strong>",
        f"fusion={analysis['pred_prob']:.3f}",
        f"sequence={analysis['esm_prob']:.3f}",
        f"structure={analysis['struct_prob']:.3f}",
        f"gate={analysis['gate']:.3f}",
        f"chain={selected_chain}",
    ]
    summary_html = " | ".join(summary_bits)
    go_label = _go_term_display(analysis["go_term"], analysis.get("go_term_name"))

    html = f"""
<div id="{root_id}" style="font-family: Arial, sans-serif; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px; overflow: hidden;">
  <div style="padding: 14px 18px; border-bottom: 1px solid #e5e7eb; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);">
    <div style="font-size: 16px; color: #111827; line-height: 1.45;">{summary_html}</div>
    <div style="font-size: 13px; color: #334155; line-height: 1.45; margin-top: 3px;">{go_label}</div>
    <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">
      Click the ribbon or atoms to pin residues. Pinned residues stay highlighted on the structure and their term-specific attribution metrics appear in the side panel.
    </div>
  </div>
  <div style="display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 0; height: 78vh; min-height: 560px; max-height: 900px;">
    <div style="display: flex; flex-direction: column; min-width: 0; min-height: 0;">
      <div style="flex: 0 0 auto; max-height: 27%; overflow-y: auto; padding: 9px 12px 7px; border-bottom: 1px solid #e5e7eb; background: #ffffff;">
        <div style="font-size: 11px; color: #6b7280; margin-bottom: 5px;">
          Sequence - coloured by the selected score, wrapped to the panel width. Click a residue to pin it, hover to highlight it on the structure; a coloured underline marks an AS candidate site.
        </div>
        <div id="{sequence_track_id}" style="padding-bottom: 1px;"></div>
      </div>
      <div id="{viewer_id}" style="position: relative; overflow: hidden; width: 100%; flex: 1 1 0; min-height: 0; background: radial-gradient(circle at 30% 20%, #ffffff 0%, #f7fafc 55%, #eef2f7 100%);"></div>
    </div>
    <div style="border-left: 1px solid #e5e7eb; background: #fbfcfe; padding: 14px 16px; overflow-y: auto; min-height: 0;">
      <div style="margin-bottom: 16px; padding-bottom: 14px; border-bottom: 1px solid #e5e7eb;">
        <label for="{color_select_id}" style="display:block; font-size: 13px; color: #111827; font-weight: 700; margin-bottom: 6px;">Colour residues by</label>
        <select id="{color_select_id}" style="width: 100%; padding: 6px 8px; border: 1px solid #d1d5db; border-radius: 8px; background: white; color: #111827; font-size: 12px; cursor: pointer;">
          {color_options_html}
        </select>
        <div id="{color_help_id}" style="margin-top: 7px; font-size: 11px; color: #475569; line-height: 1.5;"></div>
        <div id="{color_legend_id}" style="margin-top: 9px;"></div>
      </div>
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 10px;">
        <div style="font-size: 13px; color: #111827; font-weight: 700;">Pinned residues</div>
        <button id="{clear_button_id}" type="button" style="padding: 5px 9px; border: 1px solid #d1d5db; border-radius: 999px; background: white; color: #374151; font-size: 11px; cursor: pointer;">
          Clear
        </button>
      </div>
      <div id="{selected_panel_id}" style="margin-bottom: 18px;"></div>
      <div style="font-size: 13px; color: #111827; font-weight: 700; margin-bottom: 10px;">Activity-site candidates</div>
      <div style="font-size: 12px; color: #6b7280; line-height: 1.5; margin-bottom: 12px;">
        Sequence intervals where sequence, structure and kernel-overlap evidence cluster most strongly for this GO term.
      </div>
      <table style="width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 18px;">
        <thead>
          <tr style="text-align: left; color: #374151;">
            <th style="padding: 6px 0;">Site</th>
            <th style="padding: 6px 0;">Range</th>
            <th style="padding: 6px 0;">Peak</th>
            <th style="padding: 6px 0;">Score</th>
          </tr>
        </thead>
        <tbody>{activity_rows_html}</tbody>
      </table>
      <div style="font-size: 13px; color: #111827; font-weight: 700; margin-bottom: 10px;">Top highlighted residues</div>
      <div style="font-size: 12px; color: #6b7280; line-height: 1.5; margin-bottom: 14px;">
        Hover shows quick values. Clicking a row below also pins that residue on the model and recenters the camera on it.
      </div>
      <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
        <thead>
          <tr style="text-align: left; color: #374151;">
            <th style="padding: 6px 0;">Residue</th>
            <th style="padding: 6px 0;">Comb.</th>
            <th style="padding: 6px 0;">Seq</th>
            <th style="padding: 6px 0;">Struct</th>
            <th style="padding: 6px 0;">Kernel</th>
          </tr>
        </thead>
        <tbody>{top_rows_html}</tbody>
      </table>
    </div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
<script>
(function() {{
  const rootId = {json.dumps(root_id)};
  const viewerId = {json.dumps(viewer_id)};
  const selectedPanelId = {json.dumps(selected_panel_id)};
  const clearButtonId = {json.dumps(clear_button_id)};
  const chainId = {json.dumps(selected_chain)};
  const cifText = {json.dumps(cif_path.read_text(encoding="utf-8"))};
  const structureFormat = {json.dumps(structure_format)};
  const residueMeta = {json.dumps(residue_meta_by_key)};
  const topResidues = {json.dumps(top_entries)};
  const colorSelectId = {json.dumps(color_select_id)};
  const colorLegendId = {json.dumps(color_legend_id)};
  const colorHelpId = {json.dumps(color_help_id)};
  const sequenceTrackId = {json.dumps(sequence_track_id)};
  const colorScoreHelp = {json.dumps(color_score_help)};
  const residueOrder = {json.dumps(residue_order)};
  const colorScores = {json.dumps(color_scores)};
  let activeColorScore = {json.dumps(default_color_score)};

  // Palette for the residue colouring. Deliberately asymmetric: the bottom half is pale and cool so
  // inactive parts of the fold recede, and only values approaching 1 -- the residues the score
  // actually calls active -- go warm and saturated.
  const COLOR_RAMP = [
    [0.00, [223, 229, 237]],
    [0.30, [148, 177, 200]],
    [0.55, [244, 208, 98]],
    [0.75, [235, 124, 52]],
    [1.00, [166, 20, 30]],
  ];

  const rampColor = (value) => {{
    const v = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
    for (let i = 1; i < COLOR_RAMP.length; i += 1) {{
      const [hi, hiRgb] = COLOR_RAMP[i];
      if (v <= hi) {{
        const [lo, loRgb] = COLOR_RAMP[i - 1];
        const f = hi > lo ? (v - lo) / (hi - lo) : 0;
        const rgb = loRgb.map((c, k) => Math.round(c + (hiRgb[k] - c) * f));
        return `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
      }}
    }}
    return "rgb(166,20,30)";
  }};

  const scoreOf = (meta) => {{
    const values = colorScores[activeColorScore];
    if (!values || !meta) {{
      return 0;
    }}
    const value = values[meta.index1 - 1];
    return Number.isFinite(value) ? value : 0;
  }};

  const colorOf = (meta) => rampColor(scoreOf(meta));

  // ---------------------------------------------------------------------------------------------
  // Everything below runs off the inline data alone, so it renders immediately and must NOT wait for
  // the 3Dmol CDN script. That script is external, so a browser that blocks it (an untrusted local
  // file, an offline machine) would otherwise leave the sequence ruler blank while the viewer retried
  // forever. The viewer installs its callbacks into these hooks once it does load.
  // ---------------------------------------------------------------------------------------------
  let selectedKeys = [];
  let onSelectionChange = () => {{}};   // -> applyStyles once the viewer exists
  let onFocusResidue = () => {{}};      // -> focusResidue
  let onHoverResidue = () => {{}};      // transient 3D highlight while hovering the ruler
  let onHoverEnd = () => {{}};

  const sequenceTrack = document.getElementById(sequenceTrackId);
  const CELL_WIDTH = 12;

  // Same effect as clicking a row in the top-residue table: pin (most recent first, max 8),
  // restyle the model and recenter the camera.
  const toggleResidue = (key) => {{
    if (!residueMeta[key]) {{
      return;
    }}
    selectedKeys = [key].concat(selectedKeys.filter((item) => item !== key)).slice(0, 8);
    renderSequenceTrack();
    onSelectionChange();
    onFocusResidue(key);
  }};

  // The ruler wraps instead of scrolling sideways: the sequence is cut into lines that fit the
  // available width, each line carrying its own numbering row so indices stay above their residues.
  // Line length is measured at render time and recomputed on resize, so it always fills the panel.
  let residuesPerLine = 60;
  const measureResiduesPerLine = () => {{
    const width = sequenceTrack ? sequenceTrack.clientWidth : 0;
    // clientWidth is 0 while the element is hidden; keep the previous value in that case.
    return width > CELL_WIDTH * 10 ? Math.floor(width / CELL_WIDTH) : residuesPerLine;
  }};

  const renderSequenceTrack = () => {{
    if (!sequenceTrack) {{
      return;
    }}
    residuesPerLine = measureResiduesPerLine();
    const lines = [];
    for (let offset = 0; offset < residueOrder.length; offset += residuesPerLine) {{
      lines.push(residueOrder.slice(offset, offset + residuesPerLine));
    }}
    sequenceTrack.innerHTML = lines.map((line) => {{
      const firstMeta = residueMeta[line[0]];
      const lineStart = firstMeta ? firstMeta.index1 : 0;
      const ticks = line.map((key) => {{
        const meta = residueMeta[key];
        const index1 = meta ? meta.index1 : 0;
        // Always label where a line begins, then every tenth residue.
        const show = index1 === lineStart || index1 % 10 === 0;
        return `<div style="width:${{CELL_WIDTH}}px; flex:0 0 ${{CELL_WIDTH}}px; font-size:8px; color:#94a3b8; text-align:center; white-space:nowrap;">${{show ? index1 : "&nbsp;"}}</div>`;
      }}).join("");
      const cells = line.map((key) => {{
        const meta = residueMeta[key];
        if (!meta) {{
          return "";
        }}
        const pinned = selectedKeys.includes(key);
        const value = scoreOf(meta);
        // Dark background needs light text; the ramp only gets dark at the top of the scale.
        const letterColor = value >= 0.72 ? "#ffffff" : "#1f2937";
        const segment = meta.activity_segment
          ? "border-bottom:3px solid #7c3aed;"
          : "border-bottom:3px solid transparent;";
        const pinnedStyle = pinned
          ? "outline:2px solid #0f172a; outline-offset:-2px; font-weight:700;"
          : "";
        // Single-line tooltip on purpose: a raw newline here would land inside a double-quoted HTML
        // attribute, and escaping it through the f-string is easy to get wrong.
        const title = `#${{meta.index1}} ${{meta.aa}} (resi ${{meta.resi}}) | ${{activeColorScore}}: ${{value.toFixed(3)}}${{meta.activity_segment ? " | site " + meta.activity_segment : ""}}`;
        return `<div data-role="seq-cell" data-key="${{key}}" title="${{title}}" style="width:${{CELL_WIDTH}}px; flex:0 0 ${{CELL_WIDTH}}px; height:20px; line-height:20px; text-align:center; font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11px; cursor:pointer; background:${{colorOf(meta)}}; color:${{letterColor}}; ${{segment}} ${{pinnedStyle}}">${{meta.aa}}</div>`;
      }}).join("");
      return `
        <div style="margin-bottom: 7px;">
          <div style="display:flex; align-items:flex-end;">${{ticks}}</div>
          <div style="display:flex; align-items:stretch;">${{cells}}</div>
        </div>`;
    }}).join("");
    Array.from(sequenceTrack.querySelectorAll("[data-role='seq-cell']")).forEach((cell) => {{
      const key = cell.dataset.key;
      cell.addEventListener("click", () => toggleResidue(key));
      cell.addEventListener("mouseenter", () => onHoverResidue(key));
      cell.addEventListener("mouseleave", () => onHoverEnd(key));
    }});
  }};

  // Re-wrap when the window changes width, debounced so dragging a window edge stays smooth.
  let resizeTimer = null;
  if (typeof window !== "undefined" && window.addEventListener) {{
    window.addEventListener("resize", () => {{
      if (resizeTimer) {{
        clearTimeout(resizeTimer);
      }}
      resizeTimer = setTimeout(() => {{
        if (measureResiduesPerLine() !== residuesPerLine) {{
          renderSequenceTrack();
        }}
      }}, 150);
    }});
  }}

  const colorSelect = document.getElementById(colorSelectId);
  const colorLegend = document.getElementById(colorLegendId);
  const colorHelp = document.getElementById(colorHelpId);

  const renderColorHelp = () => {{
    if (colorHelp) {{
      colorHelp.innerHTML = colorScoreHelp[activeColorScore] || "";
    }}
  }};

  const renderColorLegend = () => {{
    if (!colorLegend) {{
      return;
    }}
    const stops = COLOR_RAMP.map(([position, rgb]) =>
      `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}}) ${{(position * 100).toFixed(0)}}%`).join(", ");
    colorLegend.innerHTML = `
      <div style="height: 10px; border-radius: 999px; border: 1px solid #e5e7eb; background: linear-gradient(90deg, ${{stops}});"></div>
      <div style="display:flex; justify-content:space-between; font-size: 10px; color: #6b7280; margin-top: 3px;">
        <span>0 (inactive)</span><span>1 (most active)</span>
      </div>`;
  }};

  if (colorSelect) {{
    colorSelect.addEventListener("change", () => {{
      activeColorScore = colorSelect.value;
      renderColorHelp();
      renderSequenceTrack();
      onSelectionChange();
    }});
  }}
  renderColorHelp();
  renderColorLegend();
  renderSequenceTrack();

  const ensureViewer = () => {{
    if (typeof $3Dmol === "undefined") {{
      setTimeout(ensureViewer, 120);
      return;
    }}

    const viewer = $3Dmol.createViewer(viewerId, {{
      backgroundColor: "white",
      antialias: true,
      id: viewerId,
    }});
    const root = document.getElementById(rootId);
    const selectedPanel = document.getElementById(selectedPanelId);
    const clearButton = document.getElementById(clearButtonId);
    const topRows = Array.from(root.querySelectorAll("[data-role='top-residue-row']"));
    const activityRows = Array.from(root.querySelectorAll("[data-role='activity-row']"));
    let selectedLabels = [];

    viewer.addModel(cifText, structureFormat);

    const removeSelectedLabels = () => {{
      selectedLabels.forEach((label) => viewer.removeLabel(label));
      selectedLabels = [];
    }};

    const focusResidue = (key) => {{
      const meta = residueMeta[key];
      if (!meta) {{
        return;
      }}
      viewer.zoomTo({{chain: meta.chain, resi: meta.resi}}, 280);
      viewer.render();
    }};

    const updateTopRows = () => {{
      topRows.forEach((row) => {{
        const selected = selectedKeys.includes(row.dataset.key);
        row.style.background = selected ? "#eef6ff" : "transparent";
        row.style.boxShadow = selected ? "inset 3px 0 0 #0f766e" : "none";
      }});
      activityRows.forEach((row) => {{
        const selected = row.dataset.key && selectedKeys.includes(row.dataset.key);
        row.style.background = selected ? "#eef6ff" : "transparent";
        row.style.boxShadow = selected ? "inset 3px 0 0 #7c3aed" : "none";
      }});
    }};

    const renderSelectedPanel = () => {{
      if (!selectedKeys.length) {{
        selectedPanel.innerHTML = `
          <div style="padding: 12px 12px; border: 1px dashed #cbd5e1; border-radius: 12px; background: white; color: #64748b; font-size: 12px; line-height: 1.55;">
            No residue is pinned yet. Click the structure or a row in the top-residue table to keep residues highlighted and compare their GO-term attribution metrics.
          </div>
        `;
        return;
      }}

      selectedPanel.innerHTML = selectedKeys.map((key) => {{
        const meta = residueMeta[key];
        return `
          <div data-role="selected-card" data-key="${{key}}" style="padding: 11px 12px; border: 1px solid #dbe4ee; border-radius: 12px; background: white; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);">
            <div style="display:flex; align-items:center; justify-content:space-between; gap: 8px; margin-bottom: 8px;">
              <div style="font-size: 12px; font-weight: 700; color: #111827;">
                <span style="display:inline-block; width:10px; height:10px; border-radius:999px; background:${{colorOf(meta)}}; margin-right:7px; vertical-align:middle;"></span>
                #${{meta.index1}} ${{meta.aa}} | resi ${{meta.resi}} | rank #${{meta.rank}}
              </div>
              <div style="display:flex; gap: 6px;">
                <button type="button" data-action="focus" data-key="${{key}}" style="padding: 4px 8px; border: 1px solid #cbd5e1; border-radius: 999px; background: white; color: #0f172a; font-size: 11px; cursor: pointer;">Focus</button>
                <button type="button" data-action="remove" data-key="${{key}}" style="padding: 4px 8px; border: 1px solid #fecaca; border-radius: 999px; background: #fff1f2; color: #9f1239; font-size: 11px; cursor: pointer;">Unpin</button>
              </div>
            </div>
            <div style="display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 12px; font-size: 11px; color: #334155; line-height: 1.5;">
              <div><strong>Combined abs</strong><br>${{meta.score.toFixed(3)}}</div>
              <div><strong>Combined signed</strong><br>${{meta.combined_signed.toFixed(3)}}</div>
              <div><strong>Activity score (add.)</strong><br>${{meta.activity_score.toFixed(3)}}</div>
              <div><strong>Activity score (multi.)</strong><br>${{meta.activity_score_multiplicative.toFixed(3)}}</div>
              <div><strong>Sequence local</strong><br>${{meta.seq_grad.toFixed(3)}}</div>
              <div><strong>Sequence global</strong><br>${{meta.attention.toFixed(3)}}</div>
              <div><strong>Structure local (all)</strong><br>${{meta.struct_grad.toFixed(3)}}</div>
              <div><strong>Structure local (top)</strong><br>${{meta.overlap_weight.toFixed(3)}}</div>
              <div><strong>|diag| + |anti|</strong><br>${{meta.kernel.toFixed(3)}}</div>
              <div><strong>Overlap (count)</strong><br>${{meta.overlap_count.toFixed(3)}}</div>
              <div><strong>Activity site</strong><br>${{meta.activity_segment || "-"}}</div>
            </div>
          </div>
        `;
      }}).join("");

      selectedPanel.querySelectorAll("button[data-action='focus']").forEach((button) => {{
        button.addEventListener("click", () => focusResidue(button.dataset.key));
      }});
      selectedPanel.querySelectorAll("button[data-action='remove']").forEach((button) => {{
        button.addEventListener("click", () => {{
          selectedKeys = selectedKeys.filter((key) => key !== button.dataset.key);
          applyStyles();
        }});
      }});
    }};

    const applyStyles = () => {{
      viewer.setStyle({{}}, {{}});
      viewer.setStyle({{chain: chainId, hetflag: false}}, {{
        cartoon: {{
          style: "oval",
          thickness: 0.36,
          opacity: 0.96,
          colorfunc: function(atom) {{
            const key = `${{atom.chain}}:${{atom.resi}}`;
            return residueMeta[key] ? colorOf(residueMeta[key]) : "#dfe5ed";
          }},
        }},
      }});

      viewer.addStyle({{chain: chainId, hetflag: true}}, {{
        stick: {{radius: 0.18, colorscheme: "default", opacity: 0.95}},
      }});

      topResidues.forEach((entry) => {{
        const entryColor = colorOf(residueMeta[entry.key] || entry);
        viewer.addStyle({{chain: entry.chain, resi: entry.resi}}, {{
          stick: {{radius: 0.22, color: entryColor, opacity: 0.95}},
        }});
        viewer.addStyle({{chain: entry.chain, resi: entry.resi, atom: "CA"}}, {{
          sphere: {{radius: 0.42, color: entryColor, opacity: 0.78}},
        }});
      }});

      removeSelectedLabels();
      selectedKeys.forEach((key) => {{
        const meta = residueMeta[key];
        if (!meta) {{
          return;
        }}
        viewer.addStyle({{chain: meta.chain, resi: meta.resi}}, {{
          stick: {{radius: 0.29, color: "#0f172a", opacity: 1.0}},
        }});
        viewer.addStyle({{chain: meta.chain, resi: meta.resi, atom: "CA"}}, {{
          sphere: {{radius: 0.70, color: colorOf(meta), opacity: 0.98}},
        }});
        selectedLabels.push(
          viewer.addLabel(`#${{meta.index1}} ${{meta.aa}}`, {{
            position: {{x: meta.x, y: meta.y, z: meta.z}},
            backgroundColor: "rgba(15,23,42,0.92)",
            fontColor: "#ffffff",
            borderColor: meta.hotspot_color,
            borderThickness: 1,
            fontSize: 12,
            inFront: true,
            showBackground: true,
          }})
        );
      }});

      viewer.render();
      updateTopRows();
      renderSelectedPanel();
      renderSequenceTrack();
    }};

    // The sequence ruler and the tables both drive the viewer through these hooks.
    onSelectionChange = () => applyStyles();
    onFocusResidue = (key) => focusResidue(key);
    // Hovering a ruler cell highlights that residue on the model without pinning it; leaving the
    // cell just reapplies the normal styles.
    onHoverResidue = (key) => {{
      const meta = residueMeta[key];
      if (!meta) {{
        return;
      }}
      viewer.addStyle({{chain: meta.chain, resi: meta.resi}}, {{
        stick: {{radius: 0.34, color: "#0ea5e9", opacity: 1.0}},
      }});
      viewer.addStyle({{chain: meta.chain, resi: meta.resi, atom: "CA"}}, {{
        sphere: {{radius: 0.85, color: "#0ea5e9", opacity: 0.9}},
      }});
      viewer.render();
    }};
    onHoverEnd = () => applyStyles();

    const pinResidue = (key, focus) => {{
      if (!residueMeta[key]) {{
        return;
      }}
      selectedKeys = [key].concat(selectedKeys.filter((item) => item !== key)).slice(0, 8);
      applyStyles();
      if (focus) {{
        focusResidue(key);
      }}
    }};

    const toggleResidue = (key, focus) => {{
      if (!residueMeta[key]) {{
        return;
      }}
      if (selectedKeys.includes(key)) {{
        selectedKeys = selectedKeys.filter((item) => item !== key);
        applyStyles();
        return;
      }}
      pinResidue(key, focus);
    }};

    viewer.setHoverable({{chain: chainId, atom: "CA"}}, true,
      function(atom, viewerObj) {{
        const key = `${{atom.chain}}:${{atom.resi}}`;
        const meta = residueMeta[key];
        if (!meta || atom.label) {{
          return;
        }}
        const labelText =
          `Residue #${{meta.index1}} (${{meta.aa}})\\n` +
          `rank: #${{meta.rank}}\\n` +
          `resi ${{atom.resi}} chain ${{atom.chain}}\\n` +
          `combined abs: ${{meta.score.toFixed(3)}}\\n` +
          `combined signed: ${{meta.combined_signed.toFixed(3)}}\\n` +
          `sequence GxI: ${{meta.seq_grad.toFixed(3)}}\\n` +
          `structure GxI: ${{meta.struct_grad.toFixed(3)}}\\n` +
          `kernel: ${{meta.kernel.toFixed(3)}}\\n` +
          `overlap weight: ${{meta.overlap_weight.toFixed(3)}}\\n` +
          `overlap count: ${{meta.overlap_count.toFixed(3)}}\\n` +
          `activity score: ${{meta.activity_score.toFixed(3)}}\\n` +
          `activity site: ${{meta.activity_segment || "-"}}\\n` +
          `attention: ${{meta.attention.toFixed(3)}}`;
        atom.label = viewerObj.addLabel(labelText, {{
          position: atom,
          backgroundColor: "rgba(255,255,255,0.92)",
          fontColor: "#111827",
          borderColor: "#374151",
          borderThickness: 1,
          fontSize: 12,
          inFront: true,
          showBackground: true,
        }});
      }},
      function(atom, viewerObj) {{
        if (atom.label) {{
          viewerObj.removeLabel(atom.label);
          atom.label = null;
        }}
      }}
    );

    viewer.setClickable({{chain: chainId}}, true, function(atom) {{
      const key = `${{atom.chain}}:${{atom.resi}}`;
      toggleResidue(key, true);
    }});

    topRows.forEach((row) => {{
      row.addEventListener("click", () => pinResidue(row.dataset.key, true));
    }});
    activityRows.forEach((row) => {{
      if (row.dataset.key) {{
        row.addEventListener("click", () => pinResidue(row.dataset.key, true));
      }}
    }});
    clearButton.addEventListener("click", () => {{
      selectedKeys = [];
      applyStyles();
    }});


    viewer.setViewStyle({{style: "outline", width: 0.1, color: "black", maxpixels: 1}});
    viewer.zoomTo({{chain: chainId}});
    viewer.rotate(12, "y");
    viewer.rotate(-8, "x");
    applyStyles();
    viewer.zoom(1.15, 0);
    viewer.render();
  }};

  ensureViewer();
}})();
</script>
"""
    return html


def build_structure_3d_figure(analysis: dict[str, Any]):
    html = _build_structure_3d_html(analysis)
    if html:
        return HtmlFigure(html)
    return _build_structure_plotly_figure(analysis)


def _go_term_to_index_map(go_terms_mapping: list[str] | dict[Any, Any]) -> dict[str, int]:
    if isinstance(go_terms_mapping, dict):
        items = go_terms_mapping.items()
    else:
        items = enumerate(go_terms_mapping)
    mapping: dict[str, int] = {}
    for raw_idx, raw_go in items:
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        mapping[str(raw_go)] = idx
    return mapping


def _build_go_to_ontology_index(
    go_terms_mappings_by_ontology: dict[str, list[str] | dict[Any, Any]],
) -> dict[str, tuple[str, int]]:
    """Return {go_term_str: (ontology_key, term_idx)}, first ontology wins on collision."""
    result: dict[str, tuple[str, int]] = {}
    for ontology, mapping in go_terms_mappings_by_ontology.items():
        for term_str, idx in _go_term_to_index_map(mapping).items():
            if term_str not in result:
                result[term_str] = (ontology, idx)
    return result


def _resolve_custom_term_specs(
    protein_id: str,
    custom_terms: list[Any],
    primary_mapping: list[str] | dict[Any, Any],
    primary_model: Any,
    models_by_ontology: dict[str, Any],
    go_terms_mappings_by_ontology: dict[str, Any],
    go_to_ontology_index: dict[str, tuple[str, int]],
) -> tuple[list[tuple[int, Any, Any]], str]:
    """Resolve custom GO terms to (term_idx, model, mapping) triples.

    Falls back to the appropriate ontology model when a term is absent from the primary
    mapping. Returns (specs, selection_mode).
    """
    primary_go_to_idx = _go_term_to_index_map(primary_mapping)
    specs: list[tuple[int, Any, Any]] = []
    seen: set[tuple[str | None, int]] = set()

    for raw_term in custom_terms:
        if isinstance(raw_term, (int, np.integer)):
            key: tuple[str | None, int] = (None, int(raw_term))
            if key not in seen:
                seen.add(key)
                specs.append((int(raw_term), primary_model, primary_mapping))
            continue

        term_str = str(raw_term)

        idx = primary_go_to_idx.get(term_str)
        if idx is not None:
            key = ("__primary__", idx)
            if key not in seen:
                seen.add(key)
                specs.append((idx, primary_model, primary_mapping))
            continue

        if term_str in go_to_ontology_index:
            ontology, cross_idx = go_to_ontology_index[term_str]
            key = (ontology, cross_idx)
            if key not in seen:
                seen.add(key)
                specs.append((
                    cross_idx,
                    models_by_ontology[ontology],
                    go_terms_mappings_by_ontology[ontology],
                ))
            continue

        LOGGER.warning(
            "Interpretability | %s | skipping unknown GO term %r from custom list",
            protein_id,
            raw_term,
        )

    return specs, "custom_list"


def _select_term_indices(
    record: dict[str, Any],
    probs: np.ndarray,
    go_terms_mapping: list[str] | dict[Any, Any],
    *,
    threshold: float,
    top_k_fallback: int,
    max_terms_per_protein: int,
    custom_terms_by_protein: dict[str, list[Any]] | None = None,
) -> tuple[np.ndarray, str]:
    protein_id = str(record["protein_id"])
    if custom_terms_by_protein is not None and protein_id in custom_terms_by_protein:
        go_to_idx = _go_term_to_index_map(go_terms_mapping)
        selected: list[int] = []
        seen: set[int] = set()
        for raw_term in custom_terms_by_protein.get(protein_id, []):
            idx: int | None = None
            if isinstance(raw_term, (int, np.integer)):
                idx = int(raw_term)
            else:
                idx = go_to_idx.get(str(raw_term))
            if idx is None:
                LOGGER.warning("Interpretability | %s | skipping unknown GO term %r from custom list", protein_id, raw_term)
                continue
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(idx)
        return np.asarray(selected, dtype=np.int64), "custom_list"

    # Drop the ontology roots (molecular_function / cellular_component / biological_process): they
    # score ~1 for every protein, so a report on them would only crowd out informative terms. A
    # custom list is left untouched above — an explicitly requested term is always analyzed.
    is_root = np.array(
        [_resolve_go_term(go_terms_mapping, idx) in ROOT_GO_IDS for idx in range(len(probs))],
        dtype=bool,
    )
    selected = np.where((probs >= float(threshold)) & ~is_root)[0]
    if selected.size == 0:
        ranked = np.argsort(probs)[::-1]
        selected = ranked[~is_root[ranked]][: int(top_k_fallback)]
        selection = "topk_fallback"
    else:
        selected = selected[np.argsort(probs[selected])[::-1]]
        selection = "threshold"
    selected = selected[: int(max_terms_per_protein)]
    return np.asarray(selected, dtype=np.int64), selection


def _resolve_true_residue_indices(
    protein_id: str,
    go_term: str,
    custom_true_residues: dict[tuple[Any, Any], list[Any]] | None,
    residue_count: int,
) -> list[int]:
    if not custom_true_residues:
        return []
    raw_residues = custom_true_residues.get((str(protein_id), str(go_term)))
    if raw_residues is None:
        return []
    resolved: list[int] = []
    seen: set[int] = set()
    for raw_residue in raw_residues:
        try:
            residue_idx = int(raw_residue)
        except Exception:
            continue
        if residue_idx < 1 or residue_idx > int(residue_count) or residue_idx in seen:
            continue
        seen.add(residue_idx)
        resolved.append(residue_idx)
    return resolved


def _go_term_display(go_term: str, go_term_name: str | None) -> str:
    go_term = str(go_term)
    go_term_name = "" if go_term_name is None else str(go_term_name).strip()
    return f"{go_term} ({go_term_name})" if go_term_name else go_term


def save_analysis_bundle(
    analysis: dict[str, Any],
    output_dir: str | Path,
    *,
    log_runtime: bool = False,
) -> dict[str, str]:
    protein_id = str(analysis["protein_id"])
    go_term = str(analysis["go_term"])
    bundle_start = time.perf_counter()
    base_dir = Path(output_dir) / _safe_name(analysis["protein_id"]) / _safe_name(analysis["go_term"])
    base_dir.mkdir(parents=True, exist_ok=True)

    seq_plot_path = base_dir / "sequence_analysis.png"
    kernel_distmap_path = base_dir / "kernel_distmap.png"
    residues_path = base_dir / "residues.csv"
    top_residues_path = base_dir / "top_residues.csv"
    activity_sites_path = base_dir / "activity_site_candidates.csv"
    kernel_windows_path = base_dir / "top_kernel_windows.csv"
    summary_path = base_dir / "summary.json"
    structure_html_path = base_dir / "structure_analysis.html"

    step_start = time.perf_counter()
    plot_sequence_analysis(analysis, save_path=seq_plot_path)
    _log_runtime_step(log_runtime, protein_id, go_term, "save_sequence_plot", time.perf_counter() - step_start)

    step_start = time.perf_counter()
    plot_kernel_distmap_analysis(analysis, save_path=kernel_distmap_path)
    _log_runtime_step(log_runtime, protein_id, go_term, "save_distmap_plot", time.perf_counter() - step_start)

    step_start = time.perf_counter()
    analysis["residue_table"].to_csv(residues_path, index=False)
    analysis["top_residues"].to_csv(top_residues_path, index=False)
    analysis["activity_site_candidates"].to_csv(activity_sites_path, index=False)
    analysis["kernel_windows"].to_csv(kernel_windows_path, index=False)
    _log_runtime_step(log_runtime, protein_id, go_term, "save_csv_tables", time.perf_counter() - step_start)

    step_start = time.perf_counter()
    kernel_gradinput_path = base_dir / "kernel_gradinput.npy"
    np.save(kernel_gradinput_path, analysis["kernel_gradinput_matrix"])
    _log_runtime_step(log_runtime, protein_id, go_term, "save_kernel_gradinput", time.perf_counter() - step_start)

    step_start = time.perf_counter()
    fig3d = build_structure_3d_figure(analysis)
    if fig3d is not None:
        fig3d.write_html(structure_html_path, include_plotlyjs="cdn")
    _log_runtime_step(log_runtime, protein_id, go_term, "save_structure_html", time.perf_counter() - step_start)

    summary = {
        "protein_id": analysis["protein_id"],
        "go_term": analysis["go_term"],
        "go_term_name": analysis.get("go_term_name", ""),
        "term_idx": int(analysis["term_idx"]),
        "pred_prob": float(analysis["pred_prob"]),
        "struct_prob": float(analysis["struct_prob"]),
        "esm_prob": float(analysis["esm_prob"]),
        "gate": float(analysis["gate"]),
        "seq_branch_weight": float(analysis["seq_branch_weight"]),
        "struct_branch_weight": float(analysis["struct_branch_weight"]),
        "residues_csv": str(residues_path),
        "top_residues_csv": str(top_residues_path),
        "activity_site_candidates_csv": str(activity_sites_path),
        "top_kernel_windows_csv": str(kernel_windows_path),
        "sequence_plot_png": str(seq_plot_path),
        "kernel_distmap_png": str(kernel_distmap_path),
        "structure_html": str(structure_html_path) if fig3d is not None else None,
        "timings_s": analysis.get("timings_s", {}),
    }
    if not analysis["activity_site_candidates"].empty:
        top_site = analysis["activity_site_candidates"].iloc[0]
        summary["top_activity_site"] = {
            "segment_label": str(top_site["segment_label"]),
            "start_index_1based": int(top_site["start_index_1based"]),
            "end_index_1based": int(top_site["end_index_1based"]),
            "peak_residue_index_1based": int(top_site["peak_residue_index_1based"]),
            "peak_activity_score": float(top_site["peak_activity_score"]),
        }
    summary_path.write_text(json.dumps(summary, indent=2))
    bundle_elapsed = time.perf_counter() - bundle_start
    _log_runtime_step(log_runtime, protein_id, go_term, "save_bundle", bundle_elapsed)

    return {
        "output_dir": str(base_dir),
        "sequence_plot_png": str(seq_plot_path),
        "kernel_distmap_png": str(kernel_distmap_path),
        "residues_csv": str(residues_path),
        "top_residues_csv": str(top_residues_path),
        "activity_site_candidates_csv": str(activity_sites_path),
        "top_kernel_windows_csv": str(kernel_windows_path),
        "summary_json": str(summary_path),
        "structure_html": str(structure_html_path) if fig3d is not None else "",
        "save_bundle_s": float(bundle_elapsed),
    }


def build_interpretability_records(batch_ids, embeddings, distograms, masks, struct_info, max_seq_len):
    """Assemble one interpretability record per protein from an inference batch.

    Takes a batch as yielded by ``prepare_batches_for_inference(..., return_struct_info=True)`` and
    returns the per-protein dicts consumed by ``analyze_records_with_interpretability``: the padded
    model inputs (kept on the CPU, so the batch's GPU tensors can be released) plus the parsed
    structure trimmed to the residues the model actually saw. Predictions (``pred_proba``) are
    attached by the caller, which knows which ontology's scores to use.
    """
    records = []
    for idx, protein_id in enumerate(batch_ids):
        info = struct_info[idx]
        residue_df = info["residue_df"]
        mask = masks[idx].detach().cpu()
        # The model sees min(valid residues, parsed residues, MAX_SEQ_LEN) positions; reports must
        # cover exactly those (a longer protein is truncated upstream).
        residue_count = min(int(mask.sum().item()), len(residue_df), int(max_seq_len))
        coords = info.get("coords")
        if coords is None:
            coords = residue_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        chain = residue_df["chain"].iloc[0] if "chain" in residue_df.columns and not residue_df.empty else ""
        records.append(
            {
                "protein_id": protein_id,
                "cif_path": info["cif_path"],
                "selected_chain": chain,
                "sequence": info["sequence"][:residue_count],
                "residue_df": residue_df.iloc[:residue_count].copy(),
                "coords": coords[:residue_count],
                "residue_count": residue_count,
                "padded_embedding": embeddings[idx].detach().cpu(),
                "padded_distogram": distograms[idx].detach().cpu(),
                "mask": mask,
                "was_truncated": len(residue_df) > residue_count,
            }
        )
    return records


def analyze_records_with_interpretability(
    model,
    protein_records: list[dict[str, Any]],
    go_terms_mapping: list[str] | dict[Any, Any],
    *,
    output_dir: str | Path = "interpretability_reports",
    threshold: float = 0.5,
    max_terms_per_protein: int = 3,
    top_k_fallback: int = 3,
    smooth_window: int = 1,
    top_windows: int = 12,
    top_residues: int = 20,
    keep_in_memory: bool = False,
    go_name_map: dict[str, str] | pd.Series | None = None,
    custom_terms_by_protein: dict[str, list[Any]] | None = None,
    custom_true_residues: dict[tuple[Any, Any], list[Any]] | None = None,
    log_runtime: bool = False,
    save_in_background: bool = False,
    save_workers: int = 1,
    models_by_ontology: dict[str, Any] | None = None,
    go_terms_mappings_by_ontology: dict[str, Any] | None = None,
    write_summary: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[str, str], Any]]:
    """Analyze the selected GO terms of every record and save one report bundle per term.

    Term selection per protein: the GO ids listed in ``custom_terms_by_protein`` if the protein
    appears there, else every term with probability >= ``threshold`` (capped at
    ``max_terms_per_protein``), else the ``top_k_fallback`` highest-scoring terms. A custom term
    that is absent from ``go_terms_mapping`` is looked up in ``go_terms_mappings_by_ontology`` and
    analyzed with that ontology's model from ``models_by_ontology``, so a caller can request terms
    across aspects in one pass.

    Args:
        model: fusion model for the primary ontology (used for terms of ``go_terms_mapping``).
        protein_records: dicts holding, per protein, the padded model inputs (``padded_embedding``,
            ``padded_distogram``, ``mask``), the parsed structure (``residue_df``, ``sequence``,
            ``residue_count``, ``cif_path``) and its predictions (``pred_proba``).
        go_terms_mapping: ``{label index: GO id}`` of the primary ontology.
        output_dir: report root; each term lands in ``<output_dir>/<protein_id>/<GO_id>/``.
        threshold / max_terms_per_protein / top_k_fallback: term-selection knobs (see above).
        smooth_window: moving-average window applied to the per-residue curves.
        top_windows / top_residues: rows kept in the kernel-window / top-residue tables.
        keep_in_memory: cache the full analysis dicts instead of just the written paths.
        go_name_map: ``{GO id: name}`` used to label reports.
        custom_terms_by_protein: ``{protein_id: [GO id, ...]}`` explicit term selection.
        custom_true_residues: ``{(protein_id, GO id): [residue index, ...]}`` reference residues
            (1-based) drawn on the plots for benchmarking.
        log_runtime: log per-step timings for every term.
        save_in_background / save_workers: save a term's artifacts in worker threads while the next
            term is analyzed.
        models_by_ontology / go_terms_mappings_by_ontology: per-aspect models and label maps used to
            resolve custom terms outside the primary ontology.
        write_summary: write ``interpretability_summary.csv`` under ``output_dir``. Set False when
            the caller collects the returned frames itself (e.g. to stream one CSV across batches).

    Returns:
        ``(summary_df, cache)``: one summary row per analyzed protein/GO-term pair, and
        ``{(protein_id, GO id): analysis-or-paths}``.
    """
    summary_rows = []
    cache: dict[tuple[str, str], Any] = {}
    executor = ThreadPoolExecutor(max_workers=max(1, int(save_workers))) if save_in_background else None

    go_to_ontology_index: dict[str, tuple[str, int]] | None = None
    if models_by_ontology is not None and go_terms_mappings_by_ontology is not None:
        go_to_ontology_index = _build_go_to_ontology_index(go_terms_mappings_by_ontology)

    def _resolve_and_record(item: dict[str, Any]) -> None:
        """Resolve the save future (if any), build the summary row, update cache."""
        analysis = item["analysis"]
        future = item["future"]
        if future is not None:
            paths = future.result()
        else:
            paths = save_analysis_bundle(analysis, output_dir=output_dir, log_runtime=bool(log_runtime))
        save_elapsed = float(paths.get("save_bundle_s", 0.0))
        total_elapsed = float(item["analysis_elapsed"]) if save_in_background else float(item["analysis_elapsed"]) + save_elapsed
        _log_runtime_step(bool(log_runtime), str(item["record"]["protein_id"]), item["go_term"], "term_total", total_elapsed)
        top_residue = item["top_residue"]
        top_activity = item["top_activity"]
        # One progress line per finished report: a term takes seconds to tens of seconds, so this
        # is the run's main feedback channel (per-step timings need log_runtime).
        LOGGER.info(
            f"Report | {str(item['record']['protein_id']):<20} | "
            f"{_go_term_display(item['go_term'], item['go_term_name']):<40} | "
            f"prob={float(analysis['pred_prob']):.3f} | "
            f"top residue={int(top_residue['residue_index_1based'])}{str(top_residue['residue_aa'])} | "
            f"{total_elapsed:>7.3f}s"
        )
        summary_rows.append(
            {
                "protein_id": item["record"]["protein_id"],
                "selection": item["selection"],
                "rank": item["rank"],
                "go_term": item["go_term"],
                "go_term_name": item["go_term_name"],
                "term_idx": item["term_idx"],
                "pred_prob": float(analysis["pred_prob"]),
                "struct_prob": float(analysis["struct_prob"]),
                "esm_prob": float(analysis["esm_prob"]),
                "gate": float(analysis["gate"]),
                "top_residue_index_1based": int(top_residue["residue_index_1based"]),
                "top_residue_aa": str(top_residue["residue_aa"]),
                "top_residue_score": float(top_residue["combined_abs"]),
                "top_activity_site_label": str(top_activity["segment_label"]) if top_activity is not None else "",
                "top_activity_site_start_1based": int(top_activity["start_index_1based"]) if top_activity is not None else None,
                "top_activity_site_end_1based": int(top_activity["end_index_1based"]) if top_activity is not None else None,
                "top_activity_peak_residue_1based": int(top_activity["peak_residue_index_1based"]) if top_activity is not None else None,
                "top_activity_peak_score": float(top_activity["peak_activity_score"]) if top_activity is not None else None,
                "analyze_go_term_s": item["analysis_elapsed"],
                "save_bundle_s": save_elapsed,
                "term_total_s": total_elapsed,
                "forward_and_grad_s": float(analysis.get("timings_s", {}).get("forward_and_grad", 0.0)),
                "kernel_grad_x_input_s": float(analysis.get("timings_s", {}).get("kernel_grad_x_input", 0.0)),
                "postprocess_and_reports_s": float(analysis.get("timings_s", {}).get("postprocess_and_reports", 0.0)),
                "residues_csv": paths["residues_csv"],
                "sequence_plot_png": paths["sequence_plot_png"],
                "top_residues_csv": paths["top_residues_csv"],
                "activity_site_candidates_csv": paths["activity_site_candidates_csv"],
                "top_kernel_windows_csv": paths["top_kernel_windows_csv"],
                "structure_html": paths["structure_html"],
                "summary_json": paths["summary_json"],
            }
        )
        cache[(item["record"]["protein_id"], item["go_term"])] = analysis if keep_in_memory else paths

    try:
        for record in protein_records:
            protein_start = time.perf_counter()
            protein_id = str(record["protein_id"])
            probs = np.asarray(record["pred_proba"], dtype=np.float32)

            if (
                custom_terms_by_protein is not None
                and protein_id in custom_terms_by_protein
                and go_to_ontology_index is not None
            ):
                term_specs, selection = _resolve_custom_term_specs(
                    protein_id,
                    custom_terms_by_protein[protein_id],
                    go_terms_mapping,
                    model,
                    models_by_ontology,
                    go_terms_mappings_by_ontology,
                    go_to_ontology_index,
                )
            else:
                selected, selection = _select_term_indices(
                    record,
                    probs,
                    go_terms_mapping,
                    threshold=float(threshold),
                    top_k_fallback=int(top_k_fallback),
                    max_terms_per_protein=int(max_terms_per_protein),
                    custom_terms_by_protein=custom_terms_by_protein,
                )
                term_specs = [(int(idx), model, go_terms_mapping) for idx in selected]

            for rank, (term_idx, term_model, term_mapping) in enumerate(term_specs, start=1):
                term_start = time.perf_counter()
                go_term = _resolve_go_term(term_mapping, term_idx)
                go_term_name = _resolve_go_name(go_term, go_name_map)
                analysis = analyze_go_term(
                    term_model,
                    record,
                    term_idx,
                    go_term,
                    smooth_window=int(smooth_window),
                    top_windows=int(top_windows),
                    top_residues=int(top_residues),
                    log_runtime=bool(log_runtime),
                )
                analysis["go_term_name"] = go_term_name
                analysis["true_residue_indices_1based"] = _resolve_true_residue_indices(
                    str(record["protein_id"]),
                    str(go_term),
                    custom_true_residues,
                    int(analysis["residue_count"]),
                )
                analysis_elapsed = time.perf_counter() - term_start
                future = executor.submit(
                    save_analysis_bundle,
                    analysis,
                    output_dir,
                    log_runtime=bool(log_runtime),
                ) if save_in_background and executor is not None else None
                top_residue = analysis["top_residues"].iloc[0]
                top_activity = analysis["activity_site_candidates"].iloc[0] if not analysis["activity_site_candidates"].empty else None
                # Save and record immediately once all data for this term are ready.
                _resolve_and_record(
                    {
                        "future": future,
                        "analysis": analysis,
                        "analysis_elapsed": analysis_elapsed,
                        "record": record,
                        "selection": selection,
                        "rank": rank,
                        "term_idx": int(term_idx),
                        "go_term": go_term,
                        "go_term_name": go_term_name,
                        "top_residue": top_residue,
                        "top_activity": top_activity,
                    }
                )

            _log_runtime_step(
                bool(log_runtime),
                str(record["protein_id"]),
                "all_selected_terms",
                "protein_total",
                time.perf_counter() - protein_start,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    summary_df = pd.DataFrame(summary_rows)
    if write_summary:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_root / "interpretability_summary.csv", index=False)
    return summary_df, cache
