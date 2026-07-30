"""deepFRI2 protein function prediction models.

This module defines the three models that make up deepFRI2 together with the
helper functions they rely on:

- ``StructuralProber``  -- reads a residue-residue distogram with a bank of
  learnable 2D kernels and predicts function from structural motifs (the
  "kernel model").
- ``SequenceAnalyzer``  -- pools per-residue ESM embeddings and predicts
  function from sequence (the "ESM model").
- ``FusionModel``       -- combines a frozen ``StructuralProber`` and a frozen
  ``SequenceAnalyzer`` through a small trainable per-label gate.

All three expose the same ``forward(inputs, disto, mask, ...)`` signature so
they are interchangeable at inference time, and each supports optional
interpretability outputs via ``return_attr`` / ``return_details``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.fft import fft

logger = getLogger("inference")


# =================
# General utilities
# =================

def load_run_weights(model, ontology, run_name: str, strict: bool = False, params_dir="params"):
    """Load weights saved under ``<params_dir>/<ontology>/<run_name>.pth`` into ``model``.

    Accepts either a raw ``state_dict`` or a checkpoint dict containing a
    ``"state_dict"`` key. When ``strict`` is False (the default), the number of
    missing/unexpected keys is printed instead of raising, which is convenient
    when loading sub-model weights into a composite model. ``params_dir`` defaults to
    the relative ``"params"`` folder; pass an absolute path for CWD-independent loading.
    """
    ckpt = Path(params_dir) / ontology / f"{run_name}.pth"
    # weights_only=True: our checkpoints are plain tensor state_dicts, so this is safe
    # and avoids the arbitrary-pickle security risk (and its FutureWarning).
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    out = model.load_state_dict(sd, strict=strict)
    if not strict:
        logger.debug(f"{run_name}: missing={len(out.missing_keys)} unexpected={len(out.unexpected_keys)}")
    return model


# ===========================
# Structural prober utilities
# ===========================

def prepare_template_kernel(tpl: torch.Tensor) -> torch.Tensor:
    """Normalize a kernel to zero mean and unit variance (cosine-like matching)."""
    flat = tpl.reshape(-1)
    eps = 1e-6
    mean = flat.mean()
    std = flat.std().clamp_min(eps)
    return (tpl - mean) / std


def grid_starts_from_shape(nH: int, nW: int, stride: int, device) -> torch.Tensor:
    """Return the (row, col) top-left coordinate of every sliding window.

    Output shape is ``(nH * nW, 2)``.
    """
    rows = torch.arange(nH, device=device) * stride
    cols = torch.arange(nW, device=device) * stride
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([rr, cc], dim=-1).reshape(-1, 2)  # (W, 2)


def effective_length_from_mask(mask_1d: torch.Tensor) -> torch.Tensor:
    """Number of valid residues per sequence (clamped to >= 1).

    ``mask_1d`` is ``(B, N)`` with 1 = valid residue, 0 = padding.
    """
    return mask_1d.float().sum(dim=1).clamp_min(1.0)


def build_diag_band_mask(N: int, m: int, device, dtype, bandwidth: Optional[int] = None) -> torch.Tensor:
    """Build a ``(1, 1, N, N)`` mask covering a band around the main diagonal.

    The band is wide enough to fully contain any ``m x m`` kernel whose top-left
    corner sits on the diagonal at ``(i, i)`` (half-width ``m - 1``). ``bandwidth``
    overrides the automatically computed full band width of ``2*m - 1``.
    """
    if bandwidth is None:
        # ensure |i - j| <= m - 1  ->  full band width = 2*m - 1
        bandwidth = 2 * (m - 1) + 1
    half_w = (bandwidth - 1) // 2
    i = torch.arange(N, device=device)
    j = torch.arange(N, device=device)
    ii, jj = torch.meshgrid(i, j, indexing="ij")
    band = (jj - ii).abs() <= half_w
    return band.to(dtype=dtype).view(1, 1, N, N)


def build_upper_triangle_mask(N: int, device, dtype, exclude_diag: bool = True) -> torch.Tensor:
    """Build a ``(1, 1, N, N)`` upper-triangle mask (optionally excluding the diagonal)."""
    i = torch.arange(N, device=device)
    j = torch.arange(N, device=device)
    ii, jj = torch.meshgrid(i, j, indexing="ij")
    ut = (jj > ii) if exclude_diag else (jj >= ii)
    return ut.to(dtype=dtype).view(1, 1, N, N)


def build_kernel_bank(kernels_2d: List[torch.Tensor], device, dtype) -> Optional[torch.Tensor]:
    """Stack a list of ``(m, m)`` kernels into a normalized ``(K, 1, m, m)`` bank.

    Each kernel is normalized to zero mean / unit std via ``prepare_template_kernel``.
    Returns ``None`` if the list is empty.
    """
    bank = []
    for tpl in kernels_2d:
        kz = prepare_template_kernel(tpl.to(device=device, dtype=dtype))
        bank.append(kz.unsqueeze(0).unsqueeze(0))  # (1, 1, m, m)
    if len(bank) == 0:
        return None
    return torch.cat(bank, dim=0)  # (K, 1, m, m)


class KernelParam(nn.Module):
    """A single learnable ``(m x m)`` kernel matrix.

    Supports optional diagonal zeroing (for diagonal-band kernels, so they cannot
    exploit the 1s on the distogram's own diagonal), symmetry enforcement, and a
    positivity constraint.
    """

    def __init__(
        self,
        m: int,
        enforce_sym: bool = False,
        positive: bool = False,
        mode: str = "relu",
        zero_diag: bool = False,
    ):
        """
        Args:
            m: kernel size.
            enforce_sym: enforce a symmetric kernel after transformations.
            positive: enforce non-negative weights.
            mode: positivity transform ('relu', 'softplus', or 'exp').
            zero_diag: whether to zero out the diagonal (for diag-type kernels).
        """
        super().__init__()
        self.m = m
        self.enforce_sym = enforce_sym
        self.positive = positive
        self.mode = mode.lower()
        self.zero_diag = zero_diag
        self.weight = nn.Parameter(torch.randn(m, m) * 0.02)

    def _positivity(self, W: torch.Tensor) -> torch.Tensor:
        if not self.positive:
            return W
        if self.mode == "relu":
            return F.relu(W)
        elif self.mode == "softplus":
            return F.softplus(W)
        elif self.mode == "exp":
            return torch.exp(W)
        else:
            raise ValueError(f"Unknown positivity mode: {self.mode}")

    def forward(self) -> torch.Tensor:
        M = self._positivity(self.weight)

        # Optionally zero out the diagonal.
        if self.zero_diag:
            M = M - torch.diag_embed(torch.diag(M))

        # Optionally enforce symmetry.
        if self.enforce_sym:
            M = 0.5 * (M + M.T)

        return M


# ===========================
# Sequence analyzer utilities
# ===========================

class Pooling(nn.Module):
    """Pool a ``(B, L, D)`` sequence of residue embeddings into ``(B, D')``.

    Supported strategies:
    - ``cls``:        use only the <cls> token embedding (position 0).
    - ``mean``:       mask-aware mean over residues.
    - ``max``:        mask-aware max over residues.
    - ``attn_light``: lightweight single-head additive attention pooling. It is
                      mask-aware, initialized close to mean pooling, and stores
                      per-residue attention weights in ``self.last_attn_weights``
                      for interpretability. Output dim equals ``embedding_dim``.
    """

    def __init__(self, pooling_method='mean', embedding_dim=1280, **kwargs):
        super().__init__()
        self.pooling_method = pooling_method
        self.embedding_dim = embedding_dim

        # For interpretability/debugging (filled on forward for attention-based poolers).
        self.last_attn_weights = None

        if pooling_method == 'attn_light':
            # Lightweight additive attention pooling.
            # Key properties:
            # - mask-aware
            # - initialized near uniform weights => starts close to mean pooling
            # - stores per-residue weights in `self.last_attn_weights`
            self.attn_hidden = kwargs.get("attn_hidden", 64)
            self.attn_temperature = kwargs.get("attn_temperature", 2.0)
            self.attn_ln = nn.LayerNorm(self.embedding_dim)
            self.attn_proj = nn.Linear(self.embedding_dim, self.attn_hidden)
            self.attn_score = nn.Linear(self.attn_hidden, 1, bias=False)
            self.attn_dropout = nn.Dropout(p=0.1)
            # Init near mean pooling but NOT exactly uniform (avoid zero-gradient symmetry):
            # small random scores => softmax ~ uniform initially, but trainable.
            nn.init.normal_(self.attn_proj.weight, mean=0.0, std=2e-2)
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.normal_(self.attn_score.weight, mean=0.0, std=2e-2)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor ``(batch_size, seq_len, embedding_dim)``.
            mask: Optional mask ``(batch_size, seq_len)``, True = valid, False = pad.
        Returns:
            pooled: Tensor ``(batch_size, pooled_dim)``.
        """
        if self.pooling_method == 'cls':
            # Use only the <cls> token representation (assumed to be at position 0).
            # Note: the mask is ignored here on purpose.
            self.last_attn_weights = None
            pooled = x[:, 0, :]

        elif self.pooling_method == 'mean':
            if mask is not None:
                mask_exp = mask.unsqueeze(-1).float()
                x_masked = x * mask_exp
                pooled = x_masked.sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)

        elif self.pooling_method == 'max':
            if mask is not None:
                # Use a dtype-safe -inf (important for fp16 / autocast).
                neg_inf = torch.finfo(x.dtype).min
                x_masked = x.masked_fill(~mask.unsqueeze(-1), neg_inf)
                pooled = x_masked.max(dim=1)[0]
            else:
                pooled = x.max(dim=1)[0]

        elif self.pooling_method == 'attn_light':
            x_norm = self.attn_ln(x)
            scores = self.attn_score(torch.tanh(self.attn_proj(x_norm))).squeeze(-1)
            scores = scores / float(self.attn_temperature)

            if mask is not None:
                neg_inf = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(~mask, neg_inf)

            weights = F.softmax(scores, dim=1)  # (B, L)
            weights = self.attn_dropout(weights)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            self.last_attn_weights = weights.detach()

            pooled = torch.einsum("bl,bld->bd", weights, x)

        else:
            pooled = x.mean(dim=1)

        return pooled


# ================================
# Structural prober (kernel model)
# ================================

class StructuralProber(nn.Module):
    """Distogram-based function predictor using two banks of 2D kernels.

    Two kernel groups scan a residue-residue distogram in complementary regions:
    - Diagonal-band kernels (group "diag") slide only within a band around the
      main diagonal, capturing local/contact-order structure.
    - Off-diagonal kernels (group "anti") slide only in the upper triangle,
      capturing long-range contacts.

    For each group, per-window match scores are reduced to a small set of scalar
    features (see ``diag_feats`` / ``anti_feats``); those features feed a small
    MLP head (``BatchNorm -> Linear -> ReLU -> Linear``) that produces per-label logits.

    Features:
    - Mask-aware handling of padding via a 2D pairwise mask.
    - Grouped conv2d over all kernels in a group for speed.
    - Location features normalized by effective (unpadded) length.
    - Optional per-residue attribution for interpretability (``return_attr``).
    - Optional symmetry / positivity constraints per group.
    - Optional diagonal reparameterization (zero diagonal weights).

    Args:
        num_labels: number of output labels (GO terms).
        arch_to_size_diag: maps each diagonal kernel id to a nominal size. Only the
            keys (kernel ids) and their count matter; the canonical size below is
            what is actually used.
        arch_to_size_anti: same as above for off-diagonal kernels.
        canonical_diag_ms / canonical_anti_ms: kernel sizes actually used per group.
        diag_stride / anti_stride: sliding-window strides per group.
        diag_feats / anti_feats: names of scalar features computed per group.
        peak_thresh: threshold used by the "count" feature.
        topk_k: k used by top-k features and by attribution.
        frozen_kernels: if True, use fixed user-provided kernels instead of learnable
            ones (requires ``user_kernels_diag`` / ``user_kernels_anti``).
        user_kernels_diag / user_kernels_anti: frozen kernels, keyed by kernel id.
        amp_dtype: autocast dtype for the conv/feature block (None disables autocast).
        enforce_symmetry_* / enforce_positivity_*: per-group kernel constraints.
        diag_bandwidth: override for the diagonal band width.
        reparam_zero_diag_for_diag_kernels: reparameterize diagonal kernels to have
            zero diagonal weights (prevents leveraging the 1s on the distogram diagonal).
        use_coverage_weighting: down-weight windows with few valid entries.
    """

    def __init__(
        self,
        num_labels: int,
        arch_to_size_diag: Dict[str, int],
        arch_to_size_anti: Dict[str, int],
        canonical_diag_ms: int = 32,
        diag_stride: int = 1,
        canonical_anti_ms: int = 16,
        anti_stride: int = 2,
        diag_feats: Tuple[str, ...] = ("max", "mean", "count", "topk_mean", "topk_std", "argmax_r", "argmax_c"),
        anti_feats: Tuple[str, ...] = ("max", "mean", "count", "topk_mean", "topk_std", "argmax_r", "argmax_c"),
        peak_thresh: float = 0.6,
        topk_k: int = 3,
        frozen_kernels: bool = False,
        user_kernels_diag: Optional[Dict[str, torch.Tensor]] = None,  # key arch_id -> (m_d, m_d)
        user_kernels_anti: Optional[Dict[str, torch.Tensor]] = None,  # key arch_id -> (m_a, m_a)
        amp_dtype: Optional[torch.dtype] = torch.bfloat16,
        enforce_symmetry_diag: bool = False,
        enforce_symmetry_anti: bool = False,
        enforce_positivity_diag: bool = False,
        enforce_positivity_anti: bool = False,
        diag_bandwidth: Optional[int] = None,
        reparam_zero_diag_for_diag_kernels: bool = True,
        use_coverage_weighting: bool = False,
        hidden_dim_in: int = 512,
        hidden_dim_out: int = 512,
        bn1_clamp: float = 10,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.arch_ids_diag = list(arch_to_size_diag.keys())
        self.arch_ids_anti = list(arch_to_size_anti.keys())
        self.canonical_diag_ms = int(canonical_diag_ms)
        self.canonical_anti_ms = int(canonical_anti_ms)
        self.diag_stride = int(diag_stride)
        self.anti_stride = int(anti_stride)

        self.peak = peak_thresh
        self.k_top = topk_k
        self.diag_features = diag_feats
        self.anti_features = anti_feats
        self.amp_dtype = amp_dtype
        self.diag_bandwidth = diag_bandwidth
        self.use_coverage_weighting = use_coverage_weighting

        # Per-group kernel configuration.
        self.frozen = frozen_kernels
        self.enforce_sym_diag = enforce_symmetry_diag
        self.enforce_sym_anti = enforce_symmetry_anti
        self.enforce_pos_diag = enforce_positivity_diag
        self.enforce_pos_anti = enforce_positivity_anti
        self.reparam_zero_diag_for_diag_kernels = reparam_zero_diag_for_diag_kernels
        self.bn1_clamp = bn1_clamp
        self.hidden_dim_in = hidden_dim_in
        self.hidden_dim_out = hidden_dim_out
        self.relu = nn.ReLU()
        
        m_d = self.canonical_diag_ms
        m_a = self.canonical_anti_ms

        if self.frozen:
            # Store frozen kernels as buffers.
            if user_kernels_diag is None or user_kernels_anti is None:
                raise ValueError("Frozen mode requires user_kernels_diag and user_kernels_anti.")
            self.buffers_diag = {}
            for a in self.arch_ids_diag:
                tpl = user_kernels_diag[a].float()
                if tpl.shape != (m_d, m_d):
                    raise ValueError(f"diag kernel for {a} must be {(m_d, m_d)}")
                if self.enforce_sym_diag:
                    tpl = 0.5 * (tpl + tpl.T)
                # If reparameterizing to zero diagonal, apply it to frozen kernels too.
                if self.reparam_zero_diag_for_diag_kernels:
                    tpl = tpl - torch.diag_embed(torch.diag(tpl))
                key = f"diag::{a}"
                self.register_buffer(key, tpl.clone(), persistent=True)
                self.buffers_diag[a] = key

            self.buffers_anti = {}
            for a in self.arch_ids_anti:
                tpl = user_kernels_anti[a].float()
                if tpl.shape != (m_a, m_a):
                    raise ValueError(f"anti kernel for {a} must be {(m_a, m_a)}")
                if self.enforce_sym_anti:
                    tpl = 0.5 * (tpl + tpl.T)
                key = f"anti::{a}"
                self.register_buffer(key, tpl.clone(), persistent=True)
                self.buffers_anti[a] = key

            self.params_diag = None
            self.params_anti = None

        else:
            # Trainable kernels: one KernelParam submodule per kernel id, per group.
            if self.reparam_zero_diag_for_diag_kernels:
                self.params_diag = nn.ModuleDict()
                for a in self.arch_ids_diag:
                    self.params_diag[a] = KernelParam(m_d, enforce_sym=self.enforce_sym_diag,
                                                      positive=self.enforce_pos_diag, zero_diag=True)
            else:
                # No reparam: store plain trainable matrices.
                self.params_diag = nn.ParameterDict()
                for a in self.arch_ids_diag:
                    p = nn.Parameter(torch.randn(m_d, m_d) * 0.02)
                    self.params_diag[a] = p

            self.params_anti = nn.ModuleDict()
            for a in self.arch_ids_anti:
                self.params_anti[a] = KernelParam(m_a, enforce_sym=self.enforce_sym_anti,
                                                  positive=self.enforce_pos_anti, zero_diag=False)

            self.buffers_diag = None
            self.buffers_anti = None

        # Classifier head input dim.
        feat_dim_diag = len(self.diag_features)
        feat_dim_anti = len(self.anti_features)
        in_dim = len(self.arch_ids_diag) * feat_dim_diag + len(self.arch_ids_anti) * feat_dim_anti
        self.fc1 = nn.Linear(in_dim, self.hidden_dim_in)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.out = nn.Linear(self.hidden_dim_out, self.num_labels)
        self.bn1_clamp = 10.0  

        self.ARCHITECTURE = {
            "name": "Dual kernel conv2d: diagonal band + upper triangle (grouped conv, mask-aware)",
            "canonical_diag_ms": self.canonical_diag_ms,
            "diag_stride": self.diag_stride,
            "canonical_anti_ms": self.canonical_anti_ms,
            "anti_stride": self.anti_stride,
            "enforce_sym_diag": self.enforce_sym_diag,
            "enforce_sym_anti": self.enforce_sym_anti,
            "enforce_pos_diag": self.enforce_pos_diag,
            "enforce_pos_anti": self.enforce_pos_anti,
            "reparam_zero_diag_for_diag_kernels": self.reparam_zero_diag_for_diag_kernels,
            "coverage_weighting": self.use_coverage_weighting,
        }

    # ----- Kernel accessors -----

    def _tpl_diag(self, arch_id: str, device, dtype):
        if self.frozen:
            return getattr(self, self.buffers_diag[arch_id]).to(device=device, dtype=dtype)
        else:
            if self.reparam_zero_diag_for_diag_kernels:
                # Submodule that returns (m, m).
                return self.params_diag[arch_id]()
            else:
                M = self.params_diag[arch_id].to(device=device, dtype=dtype)
                return 0.5 * (M + M.T) if self.enforce_sym_diag else M

    def _tpl_anti(self, arch_id: str, device, dtype):
        if self.frozen:
            return getattr(self, self.buffers_anti[arch_id]).to(device=device, dtype=dtype)
        else:
            return self.params_anti[arch_id]()

    # ----- Feature computation -----

    def _features_from_scores(self, scores_flat: torch.Tensor, mask_flat: torch.Tensor,
                              starts: torch.Tensor, m: int, Leff: torch.Tensor, N_pad: int,
                              features: list[str], soft_beta: float = 3.0, peak_thresh: float = 0.5):
        """Reduce per-window match scores to a set of scalar features.

        Args:
            scores_flat: ``(B, W)`` patch scores (from ``S * W_mask``).
            mask_flat:   ``(B, W)`` same layout, 1 for valid patches.
            starts:      ``(W, 2)`` patch top-left coordinates.
            m:           patch size.
            Leff:        ``(B,)`` effective (unpadded) protein lengths.
            N_pad:       padded protein length (FFT base length).
            features:    names of features to compute.

        Returns:
            A list of ``(B,)`` tensors, one per requested feature (in order).
        """
        B, W = scores_flat.shape
        device = scores_flat.device
        feats = []

        valid = mask_flat.float()
        counts = valid.sum(dim=1).clamp_min(1.0)

        # ---------------- top-K preparation ---------------- #
        need_topk = any(f in features for f in (
            "topk_mean", "topk_std", "argmax_r", "argmax_c",
            "soft_x", "soft_y", "sigma_x", "sigma_y", "rho", "period",
            "freq", "freq_amp"
        ))

        if need_topk:
            masked_scores = scores_flat.masked_fill(valid < 0.5, -float("inf"))
            k = min(max(1, self.k_top), W)
            vals, idxs = torch.topk(masked_scores, k=k, dim=1)
            argmax_idx = masked_scores.argmax(dim=1)
            argmax_pos = starts[argmax_idx]  # (B, 2)

        # ---------------- coordinate grid normalized by Leff ---------------- #
        starts_f = starts.float().to(device)
        x_coords = (starts_f[:, 1] + m / 2.0).view(1, W)
        y_coords = (starts_f[:, 0] + m / 2.0).view(1, W)
        x_coords = x_coords / Leff[:, None].clamp_min(1.0)
        y_coords = y_coords / Leff[:, None].clamp_min(1.0)

        # ---------------- soft weighting & spatial spreads ---------------- #
        if any(f in features for f in ("soft_x", "soft_y", "sigma_x", "sigma_y", "rho")):
            logits = soft_beta * scores_flat
            logits = logits.masked_fill(valid < 0.5, float("-inf"))
            weights = F.softmax(logits, dim=1)
            weights = weights * valid
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

            x_soft = (weights * x_coords).sum(dim=1)
            y_soft = (weights * y_coords).sum(dim=1)

            sigma_x = torch.sqrt(((weights * (x_coords - x_soft[:, None])**2)
                                  .sum(dim=1)).clamp_min(1e-8)) if "sigma_x" in features else None
            sigma_y = torch.sqrt(((weights * (y_coords - y_soft[:, None])**2)
                                  .sum(dim=1)).clamp_min(1e-8)) if "sigma_y" in features else None
            if "rho" in features and sigma_x is not None and sigma_y is not None:
                cov_xy = (weights * (x_coords - x_soft[:, None]) *
                          (y_coords - y_soft[:, None])).sum(dim=1)
                rho_xy = cov_xy / (sigma_x * sigma_y + 1e-8)
            else:
                rho_xy = torch.zeros(B, device=device)
        else:
            x_soft = y_soft = sigma_x = sigma_y = rho_xy = None

        # ---------------- periodicity & leading frequency ---------------- #
        if any(f in features for f in ("period", "freq", "freq_amp")):
            mean_valid = (scores_flat * valid).sum(dim=1, keepdim=True) / counts[:, None]
            sig = (scores_flat - mean_valid) * valid

            # pad/trim to N_pad to standardize the FFT base length
            if W < N_pad:
                pad = torch.zeros(B, N_pad - W, device=device)
                sig = torch.cat([sig, pad], dim=1)
            elif W > N_pad:
                sig = sig[:, :N_pad]

            fft_mag = torch.abs(fft(sig, dim=1))
            fft_mag[:, 0] = 0.0

            # leading amplitude and frequency index (safe handling of empty signal)
            amp, idx = fft_mag.max(dim=1)
            no_signal = (amp == 0) | (~torch.isfinite(amp))
            idx = idx.float()
            freq = torch.where(no_signal, torch.zeros_like(idx), idx / N_pad)   # cycles per residue
            amp_norm = torch.where(no_signal, torch.zeros_like(amp),
                                   amp / (fft_mag.sum(dim=1) + 1e-6))

            # legacy "period" feature for compatibility
            periodicity = amp / (fft_mag.sum(dim=1) + 1e-6)

        # ---------------- scalar statistics ---------------- #
        for name in features:
            if name == "max":
                feats.append(scores_flat.masked_fill(valid < 0.5, -float("inf")).max(dim=1).values)
            elif name == "mean":
                feats.append((scores_flat * valid).sum(dim=1) / counts)
            elif name == "count":
                feats.append(((scores_flat > peak_thresh) & (valid > 0.5)).float().sum(dim=1))
            elif name == "topk_mean":
                feats.append(vals.mean(dim=1))
            elif name == "topk_std":
                feats.append(vals.std(dim=1))
            elif name == "argmax_r":
                feats.append((argmax_pos[:, 0].float() + m/2.0) / Leff.clamp_min(1.0))
            elif name == "argmax_c":
                feats.append((argmax_pos[:, 1].float() + m/2.0) / Leff.clamp_min(1.0))
            elif name == "soft_x" and x_soft is not None:
                feats.append(x_soft)
            elif name == "soft_y" and y_soft is not None:
                feats.append(y_soft)
            elif name == "sigma_x" and sigma_x is not None:
                feats.append(sigma_x)
            elif name == "sigma_y" and sigma_y is not None:
                feats.append(sigma_y)
            elif name == "rho" and rho_xy is not None:
                feats.append(rho_xy)
            elif name == "period":
                feats.append(periodicity)
            elif name == "freq":
                feats.append(freq)        # leading frequency (cycles per residue)
            elif name == "freq_amp":
                feats.append(amp_norm)    # normalized amplitude
            elif name == "len_rel":
                feats.append(Leff / float(N_pad))

        return feats

    # ----- Forward -----

    def forward(
        self,
        inputs,
        disto,
        mask,
        return_attr: bool = False,
        use_diag: Optional[bool] = None,
        use_anti: Optional[bool] = None,
        return_details: bool = False,
    ):
        """
        Args:
            inputs: ``(B, N, D)`` residue embeddings (unused here; kept for a common signature).
            disto:  ``(B, N, N)`` residue-residue distogram.
            mask:   ``(B, N)`` where 1 = valid residue, 0 = padding.
            return_attr: also return per-residue attribution for both groups.
            use_diag / use_anti: enable/disable each kernel group (default: both on).
            return_details: also return a dict of intermediate tensors for interpretability.
        """
        use_diag = True if use_diag is None else bool(use_diag)
        use_anti = True if use_anti is None else bool(use_anti)
        B, N, _ = disto.shape
        assert disto.shape == (B, N, N), "disto must be (B,N,N)"
        device = disto.device

        # 1D residue validity and effective length.
        residue_valid = mask.to(device).float() if mask is not None else torch.ones(B, N, device=device)
        Leff = effective_length_from_mask(mask) if mask is not None else torch.full((B,), float(N), device=device)

        # 2D pairwise mask.
        Mpair = (residue_valid.unsqueeze(1) * residue_valid.unsqueeze(2)).unsqueeze(1)  # (B,1,N,N)
        S = disto.unsqueeze(1).float()

        # Base masked S.
        S = S * Mpair

        # Geometric masks.
        W_diag_band = build_diag_band_mask(N, self.canonical_diag_ms, device, S.dtype, self.diag_bandwidth)
        W_upper = build_upper_triangle_mask(N, device, S.dtype, exclude_diag=True)

        feats_all = []
        attr_diag = torch.zeros(B, N, device=device) if return_attr else None
        attr_anti = torch.zeros(B, N, device=device) if return_attr else None
        details = None
        if return_details:
            details = {
                "residue_valid": residue_valid,
                "effective_length": Leff,
                "pair_mask": Mpair.squeeze(1),
                "diag": None,
                "anti": None,
            }

        # Autocast only applies on CUDA; on CPU it is disabled (runs in fp32) to avoid a
        # spurious "CUDA is not available" warning while keeping GPU numerics unchanged.
        use_amp = self.amp_dtype is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=self.amp_dtype, enabled=use_amp):
            # ===== Diagonal group =====
            if use_diag and len(self.arch_ids_diag) > 0:
                m = self.canonical_diag_ms
                s = self.diag_stride

                Sd = S * W_diag_band
                Md = Mpair * W_diag_band
                # Count valid cells per patch.
                ones_kernel = torch.ones((1, 1, m, m), device=Md.device, dtype=Md.dtype)
                valid_counts = F.conv2d(Md, ones_kernel, stride=s)  # (B,1,nH,nW)
                # Flatten to (B,W) and threshold to a boolean validity mask.
                mask_flat = (valid_counts.view(B, -1) > 0).float()

                # Require the WHOLE m×m window to lie inside valid (unpadded) residues.
                # This prevents diag attribution from bleeding into the padded tail.
                full_valid_counts = F.conv2d(Mpair, ones_kernel, stride=s)  # (B,1,nH,nW)
                full_valid_window = (full_valid_counts >= (m * m - 0.5)).float()  # (B,1,nH,nW)

                N = S.shape[-1]                     # S is (B, 1, N, N)
                nH = (N - m) // s + 1
                nW = (N - m) // s + 1

                starts = grid_starts_from_shape(nH, nW, s, device)
                Wtot = nH * nW

                # Build kernel bank for the diagonal group.
                diag_tpls = [self._tpl_diag(a, device, S.dtype) for a in self.arch_ids_diag]
                Kbank = build_kernel_bank(diag_tpls, device, S.dtype)  # (K_d,1,m,m)
                if Kbank is not None:
                    conv_raw = F.conv2d(Sd, Kbank, stride=s)           # (B, K_d, nH, nW)
                    scores = conv_raw                                  # broadcast over channel dim
                    if self.use_coverage_weighting:
                        coverage = valid_counts / float(m * m)
                        scores = scores * coverage

                    # ===== feasibility mask =====
                    # only keep top-left corners on the diagonal
                    feasible_diag = (starts[:, 0] == starts[:, 1]).float().view(1, 1, nH, nW)
                    scores = scores * feasible_diag

                    # Only keep windows fully within the unpadded residue range.
                    scores = scores * full_valid_window

                    scores_flat = scores.view(B, len(self.arch_ids_diag), -1)  # (B, K_d, W)
                    if return_details:
                        details["diag"] = {
                            "arch_ids": tuple(self.arch_ids_diag),
                            "starts": starts,
                            "scores": scores_flat,
                            "valid_mask": mask_flat,
                            "valid_counts": valid_counts.view(B, -1),
                            "feasible_mask": feasible_diag.view(-1),
                            "kernel_bank": Kbank.detach(),
                            "window_size": m,
                            "stride": s,
                            "grid_shape": (nH, nW),
                        }
                    for k, a in enumerate(self.arch_ids_diag):
                        feats = self._features_from_scores(scores_flat[:, k, :], mask_flat,
                                                           starts, m, Leff, N, self.diag_features)
                        feats_all.append(torch.stack(feats, dim=-1))
                        if return_attr:
                            topk = min(max(1, self.k_top), Wtot)
                            vals, idxs = torch.topk(scores_flat[:, k, :], k=topk, dim=1)
                            for b in range(B):
                                for j in range(topk):
                                    s_val = vals[b, j].float()
                                    r0, c0 = starts[idxs[b, j]].tolist()
                                    r1, c1 = min(r0 + m, N), min(c0 + m, N)
                                    attr_diag[b, r0:r1] += s_val / m
                                    attr_diag[b, c0:c1] += s_val / m
            elif len(self.arch_ids_diag) > 0:
                zero_feat = torch.zeros(B, len(self.diag_features), device=device, dtype=S.dtype)
                feats_all.extend(zero_feat for _ in self.arch_ids_diag)
                if return_details:
                    details["diag"] = {
                        "arch_ids": tuple(self.arch_ids_diag),
                        "disabled": True,
                        "window_size": self.canonical_diag_ms,
                        "stride": self.diag_stride,
                    }

            # ===== Off-diagonal (upper triangle) group =====
            if use_anti and len(self.arch_ids_anti) > 0:
                m = self.canonical_anti_ms
                s = self.anti_stride

                Sa = S * W_upper
                Ma = Mpair * W_upper
                # Count valid cells per patch.
                ones_kernel = torch.ones((1, 1, m, m), device=Ma.device, dtype=Ma.dtype)
                valid_counts = F.conv2d(Ma, ones_kernel, stride=s)
                # Flatten to (B,W) and threshold to a boolean validity mask.
                mask_flat = (valid_counts.view(B, -1) > 0).float()

                N = S.shape[-1]                     # S is (B, 1, N, N)
                nH = (N - m) // s + 1
                nW = (N - m) // s + 1

                starts = grid_starts_from_shape(nH, nW, s, device)
                Wtot = nH * nW

                anti_tpls = [self._tpl_anti(a, device, S.dtype) for a in self.arch_ids_anti]
                Kbank = build_kernel_bank(anti_tpls, device, S.dtype)  # (K_a,1,m,m)
                if Kbank is not None:
                    conv_raw = F.conv2d(Sa, Kbank, stride=s)           # (B, K_a, nH, nW)
                    scores = conv_raw
                    if self.use_coverage_weighting:
                        coverage = valid_counts / float(m * m)
                        scores = scores * coverage

                    # ===== feasibility mask =====
                    # valid only if the whole m×m window lies strictly above the diagonal: r0 + m ≤ c0
                    feasible = (starts[:, 0] + m <= starts[:, 1]).float().view(1, 1, nH, nW)
                    scores = scores * feasible

                    scores_flat = scores.view(B, len(self.arch_ids_anti), -1)  # (B, K_a, W)
                    if return_details:
                        details["anti"] = {
                            "arch_ids": tuple(self.arch_ids_anti),
                            "starts": starts,
                            "scores": scores_flat,
                            "valid_mask": mask_flat,
                            "valid_counts": valid_counts.view(B, -1),
                            "feasible_mask": feasible.view(-1),
                            "kernel_bank": Kbank.detach(),
                            "window_size": m,
                            "stride": s,
                            "grid_shape": (nH, nW),
                        }
                    for k, a in enumerate(self.arch_ids_anti):
                        feats = self._features_from_scores(scores_flat[:, k, :], mask_flat,
                                                           starts, m, Leff, N, self.anti_features)
                        feats_all.append(torch.stack(feats, dim=-1))
                        if return_attr:
                            topk = min(max(1, self.k_top), Wtot)
                            vals, idxs = torch.topk(scores_flat[:, k, :], k=topk, dim=1)
                            for b in range(B):
                                for j in range(topk):
                                    s_val = vals[b, j].float()
                                    r0, c0 = starts[idxs[b, j]].tolist()
                                    r1, c1 = min(r0 + m, N), min(c0 + m, N)
                                    attr_anti[b, r0:r1] += s_val / m
                                    attr_anti[b, c0:c1] += s_val / m
            elif len(self.arch_ids_anti) > 0:
                zero_feat = torch.zeros(B, len(self.anti_features), device=device, dtype=S.dtype)
                feats_all.extend(zero_feat for _ in self.arch_ids_anti)
                if return_details:
                    details["anti"] = {
                        "arch_ids": tuple(self.arch_ids_anti),
                        "disabled": True,
                        "window_size": self.canonical_anti_ms,
                        "stride": self.anti_stride,
                    }

        # Ensure attribution never highlights padded residues.
        if return_attr:
            attr_diag = attr_diag * residue_valid
            attr_anti = attr_anti * residue_valid

        feat_mat = torch.cat(feats_all, dim=1) if len(feats_all) > 0 else torch.zeros(B, 0, device=device)

        # MLP head.
        pooled = self.bn1(feat_mat)
        if self.bn1_clamp is not None:
            pooled = pooled.clamp(-self.bn1_clamp, self.bn1_clamp)
        hidden = self.fc1(pooled)
        hidden = self.relu(hidden)  # (batch_size, hidden_dim)
        logits = self.out(hidden)   # (batch_size, num_labels)

        if return_attr and return_details:
            return logits, attr_diag, attr_anti, details
        if return_attr:
            return logits, attr_diag, attr_anti
        if return_details:
            return logits, details
        return logits


# =============================
# Sequence analyzer (ESM model)
# =============================

class SequenceAnalyzer(nn.Module):
    """ESM embedding classifier: pool residue embeddings, then a small MLP head.

    Residue embeddings (produced upstream by ESM) are pooled by the configured
    ``Pooling`` strategy and passed through ``LayerNorm -> Linear -> ReLU -> Linear``
    to produce per-label logits. When ``pooling_method='attn_light'``, per-residue
    attention weights are exposed for interpretability via ``return_attr`` /
    ``return_details``.

    Args:
        num_labels: number of output labels (GO terms).
        hidden_dim: hidden width of the MLP head.
        pooling_method: one of the strategies supported by ``Pooling``.
        emb_size: ESM embedding dimension.
        exclude_special_tokens: exclude the <cls> token from pooling (except when
            ``pooling_method='cls'``).
        **pooling_kwargs: forwarded to ``Pooling`` (e.g. ``attn_hidden``, ``attn_temperature``).
    """

    def __init__(self, num_labels, hidden_dim=512, pooling_method='mean',
                 emb_size=1280, exclude_special_tokens: bool = True, **pooling_kwargs):
        super(SequenceAnalyzer, self).__init__()

        self.ARCHITECTURE = {"name": f"ESM with {pooling_method} pooling",
                             "hidden_dim": hidden_dim,
                             "num_labels": num_labels,
                             **pooling_kwargs}

        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.emb_size = emb_size
        self.pooling_method = pooling_method
        self.exclude_special_tokens = exclude_special_tokens
        pooling_dim_map = {
            "cls": self.emb_size,
            "mean": self.emb_size,
            "max": self.emb_size,
            "attn_light": self.emb_size,   # lightweight attention (same dim as mean)
        }
        pooling_dim = pooling_dim_map[self.pooling_method]

        self.pooler = Pooling(self.pooling_method, self.emb_size, **pooling_kwargs)
        self.ln1 = nn.LayerNorm(pooling_dim)
        self.fc1 = nn.Linear(pooling_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, inputs, disto, mask, return_attr: bool = False, return_details: bool = False):
        """
        Args:
            inputs: ``(B, L, D)`` token embeddings (<cls> followed by residues).
            disto:  unused; kept so all models share a common signature.
            mask:   ``(B, N)`` residue-space validity (1 = valid residue, 0 = padding).
            return_attr / return_details: return extra interpretability outputs.
        """
        # Dataloader mask is residue-space: 1 = valid residue, 0 = padded residue.
        token_mask = None
        if mask is not None:
            residue_mask = mask.bool()
            batch_size, token_len = inputs.shape[:2]
            residue_lengths = residue_mask.long().sum(dim=1)
            residue_lengths = residue_lengths.clamp(min=0, max=max(token_len - 1, 0))

            # Convert residue-space validity into token-space validity.
            # Embeddings contain <cls> followed by residues; <eof> is removed in the dataloader.
            mask = torch.zeros(batch_size, token_len, dtype=torch.bool, device=residue_mask.device)
            if token_len > 0:
                mask[:, 0] = True  # <cls>
                positions = torch.arange(token_len, device=residue_mask.device).unsqueeze(0)
                mask |= (positions >= 1) & (positions < (residue_lengths.unsqueeze(1) + 1))

            # Optionally exclude the <cls> token from pooling.
            if self.exclude_special_tokens and self.pooling_method != 'cls' and mask.size(1) >= 1:
                mask[:, 0] = False
            token_mask = mask

        # Embedding pooling. For `pooling_method='attn_light'`, per-residue weights
        # are stored in `self.pooler.last_attn_weights`.
        pooled = self.pooler(inputs, mask)

        # Collect intermediates for interpretability (no architecture change).
        # These are *views* into the same computation graph used for logits.
        attr = {
            "inputs": inputs,   # (B, L, D)
            "pooled": pooled,   # (B, D) pooled embedding before MLP
            "hidden": None,     # (B, H) pre-output representation: relu(fc1(ln(pooled)))
            "attn": None,       # (B, L) (only for attn_light)
            "attn_x": None,     # (B, L, D) = attn[:, :, None] * inputs (only for attn_light)
        }
        if self.pooling_method == 'attn_light':
            attn = self.pooler.last_attn_weights
            attr["attn"] = attn
            attr["attn_x"] = attn.unsqueeze(-1) * inputs

        # MLP head.
        pooled = self.ln1(pooled)
        hidden = self.fc1(pooled)
        hidden = self.relu(hidden)  # (batch_size, hidden_dim)
        if return_attr:
            attr["hidden"] = hidden
        logits = self.output_layer(hidden)  # (batch_size, num_labels)

        details = None
        if return_details:
            details = {
                "token_mask": token_mask,
                "attn_weights": attr["attn"],
                "attn_x": attr["attn_x"],
                "pooled": attr["pooled"],
                "hidden": hidden,
            }
        if return_attr and return_details:
            return logits, attr, details
        if return_attr:
            return logits, attr
        if return_details:
            return logits, details
        return logits


# ============
# Fusion model
# ============

class FusionModel(nn.Module):
    """Frozen ``StructuralProber`` + frozen ``SequenceAnalyzer`` + trainable gate.

    Predictions blend the two sub-models with a per-label gate driven only by the
    sequence model's internal representation::

        gate   = sigmoid(Linear(x_gate))                    # (B, num_labels)
        logits = gate * logits_seq + (1 - gate) * logits_struct

    The gate starts near 0 (via ``gate_init_bias``) so the model begins close to
    the sequence model and learns to mix in structure only where it helps.

    ``gate_input`` selects which sequence-model tensor drives the gate:
      - ``'hidden'``: ``relu(fc1(ln(pooled)))`` (stronger signal)
      - ``'pooled'``: the pooled embedding before the MLP

    Args:
        structure_model: a ``StructuralProber`` (frozen at fusion time).
        esm_model: a ``SequenceAnalyzer`` (frozen at fusion time).
        gate_input: ``'hidden'`` or ``'pooled'``.
        gate_init_bias: initial bias of the gate linear layer.
    """

    def __init__(
        self,
        structure_model: nn.Module,
        esm_model: nn.Module,
        gate_input: str = "hidden",     # 'hidden' or 'pooled'
        gate_init_bias: float = -5.0,
    ):
        super().__init__()
        # Attribute names (`kernel`, `esm`) are kept for checkpoint compatibility:
        # saved fusion state dicts use the `kernel.*` / `esm.*` key prefixes.
        self.kernel = structure_model
        self.esm = esm_model
        self.gate_input = str(gate_input)

        if self.gate_input not in {"hidden", "pooled"}:
            raise ValueError("gate_input must be 'hidden' or 'pooled'")

        self.num_labels = getattr(self.kernel, "num_labels", None)
        if self.num_labels is None:
            self.num_labels = int(self.esm.output_layer.out_features)

        # Gate input dimensionality (derived from the sequence model's MLP shapes).
        if self.gate_input == "hidden":
            gate_dim = int(self.esm.fc1.out_features)   # == esm.output_layer.in_features
        else:  # pooled
            gate_dim = int(self.esm.fc1.in_features)    # pooling dim

        self.refine_gate = nn.Linear(gate_dim, self.num_labels)
        nn.init.zeros_(self.refine_gate.weight)
        nn.init.constant_(self.refine_gate.bias, float(gate_init_bias))

        self.ARCHITECTURE = {
            "name": f"FrozenKernel + FrozenESM + gate({self.gate_input})",
            "gate_input": self.gate_input,
            "gate_init_bias": float(gate_init_bias),
            "kernel": getattr(self.kernel, "ARCHITECTURE", None),
            "esm": getattr(self.esm, "ARCHITECTURE", None),
        }

        self.freeze_submodels()

    def freeze_submodels(self):
        """Freeze both sub-models (no grad, eval mode)."""
        for p in self.kernel.parameters():
            p.requires_grad = False
        for p in self.esm.parameters():
            p.requires_grad = False
        self.kernel.eval()
        self.esm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep sub-models frozen and in eval mode regardless of the parent's mode.
        self.kernel.eval()
        self.esm.eval()
        return self

    def forward(
        self,
        inputs,
        disto,
        mask,
        return_attr: bool = False,
        use_diag: Optional[bool] = None,
        use_anti: Optional[bool] = None,
        return_details: bool = False,
        return_branches: bool = False,
    ):
        # Fast path for inference that needs the per-branch logits and the gate but NOT the
        # (expensive, discarded) per-residue attribution: run the kernel with return_attr=False.
        if return_branches and not return_attr and not return_details:
            with torch.no_grad():
                logits_struct = self.kernel(inputs, disto, mask, return_attr=False,
                                            use_diag=use_diag, use_anti=use_anti)
                logits_esm, attr_esm = self.esm(inputs, disto, mask, return_attr=True)
            x_gate = attr_esm.get(self.gate_input, None)
            if x_gate is None:
                raise ValueError(f"Fusion Model gate requires attr_esm['{self.gate_input}']")
            gate = torch.sigmoid(self.refine_gate(x_gate.detach()))
            logits = gate * logits_esm.detach() + (1 - gate) * logits_struct
            return logits, logits_struct, logits_esm, gate

        if return_details:
            k_out = self.kernel(
                inputs,
                disto,
                mask,
                return_attr=return_attr,
                use_diag=use_diag,
                use_anti=use_anti,
                return_details=True,
            )
            if return_attr:
                logits_struct, attr_diag, attr_anti, kernel_details = k_out
            else:
                logits_struct, kernel_details = k_out
                attr_diag, attr_anti = None, None

            logits_esm, attr_esm, esm_details = self.esm(
                inputs,
                disto,
                mask,
                return_attr=True,
                return_details=True,
            )
            x_gate = attr_esm.get(self.gate_input, None)
            if x_gate is None:
                raise ValueError(f"Fusion Model gate requires attr_esm['{self.gate_input}']")

            gate = torch.sigmoid(self.refine_gate(x_gate))
            logits = gate * logits_esm + (1 - gate) * logits_struct
            details = {
                "diag": None if kernel_details is None else kernel_details.get("diag"),
                "anti": None if kernel_details is None else kernel_details.get("anti"),
                "attn_weights": None if esm_details is None else esm_details.get("attn_weights"),
                "attn_x": None if esm_details is None else esm_details.get("attn_x"),
                "token_mask": None if esm_details is None else esm_details.get("token_mask"),
                "gate": gate,
                "logits_struct": logits_struct,
                "logits_esm": logits_esm,
                "kernel": kernel_details,
                "esm": esm_details,
            }
            if return_attr:
                return logits, logits_struct, logits_esm, attr_diag, attr_anti, details
            return logits, details

        with torch.no_grad():
            k_out = self.kernel(
                inputs,
                disto,
                mask,
                return_attr=return_attr,
                use_diag=use_diag,
                use_anti=use_anti,
            )
        if return_attr:
            logits_struct, attr_diag, attr_anti = k_out
        else:
            logits_struct, attr_diag, attr_anti = k_out, None, None

        with torch.no_grad():
            logits_esm, attr_esm = self.esm(inputs, disto, mask, return_attr=True)

        x_gate = attr_esm.get(self.gate_input, None)
        if x_gate is None:
            raise ValueError(f"Fusion Model gate requires attr_esm['{self.gate_input}']")

        gate = torch.sigmoid(self.refine_gate(x_gate.detach()))  # (B, num_labels)
        logits = gate * logits_esm.detach() + (1 - gate) * logits_struct

        if return_attr:
            return logits, logits_struct, logits_esm, gate, attr_diag, attr_anti, attr_esm
        return logits


# =============
# Model factory
# =============

def build_deepfri2_model(
    ontology,
    run_names,
    num_labels,
    device,
    params_dir="params",
    esm_dim=1280,
    m_diag=60,
    m_anti=60,
    num_diag=30,
    num_anti=30,
    diag_stride=4,
    anti_stride=4,
    diag_feats=("max", "mean", "topk_mean", "count", "argmax_r", "topk_std",
                "soft_x", "sigma_x", "period", "freq", "freq_amp"),
    anti_feats=("max", "mean", "topk_mean", "count", "argmax_r", "argmax_c", "topk_std",
                "soft_x", "soft_y", "sigma_x", "sigma_y", "rho"),
    peak_thresh=0.6,
    topk_k=5,
    hidden_dim=512,
    attn_hidden=64,
    attn_temperature=2.0,
    gate_input="hidden",
    gate_init_bias=-3.0,
    model_type="fusion",
):
    """Build a deepFRI2 model for one ontology and load its weights.

    ``model_type`` selects which model to build and which checkpoint(s) to load:
    - ``"fusion"`` (default): a frozen ``FusionModel`` wrapping a ``StructuralProber`` and a
      ``SequenceAnalyzer`` (loads the structure, sequence and fusion checkpoints).
    - ``"structure"``: just the ``StructuralProber`` (loads only the structure checkpoint).
    - ``"sequence"``: just the ``SequenceAnalyzer`` (loads only the sequence checkpoint).

    Returns the model in eval mode on ``device``. Architecture hyperparameters default to the
    values used to train the released deepFRI2 checkpoints; override them only if loading
    differently-trained weights.

    Args:
        ontology: ontology tag (e.g. ``"MF"``); locates weights under ``params/<ontology>/``.
        run_names: dict with keys ``"structure"``, ``"sequence"``, ``"fusion"`` -> run names.
        num_labels: number of GO labels for this ontology.
        device: torch device to place the model on.
        params_dir: directory holding ``<ontology>/<run_name>.pth`` checkpoints.
        esm_dim: ESM embedding dimension.
        m_diag / m_anti: diagonal / off-diagonal kernel sizes.
        num_diag / num_anti: number of diagonal / off-diagonal kernels.
        model_type: ``"fusion"`` (default), ``"structure"`` or ``"sequence"``.
        (remaining args): fixed architecture hyperparameters of the released checkpoints.
    """
    if model_type not in ("fusion", "structure", "sequence"):
        raise ValueError(f"model_type must be 'fusion', 'structure' or 'sequence', got {model_type!r}")

    def _build_structure():
        arch_to_size_diag = {f"diag_{i}": m_diag for i in range(1, num_diag + 1)}
        arch_to_size_anti = {f"anti_{i}": m_anti for i in range(1, num_anti + 1)}
        m = StructuralProber(
            num_labels=num_labels,
            arch_to_size_diag=arch_to_size_diag,
            arch_to_size_anti=arch_to_size_anti,
            canonical_diag_ms=m_diag,
            diag_stride=diag_stride,
            canonical_anti_ms=m_anti,
            anti_stride=anti_stride,
            diag_feats=diag_feats,
            anti_feats=anti_feats,
            peak_thresh=peak_thresh,
            topk_k=topk_k,
            frozen_kernels=False,
            amp_dtype=None,
            enforce_symmetry_diag=True,
            enforce_symmetry_anti=False,
            enforce_positivity_diag=False,
            enforce_positivity_anti=False,
        )
        return load_run_weights(m, ontology, run_names["structure"], strict=False, params_dir=params_dir)

    def _build_sequence():
        m = SequenceAnalyzer(
            num_labels=num_labels,
            hidden_dim=hidden_dim,
            pooling_method="attn_light",
            emb_size=esm_dim,
            attn_hidden=attn_hidden,
            attn_temperature=attn_temperature,
        )
        return load_run_weights(m, ontology, run_names["sequence"], strict=False, params_dir=params_dir)

    if model_type == "structure":
        return _build_structure().eval().to(device)
    if model_type == "sequence":
        return _build_sequence().eval().to(device)

    fusion_model = FusionModel(
        structure_model=_build_structure(),
        esm_model=_build_sequence(),
        gate_input=gate_input,
        gate_init_bias=gate_init_bias,
    )
    fusion_model = load_run_weights(fusion_model, ontology, run_names["fusion"], strict=False, params_dir=params_dir)
    return fusion_model.eval().to(device)
