"""
QuantAlphaFusionNet: Quantitative Fusion Core for AlphaFusionNet
=======================================================

Overview
--------
This module contains the QuantAlphaFusionNet class, which is the core quantitative engine 
for fusing portfolio recommendations from NeuralFusionCore and NetWeaver. 
It produces final, risk-aware, normalized portfolio weights while preserving 
long-short exposure. This module is purely numerical and independent of any LLM logic.

Key Features
------------
1. Fusion of two intelligence sources:
   - NeuralFusionCore: provides risk-adjusted portfolio weights (can be negative for shorts).
   - NetWeaver: provides predicted return ratios (pseudo-weights) for top-k assets.

2. Weight conversion for NetWeaver predictions:
   - Methods supported: 'rank', 'softmax', 'proportional'.
   - Handles optional top-k filtering.
   - Preserves the sign of predictions (long/short).

3. Weighted fusion:
   - Formula: w_final = alpha * w_neural + (1 - alpha) * w_net
   - Applies optional ticker overrides and sector multipliers.
   - Clips final weights to configurable min/max limits.
   - Renormalizes to ensure total gross exposure equals specified gross.

4. Supports long-short portfolios and risk management constraints.

Mathematical Formulation
------------------------
Let:
    w_nf ∈ ℝᴺ — NeuralFusionCore portfolio weights (signed, sum absolute = 1)
    s_net ∈ ℝᴺ — NetWeaver predicted returns (may be top-k subset)
    α ∈ [0,1] — fusion coefficient

1. Convert NetWeaver predictions to pseudo-weights:
       w_net = normalize(s_net, method='rank'|'softmax'|'proportional')

2. Fuse weights (signed):
       w_final = α * w_nf + (1 - α) * w_net

3. Apply optional overrides:
       w_final[ticker] = override_value (if specified)

4. Apply sector multipliers:
       w_final[ticker] *= multiplier (if specified)

5. Clip weights:
       w_final = clip(w_final, w_min, w_max)

6. Renormalize gross exposure:
       w_final = (gross / Σ|w_final|) * w_final

7. Optional post-clipping renormalization to maintain gross exposure.

Inputs
------
- w_neural : dict[str, float] — NeuralFusionCore weights per asset
- s_net : dict[str, float] — NetWeaver predicted returns per asset
- alpha : float — fusion coefficient between NeuralFusionCore and NetWeaver
- gross : float — desired total gross exposure (Σ|w_final|)
- w_min / w_max : float — minimum/maximum allowed individual weights
- overrides : dict[str, float] — optional ticker-specific weight overrides
- sector_multipliers : dict[str, float] — optional sector/ticker multipliers

Outputs
-------
- w_net_converted : dict[str, float] — normalized NetWeaver weights
- w_final : pd.Series — final fused, clipped, and normalized portfolio weights

Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 22
Version: 1.1.0 
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

class QuantAlphaFusionNet:
    """
    Quantitative fusion core for AlphaFusionNet.
    Handles fusion of NeuralFusionCore and NetWeaver outputs.
    """
    def __init__(self, alpha: float = 0.7, gross: float = 1.0, w_min: float = -0.3, w_max: float = 0.3):
        self.alpha = float(alpha)
        self.gross = float(gross)
        self.w_min = float(w_min)
        self.w_max = float(w_max)

    def convert_netweaver_to_weights(
        self,
        s_net: Dict[str, float],
        gross_net: float,
        method: str = "rank",
        topk: Optional[int] = None,
    ) -> Dict[str, float]:
        if not s_net:
            return {}

        tickers = list(s_net.keys())
        scores = np.array([s_net[t] for t in tickers], dtype=float)
        signs = np.sign(scores)
        abs_scores = np.abs(scores)

        if topk is not None and topk > 0 and topk < len(tickers):
            order = np.argsort(-abs_scores)
            keep_idx = order[:topk]
            mask = np.zeros_like(abs_scores, dtype=bool)
            mask[keep_idx] = True
        else:
            mask = np.ones_like(abs_scores, dtype=bool)

        abs_scores_masked = abs_scores * mask
        if abs_scores_masked.sum() == 0:
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                return {}
            weights_raw = np.zeros_like(abs_scores)
            weights_raw[idxs] = 1.0
        else:
            if method == "rank":
                ranks = np.zeros_like(abs_scores)
                masked_idxs = np.where(mask)[0]
                desc = np.argsort(-abs_scores_masked[masked_idxs])
                m = len(masked_idxs)
                for pos, idx_in_mask in enumerate(desc):
                    global_idx = masked_idxs[idx_in_mask]
                    ranks[global_idx] = m - pos
                weights_raw = ranks
            elif method == "softmax":
                masked_vals = abs_scores_masked.copy()
                exps = np.exp(masked_vals - masked_vals.max())
                weights_raw = exps
            else:  # proportional
                weights_raw = abs_scores_masked + 1e-12

        total = weights_raw.sum()
        if total == 0:
            return {t: 0.0 for t in tickers}

        signed_weights = signs * (weights_raw / total) * float(gross_net)
        return dict(zip(tickers, signed_weights))

    def fuse_signed(
        self,
        w_neural: Dict[str, float],
        w_net: Dict[str, float],
        overrides: Optional[Dict[str, float]] = None,
        sector_multipliers: Optional[Dict[str, float]] = None,
        renormalize_after_clip: bool = True,
    ) -> pd.Series:
        """
        Fuse NeuralFusionCore (risk-adjusted) and NetWeaver (alpha prediction) signals
        into final signed portfolio weights.

        Combines two model outputs using the fusion coefficient α:
            w_final = α * w_neural + (1 - α) * w_net
        Applies optional sector multipliers and overrides, clips to [w_min, w_max],
        and normalizes to ensure |weights| sum to gross exposure.

        Args:
            w_neural: Dict[str, float] - NeuralFusionCore risk-adjusted weights.
            w_net: Dict[str, float] - NetWeaver return prediction weights.
            overrides: Optional dict for manual ticker overrides.
            sector_multipliers: Optional dict of multiplicative factors per ticker.
            renormalize_after_clip: Whether to re-normalize after clipping.

        Returns:
            pd.Series: final fused weights (sum(|weights|) ≈ gross, within limits).
        """
        overrides = overrides or {}
        sector_multipliers = sector_multipliers or {}

        tickers = sorted(set(w_neural) | set(w_net) | set(overrides))
        if not tickers:
            return pd.Series(dtype=float)

        w_nf = np.array([w_neural.get(t, 0.0) for t in tickers], dtype=float)
        w_nw = np.array([w_net.get(t, 0.0) for t in tickers], dtype=float)

        combined = self.alpha * w_nf + (1.0 - self.alpha) * w_nw

        if sector_multipliers:
            mults = np.array([float(sector_multipliers.get(t, 1.0)) for t in tickers])
            combined *= mults

        for i, t in enumerate(tickers):
            if t in overrides:
                combined[i] = float(overrides[t])

        # Normalize to target gross exposure
        denom = np.sum(np.abs(combined)) + 1e-12
        combined = (self.gross / denom) * combined

        # Optional re-normalization (but always clip at the end)
        if renormalize_after_clip:
            combined = np.clip(combined, self.w_min, self.w_max)
            denom2 = np.sum(np.abs(combined)) + 1e-12
            combined = (self.gross / denom2) * combined

        # Final safety clip (guarantee limits)
        combined = np.clip(combined, self.w_min, self.w_max)

        return pd.Series(combined, index=tickers, name="weights")

