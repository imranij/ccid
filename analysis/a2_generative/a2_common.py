"""Shared harness for the a2 generative-model control experiments (MTCOMM-D-26-04679).

Every filter used here is imported from the project's own modules
(`composition_cvae_common.py`, `generate_composition_candidates.py`) so that the
baselines pass through a byte-identical decode / charge-balance / whitelist path
to the one that produced the manuscript's numbers.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(_CCID_ROOT + "/dgx")
WORKSPACE_ROOT = PROJECT_ROOT.parent
CODE_DIR = PROJECT_ROOT / "code"
RESULTS_DIR = Path(_CCID_ROOT + "/results")
sys.path.insert(0, str(CODE_DIR))

from composition_cvae_common import (  # noqa: E402
    COMMON_CERAMIC_ELEMENTS,
    HIGH_RISK_ELEMENTS,
    NOBLE_GASES,
    OXIDATION_STATES,
    decode_composition_vector,
    load_material_table,
    parse_formula,
)
from generate_composition_candidates import (  # noqa: E402
    composition_vector,
    load_model,
    mean_pairwise_distance,
)

_CACHE: dict = {}


def safe_mean_pairwise_distance(vectors: np.ndarray, max_pairs: int = 5000) -> float:
    """Memory-safe drop-in for generate_composition_candidates.mean_pairwise_distance.

    The project's version materialises every pair before subsampling, which OOMs once
    a baseline arm yields thousands of unique formulas. For n*(n-1)/2 <= max_pairs this
    returns exactly the project's value (all pairs, same order-independent mean);
    above that it draws max_pairs random pairs directly, an unbiased estimate of the
    same quantity.
    """
    n = len(vectors)
    if n < 2:
        return 0.0
    n_pairs = n * (n - 1) // 2
    if n_pairs <= max_pairs:
        return mean_pairwise_distance(vectors, max_pairs=max_pairs)
    rng = np.random.default_rng(5)
    i = rng.integers(0, n, size=max_pairs * 2)
    j = rng.integers(0, n, size=max_pairs * 2)
    keep = i != j
    i, j = i[keep][:max_pairs], j[keep][:max_pairs]
    return float(np.mean(np.linalg.norm(vectors[i] - vectors[j], axis=1)))


def get_model():
    if "model" not in _CACHE:
        _CACHE["model"], _CACHE["metadata"] = load_model(PROJECT_ROOT, "composition_cvae")
    return _CACHE["model"], _CACHE["metadata"]


def get_known():
    """Known-material table used by the paper for the novelty test."""
    if "known_table" not in _CACHE:
        _, meta = get_model()
        table = load_material_table(
            WORKSPACE_ROOT, include_double=bool(meta.get("include_double_perovskites", True))
        )
        _CACHE["known_table"] = table
        _CACHE["known_formulas"] = set(table["canonical_formula"])
    return _CACHE["known_table"], _CACHE["known_formulas"]


def element_vocab() -> list[str]:
    return list(get_model()[1]["element_vocab"])


def decode_batch(decoded: np.ndarray, vocab: list[str], top_cations: int = 3) -> pd.DataFrame:
    """Run the project's decode_composition_vector over a batch of decoder outputs."""
    rows = []
    for idx in range(len(decoded)):
        r = decode_composition_vector(decoded[idx], vocab, top_cations=top_cations)
        rows.append(
            {
                "sample_id": idx,
                "formula": r.get("formula", ""),
                "canonical_formula": r.get("canonical_formula", ""),
                "status": r.get("reason", ""),
                "valid": bool(r.get("valid", False)),
                "charge_balanced": bool(r.get("charge_balanced", False)),
                "elements": r.get("elements", ""),
                "high_risk_elements": r.get("high_risk_elements", ""),
                "unusual_elements": r.get("unusual_elements", ""),
            }
        )
    return pd.DataFrame(rows)


def score_frame(cand: pd.DataFrame, vocab: list[str]) -> dict:
    """Validity / uniqueness / novelty / diversity, defined exactly as in
    generate_composition_candidates.main()."""
    _, known_formulas = get_known()
    n = len(cand)
    cand = cand.copy()
    cand["novel"] = [
        bool(c) and c not in known_formulas for c in cand["canonical_formula"].astype(str)
    ]
    valid = cand[cand["valid"]]
    unique_valid = valid.drop_duplicates("canonical_formula")
    novel_unique = unique_valid[unique_valid["novel"]]
    vectors = (
        np.vstack([composition_vector(f, vocab) for f in novel_unique["formula"]])
        if len(novel_unique)
        else np.zeros((0, len(vocab)))
    )
    return {
        "total_samples": int(n),
        "valid_count": int(len(valid)),
        "unique_valid_count": int(len(unique_valid)),
        "novel_unique_count": int(len(novel_unique)),
        "validity_rate": float(len(valid) / max(1, n)),
        "uniqueness_rate": float(len(unique_valid) / max(1, len(valid))),
        "novelty_rate": float(len(novel_unique) / max(1, len(unique_valid))),
        "mean_pairwise_composition_distance": float(safe_mean_pairwise_distance(vectors)),
        "charge_balanced_rate": float(cand["charge_balanced"].mean()) if n else 0.0,
        "n_pairs_exact": bool(len(vectors)*(len(vectors)-1)//2 <= 5000),
        # diversity over ALL decoded samples, not just the ones that pass the filters
        "unique_formula_count_all": int(cand["canonical_formula"].astype(str).nunique()),
        "modal_formula_share_all": float(
            cand["canonical_formula"].astype(str).value_counts(normalize=True).iloc[0]
        ) if n else 0.0,
        "unique_valid_per_1000_samples": float(1000.0 * len(unique_valid) / max(1, n)),
        "unique_valid_per_1000_valid": float(
            1000.0 * len(unique_valid) / max(1, len(valid))
        ),
    }


def cvae_decode(n: int, seed: int, target_stability=0.9, target_gap=0.8, jitter=0.08,
                latent_sigma=1.0, conditions=None, z=None):
    """Replicates the archived generation path. Returns (decoded_comp, conditions)."""
    model, meta = get_model()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if z is None:
        z = torch.randn(n, int(meta["latent_dim"])) * float(latent_sigma)
    if conditions is None:
        conditions = np.column_stack(
            [
                np.clip(rng.normal(target_stability, jitter, n), 0.0, 1.0),
                np.clip(rng.normal(target_gap, jitter, n), 0.0, 1.0),
            ]
        ).astype(np.float32)
    with torch.no_grad():
        decoded = model.decode(z, torch.tensor(conditions, dtype=torch.float32)).numpy()
    return decoded[:, : int(meta["composition_dim"])], np.asarray(conditions), z
