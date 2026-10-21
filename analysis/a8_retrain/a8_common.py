"""Shared harness for a8 (loss fix + retrain) — MTCOMM-D-26-04679.

Reuses the a2 evaluation harness verbatim so that every number computed for a
retrained checkpoint passes through the byte-identical decode / charge-balance /
whitelist / novelty path used for the deposited baselines.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys
from pathlib import Path

A2_DIR = Path(_CCID_ROOT + "/analysis/a2_generative")
PROJECT_ROOT = Path(_CCID_ROOT + "/dgx")
CODE_DIR = PROJECT_ROOT / "code"
RESULTS_DIR = Path(_CCID_ROOT + "/results")
WORK_DIR = Path(_CCID_ROOT + "/analysis/a8_retrain")
CKPT_DIR = WORK_DIR / "checkpoints"
sys.path.insert(0, str(A2_DIR))
sys.path.insert(0, str(CODE_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import a2_common as C  # noqa: E402
from composition_cvae_common import CompositionCVAE, featurize, load_material_table, split_indices  # noqa: E402


def load_ckpt(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]
    model = CompositionCVAE(
        input_dim=int(meta["input_dim"]),
        cond_dim=int(meta["condition_dim"]),
        latent_dim=int(meta["latent_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
    )
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, meta, ckpt


def use_checkpoint(path: Path):
    """Point the a2 harness (C) at an arbitrary checkpoint file."""
    model, meta, ckpt = load_ckpt(Path(path))
    C._CACHE["model"] = model
    C._CACHE["metadata"] = meta
    # known_table / known_formulas are checkpoint-independent (same data, same
    # include_double flag) so the cache is deliberately kept across checkpoints.
    return model, meta, ckpt


def use_deposited():
    return use_checkpoint(PROJECT_ROOT / "outputs_dgx" / "composition_cvae_checkpoint.pt")


def latent_diagnostics(model, meta) -> dict:
    """Per-dimension KL of the aggregate posterior over the FULL material table."""
    table = load_material_table(PROJECT_ROOT.parent,
                                include_double=bool(meta.get("include_double_perovskites", True)))
    vocab = list(meta["element_vocab"])
    feats, conds = featurize(table, vocab)
    x = torch.tensor(feats, dtype=torch.float32)
    c = torch.tensor(conds, dtype=torch.float32)
    with torch.no_grad():
        mu, logvar = model.encode(x, c)
    kl_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0).numpy()
    return {
        "n_records_encoded": int(len(x)),
        "latent_dim": int(meta["latent_dim"]),
        "kl_per_dim_nats_sorted": [float(v) for v in np.sort(kl_dim)[::-1]],
        "total_kl_nats": float(kl_dim.sum()),
        "active_units_kl_gt_0.01": int((kl_dim > 0.01).sum()),
        "active_units_kl_gt_0.05": int((kl_dim > 0.05).sum()),
        "active_units_kl_gt_0.1": int((kl_dim > 0.1).sum()),
        "max_kl_dim_nats": float(kl_dim.max()),
        "var_of_posterior_mean_per_dim_sorted": [float(v) for v in np.sort(mu.var(dim=0).numpy())[::-1]],
        "active_units_au_var_gt_0.01": int((mu.var(dim=0).numpy() > 0.01).sum()),
    }
