"""Shared utilities for the composition-only property surrogates (Agent 5).

Composition-only Magpie-style featurization (via matminer's ElementProperty "magpie"
preset over pymatgen Composition objects), plus a small bootstrap-ensemble regressor
used for uncertainty estimates throughout this deliverable.

Kept deliberately dependency-light and importable from every a5_*.py script.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import json
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(_CCID_ROOT + "")
DGX_CODE = PROJECT_ROOT / "dgx" / "code"
DGX_OUT = PROJECT_ROOT / "dgx" / "outputs_dgx"
RESULTS_DIR = Path(_CCID_ROOT + "/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import sys

if str(DGX_CODE) not in sys.path:
    sys.path.insert(0, str(DGX_CODE))

from composition_cvae_common import (  # noqa: E402
    parse_formula,
    canonical_formula,
    split_indices,
    OXIDATION_STATES,
    COMMON_CERAMIC_ELEMENTS,
    HIGH_RISK_ELEMENTS,
)

from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

_MAGPIE = ElementProperty.from_preset("magpie")
MAGPIE_LABELS = _MAGPIE.feature_labels()


def formula_to_composition(formula: str) -> Composition | None:
    """Robust formula -> pymatgen Composition, using the project's own parser
    (handles the fractional/decimal formulas that appear in the training table)
    so we do not depend on pymatgen's own string parser succeeding on every string.
    """
    counts = parse_formula(formula)
    if not counts:
        return None
    try:
        return Composition(counts)
    except Exception:
        return None


def magpie_features(formulas: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features [n, 132], ok_mask [n]) — ok_mask False rows are all-zero
    and should be dropped by the caller (formula could not be parsed/featurized)."""
    feats = np.zeros((len(formulas := list(formulas)), len(MAGPIE_LABELS)), dtype=np.float64)
    ok = np.zeros(len(formulas), dtype=bool)
    for i, f in enumerate(formulas):
        comp = formula_to_composition(f)
        if comp is None or len(comp) == 0:
            continue
        try:
            feats[i] = _MAGPIE.featurize(comp)
            ok[i] = True
        except Exception:
            continue
    return feats, ok


def rf_factory_default(seed, n_estimators=150, max_depth=None, min_samples_leaf=2):
    """Default RandomForest factory, defined at module scope (not inside a
    training script's __main__) so BootstrapEnsemble instances unpickle cleanly
    in any other script that imports a5_common."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        n_jobs=-1, random_state=seed,
    )


class BootstrapEnsemble:
    """A bagging ensemble of a base sklearn-style regressor, giving a mean
    prediction and an ensemble-spread uncertainty (std across members).
    This is NOT a calibrated predictive interval; it reflects sensitivity of the
    fit to which training rows were resampled (model/estimation uncertainty),
    not the irreducible noise of the property itself. Documented as such in
    a5_REPORT.md.
    """

    def __init__(self, base_factory, n_members: int = 15, random_state: int = 0):
        self.base_factory = base_factory
        self.n_members = n_members
        self.random_state = random_state
        self.models_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.models_ = []
        for m in range(self.n_members):
            idx = rng.integers(0, n, size=n)  # bootstrap resample
            model = self.base_factory(self.random_state + m)
            model.fit(X[idx], y[idx])
            self.models_.append(model)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = np.stack([m.predict(X) for m in self.models_], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else None
    return {"n": int(len(y_true)), "mae": float(mae), "rmse": float(rmse), "r2": (float(r2) if r2 is not None else None)}


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
