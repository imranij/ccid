"""Precompute and cache Magpie composition features for the CVAE training-records
table (25,148 rows). Composition-only: only `formula` is used, no structure, no
conditioning vector. Cached to speed up the two downstream training scripts.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import time
import numpy as np
import pandas as pd

from a5_common import magpie_features, DGX_OUT, MAGPIE_LABELS

def main():
    df = pd.read_csv(DGX_OUT / "composition_cvae_training_records.csv")
    print(f"Loaded {len(df)} training records")
    t0 = time.time()
    X, ok = magpie_features(df["formula"].tolist())
    print(f"Featurized in {time.time()-t0:.1f}s; {ok.sum()}/{len(df)} parsed OK")
    if not ok.all():
        bad = df.loc[~ok, "formula"].tolist()
        print(f"WARNING: {len(bad)} formulas failed to featurize, e.g. {bad[:10]}")
    np.savez_compressed(
        _CCID_ROOT + "/analysis/a5_surrogate/a5_training_magpie_features.npz",
        X=X, ok=ok, labels=np.array(MAGPIE_LABELS, dtype=object)
    )
    print("Saved a5_training_magpie_features.npz")

if __name__ == "__main__":
    main()
