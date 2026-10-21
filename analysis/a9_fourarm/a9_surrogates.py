"""a9_surrogates.py -- a5 surrogates applied to every arm's pooled unique valid set.

Prediction is a7_surrogates.predict (identical model objects, featurization and protocol:
band gap averaged over the three source one-hots, dielectric with is_phonon_aux = 0,
both per-source stability heads), called once on the concatenation of the four arms.
The a5 .pkl files were retrained with the unmodified a5 scripts (a9_retrain_a5.sh).
The composite score is a7's 0.4*stability + 0.3*gap + 0.3*permittivity, normalised on the
POOLED four-arm population (the same lines as a7_surrogates.main, pooled over four arms
instead of two).
Output: results/a9_surrogate_predictions.csv
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys
sys.path.insert(0, _CCID_ROOT + "/analysis/a9_fourarm")
import a9_common as K  # noqa: E402

import time
import numpy as np
import pandas as pd

import a7_surrogates as S


def main():
    t0 = time.time()
    pops = [pd.read_csv(K.RESULTS / f"a9_population_{a}.csv") for a in ("A", "B", "C", "D")]
    p = pd.concat(pops, ignore_index=True)
    # in_training_set lives in a9_chemical_sanity.csv; dropped here so that the a7_compare
    # merge (which expects it only on the chemistry side) does not create _x/_y columns.
    p = p.drop(columns=["in_training_set"])
    print(f"predicting {len(p)} formulas", flush=True)
    pr = S.predict(p["formula"].tolist())
    allf = pd.concat([p.reset_index(drop=True), pr], axis=1)
    allf["flag_low_bandgap_not_insulating"] = (
        allf["pred_band_gap_eV_mean"] < S.BANDGAP_INSULATOR_THRESHOLD)
    # --- a7_surrogates.main composite, verbatim, pooled over the four arms ---
    sc = "pred_stability_wolverton_oxides_mean"
    stab_norm = 1.0 - (allf[sc] - allf[sc].min()) / (allf[sc].max() - allf[sc].min() + 1e-12)
    gap_norm = np.clip(allf["pred_band_gap_eV_mean"] / 4.0, 0, 1)
    lg = np.log1p(allf["pred_poly_total_mean"])
    diel_norm = (lg - lg.min()) / (lg.max() - lg.min() + 1e-12)
    allf["predicted_property_rank_score"] = 0.4 * stab_norm + 0.3 * gap_norm + 0.3 * diel_norm
    allf.to_csv(K.RESULTS / "a9_surrogate_predictions.csv", index=False)
    for n, g in allf.groupby("population"):
        print(f"{n} n={len(g):6d} ok={g['featurized_ok'].mean():.4f} "
              f"Eg {g['pred_band_gap_eV_mean'].median():.3f} "
              f"eps0 {g['pred_poly_total_mean'].median():.2f} Ehull {g[sc].median():.4f} "
              f"score {g['predicted_property_rank_score'].median():.4f}")
    print(f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
