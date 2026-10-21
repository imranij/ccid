"""a7_steelman2.py -- the size-controlled "is the model better at its BEST?" test.

The naive top-N comparison in a7_steelman.csv is confounded: the random pool has 11,744
members, so its top-20 is drawn from 178x more candidates than the CVAE's.  The correct
test asks: if you had drawn only 66 random candidates (the same budget the CVAE used),
how often would their top-N beat the CVAE's top-N?

Procedure: 2,000 random subsamples of size n_cvae from the random pool; for each, the
median (and max) of the top-N by the composite predicted-property score; the CVAE's value
is then located in that null distribution as a one-sided empirical percentile / p-value.
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import json
import numpy as np, pandas as pd

RESULTS = _CCID_ROOT + "/results"
NBOOT = 2000
SEED = 20260913
SCORE_COLS = [("predicted_property_rank_score", "higher"),
              ("pred_band_gap_eV_mean", "higher"),
              ("pred_poly_total_mean", "higher"),
              ("pred_stability_wolverton_oxides_mean", "lower")]


def main():
    d = pd.read_csv(f"{RESULTS}/a7_merged_candidates.csv")
    cv = d[d["population"] == "cvae"]
    rows = []
    for pool_name in ("random", "random_freq"):
        pool = d[d["population"] == pool_name]
        rng = np.random.default_rng(SEED)
        n = len(cv)
        idx = rng.integers(0, len(pool), size=(NBOOT, n))
        for col, better in SCORE_COLS:
            pv = pool[col].to_numpy()
            cvv = cv[col].to_numpy()
            sgn = 1.0 if better == "higher" else -1.0
            for N in (1, 5, 10, 20):
                cv_top = np.sort(sgn * cvv)[::-1][:N]
                cv_stat = float(np.median(cv_top))
                samp = sgn * pv[idx]                      # (NBOOT, n)
                part = np.sort(samp, axis=1)[:, ::-1][:, :N]
                null = np.median(part, axis=1)
                # one-sided: fraction of equal-budget random draws whose top-N is at
                # least as good as the CVAE's top-N
                p = float((null >= cv_stat).mean())
                rows.append(dict(
                    pool=pool_name, axis=col, better=better, top_N=N,
                    cvae_topN_median=sgn * cv_stat,
                    null_median_of_topN_median=sgn * float(np.median(null)),
                    null_ci95=[sgn * float(np.percentile(null, 97.5)),
                               sgn * float(np.percentile(null, 2.5))][::1 if better == "higher" else -1],
                    p_random_draw_at_least_as_good=p,
                    n_budget=n, n_boot=NBOOT))
    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/a7_steelman_size_controlled.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
