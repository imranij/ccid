"""a9_compare.py -- chemistry and surrogate comparisons across the four arms.

Reuse, unchanged:
  * a7_compare.load   -- merges surrogate predictions with chemistry annotations and
                         defines has_stage1_good (Hf/Zr/Al), all_stage1_good, stoich_family
                         and the wide-gap gate (pred. E_g >= 1.0 eV AND no open-shell cation
                         AND no forced Ti3+/Sn2+).  It reads two fixed file names, so it is
                         pointed at a scratch directory holding links named as a7 expects.
  * a7_compare.run_block -- every continuous axis (Cliff's delta, 2,000-resample bootstrap
                         CI, Mann-Whitney p) and every proportion (Wilson/Newcombe CI,
                         Fisher p), group a minus group b.
  * a7_stats.compare_proportions -- element enrichment.
  * a7_steelman2 logic  -- size-controlled top-N (2,000 equal-budget subsamples of the
                         arm-A pool, drawn with replacement exactly as a7 did), budget = the
                         size of the compared arm's pooled unique valid set.
Comparisons: arm D against each of A, B, C (the brief), plus the ladder steps B vs A,
C vs A and C vs B.
Outputs: a9_merged_candidates.csv, a9_compare_continuous.csv, a9_compare_proportions.csv,
a9_element_enrichment.csv, a9_steelman_size_controlled.csv, a9_compare_summary.json
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys
sys.path.insert(0, _CCID_ROOT + "/analysis/a9_fourarm")
import a9_common as K  # noqa: E402

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import a7_compare as AC
from a7_stats import compare_proportions
import a2_common as C

ARMS = ("A", "B", "C", "D")
PAIRS = [("D", "A"), ("D", "B"), ("D", "C"), ("B", "A"), ("C", "A"), ("C", "B")]
NBOOT = 2000
SEED = K.SAMPLE_SEED
SCORE_COLS = [("predicted_property_rank_score", "higher"),
              ("pred_band_gap_eV_mean", "higher"),
              ("pred_poly_total_mean", "higher"),
              ("pred_stability_wolverton_oxides_mean", "lower")]


def load_merged() -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a7_surrogate_predictions.csv").symlink_to(
            K.RESULTS / "a9_surrogate_predictions.csv")
        (Path(tmp) / "a7_chemical_sanity.csv").symlink_to(K.RESULTS / "a9_chemical_sanity.csv")
        old = AC.RESULTS
        AC.RESULTS = tmp
        try:
            d = AC.load()
        finally:
            AC.RESULTS = old
    return d


def steelman(d: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """a7_steelman2.main, with the CVAE replaced by each arm in turn and the pool = arm A."""
    pool = d[d["population"] == "A" + suffix]
    rows = []
    for arm in ("B" + suffix, "C" + suffix, "D" + suffix):
        cv = d[d["population"] == arm]
        n = len(cv)
        for col, better in SCORE_COLS:
            pv = pool[col].to_numpy()
            cvv = cv[col].to_numpy()
            sgn = 1.0 if better == "higher" else -1.0
            rng = np.random.default_rng(SEED)
            nulls = {N: [] for N in (1, 5, 10, 20)}
            for start in range(0, NBOOT, 100):        # chunked only to bound memory
                idx = rng.integers(0, len(pool), size=(min(100, NBOOT - start), n))
                samp = np.sort(sgn * pv[idx], axis=1)[:, ::-1]
                for N in nulls:
                    nulls[N].append(np.median(samp[:, :N], axis=1))
            for N in (1, 5, 10, 20):
                cv_top = np.sort(sgn * cvv)[::-1][:N]
                cv_stat = float(np.median(cv_top))
                null = np.concatenate(nulls[N])
                p = float((null >= cv_stat).mean())
                rows.append(dict(
                    arm=arm, pool="A" + suffix, axis=col, better=better, top_N=N,
                    arm_topN_median=sgn * cv_stat,
                    null_median_of_topN_median=sgn * float(np.median(null)),
                    null_ci95=[sgn * float(np.percentile(null, 97.5)),
                               sgn * float(np.percentile(null, 2.5))][::1 if better == "higher" else -1],
                    p_armA_draw_at_least_as_good=p, n_budget=n, n_boot=NBOOT))
    return pd.DataFrame(rows)


def enrichment(d: pd.DataFrame) -> pd.DataFrame:
    els = sorted(C.COMMON_CERAMIC_ELEMENTS - {"O"})
    rows = []
    for el in els:
        has = d["cation_set"].map(lambda s, e=el: e in s)
        frac = {a: (int(has[d["population"] == a].sum()), int((d["population"] == a).sum()))
                for a in ARMS}
        for other in ("A", "B", "C"):
            r = compare_proportions(frac["D"][0], frac["D"][1], frac[other][0], frac[other][1],
                                    "D", other)
            r.update(element=el, comparison=f"D vs {other}")
            for a in ARMS:
                r[f"frac_{a}"] = frac[a][0] / frac[a][1]
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    d = load_merged()
    assert d["has_open_shell"].notna().all(), "unassigned oxidation states present"
    d.to_csv(K.RESULTS / "a9_merged_candidates.csv", index=False)
    rows_c, rows_p = [], []
    for a, b in PAIRS:
        AC.run_block(d, a, b, f"{a} vs {b}", rows_c, rows_p)
        print(f"{a} vs {b} done ({time.time()-t0:.0f}s)", flush=True)
    # Sensitivity: novel candidates only (exact formula absent from the training table).
    # The surrogates were fitted on that table, so their predictions for training compounds
    # are in-sample; arm C's unique set is 52 % training compounds.
    dn = d[~d["in_training_set"].astype(bool)].copy()
    dn["population"] = dn["population"] + "_novel"
    for a, b in PAIRS[:3]:
        AC.run_block(dn, a + "_novel", b + "_novel", f"{a} vs {b} (novel only)", rows_c, rows_p)
    print(f"novel-only blocks done ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows_c).to_csv(K.RESULTS / "a9_compare_continuous.csv", index=False)
    pd.DataFrame(rows_p).to_csv(K.RESULTS / "a9_compare_proportions.csv", index=False)
    enrichment(d).to_csv(K.RESULTS / "a9_element_enrichment.csv", index=False)
    st = pd.concat([steelman(d), steelman(dn, "_novel")], ignore_index=True)
    st.to_csv(K.RESULTS / "a9_steelman_size_controlled.csv", index=False)
    print(f"steelman done ({time.time()-t0:.0f}s)", flush=True)

    summ = {}
    for n, g in pd.concat([d, dn]).groupby("population"):
        summ[n] = dict(
            n=len(g),
            median_band_gap=float(g["pred_band_gap_eV_mean"].median()),
            median_eps0=float(g["pred_poly_total_mean"].median()),
            median_ehull_like=float(g["pred_stability_wolverton_oxides_mean"].median()),
            median_eform_like=float(g["pred_stability_castelli_perovskites_mean"].median()),
            median_composite=float(g["predicted_property_rank_score"].median()),
            frac_reduced_cation=float(g["requires_reduced_cation"].mean()),
            frac_open_shell=float(g["has_open_shell"].mean()),
            frac_pass_gate=float(g["passes_quality_gate"].mean()),
            frac_Hf_Zr_Al=float(g["has_stage1_good"].mean()),
            frac_pred_gap_ge_1eV=float((g["pred_band_gap_eV_mean"] >= 1.0).mean()),
            median_nn_train_distance=float(g["nn_train_distance"].median()),
            n_in_training_set=int(g["in_training_set"].sum()),
            n_featurized_ok=int(g["featurized_ok"].sum()),
        )
    (K.RESULTS / "a9_compare_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
