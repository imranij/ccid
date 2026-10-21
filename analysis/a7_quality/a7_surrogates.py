"""a7_surrogates.py -- apply the a5 composition-only surrogates to BOTH populations.

Identical model objects, identical featurization and identical prediction protocol to
a5_score_candidates.py (band gap averaged over the three source one-hots; dielectric
with is_phonon_aux = 0; stability from both per-source models).  Only the input formula
list differs.

CAUTION carried over from a5_REPORT.md: the dielectric surrogate was judged good enough
to REJECT bad candidates, not to rank finely.  Small differences in predicted
permittivity must not be over-read.
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import pickle, sys
import numpy as np, pandas as pd

sys.path.insert(0, _CCID_ROOT + "/analysis/a5_surrogate")
from a5_common import magpie_features, RESULTS_DIR           # noqa: E402
import a5_common                                             # noqa: E402

A5 = _CCID_ROOT + "/analysis/a5_surrogate"
RESULTS = _CCID_ROOT + "/results"
BANDGAP_INSULATOR_THRESHOLD = 1.0


def predict(formulas):
    X, ok = magpie_features(list(formulas))
    with open(f"{A5}/a5_stability_models.pkl", "rb") as f:
        stab = pickle.load(f)
    with open(f"{A5}/a5_gap_model.pkl", "rb") as f:
        gap = pickle.load(f)
    with open(f"{A5}/a5_dielectric_models.pkl", "rb") as f:
        diel = pickle.load(f)

    out = {}
    for src, m in stab.items():
        mu, sd = m.predict(X)
        out[f"pred_stability_{src}_mean"] = mu
        out[f"pred_stability_{src}_std"] = sd
    preds, stds = [], []
    for i in range(3):
        dummy = np.zeros((len(X), 3)); dummy[:, i] = 1.0
        mu, sd = gap.predict(np.hstack([X, dummy]))
        preds.append(mu); stds.append(sd)
    preds = np.stack(preds); stds = np.stack(stds)
    out["pred_band_gap_eV_mean"] = preds.mean(axis=0)
    out["pred_band_gap_eV_std"] = np.sqrt(stds.mean(axis=0) ** 2 + preds.std(axis=0) ** 2)
    Xd = np.hstack([X, np.zeros((len(X), 1))])
    for t in ("poly_electronic", "poly_total"):
        ml, sl = diel[t].predict(Xd)
        out[f"pred_{t}_mean"] = np.expm1(ml)
        out[f"pred_{t}_lower"] = np.expm1(ml - sl)
        out[f"pred_{t}_upper"] = np.expm1(ml + sl)
    d = pd.DataFrame(out)
    d["featurized_ok"] = ok
    return d


def main():
    pops = {"cvae": "a7_population_cvae.csv",
            "random": "a7_population_random.csv",
            "random_matched": "a7_population_random_matched.csv",
            "random_freq": "a7_population_random_freq.csv"}
    frames = []
    for name, path in pops.items():
        p = pd.read_csv(f"{RESULTS}/{path}")
        print(f"featurizing + predicting {name}: {len(p)} formulas", flush=True)
        pr = predict(p["formula"].tolist())
        d = pd.concat([p.reset_index(drop=True), pr], axis=1)
        d["population"] = name
        frames.append(d)
    allf = pd.concat(frames, ignore_index=True)
    allf["flag_low_bandgap_not_insulating"] = allf["pred_band_gap_eV_mean"] < BANDGAP_INSULATOR_THRESHOLD

    # the same 0.4/0.3/0.3 composite a5 used, but normalised on the POOLED population so
    # the two arms are on one scale (a5 normalised within the 66 CVAE candidates only).
    sc = "pred_stability_wolverton_oxides_mean"
    stab_norm = 1.0 - (allf[sc] - allf[sc].min()) / (allf[sc].max() - allf[sc].min() + 1e-12)
    gap_norm = np.clip(allf["pred_band_gap_eV_mean"] / 4.0, 0, 1)
    lg = np.log1p(allf["pred_poly_total_mean"])
    diel_norm = (lg - lg.min()) / (lg.max() - lg.min() + 1e-12)
    allf["predicted_property_rank_score"] = 0.4 * stab_norm + 0.3 * gap_norm + 0.3 * diel_norm

    allf.to_csv(f"{RESULTS}/a7_surrogate_predictions.csv", index=False)
    for n, g in allf.groupby("population"):
        print(f"{n:15s} n={len(g):6d}  Eg {g['pred_band_gap_eV_mean'].median():.3f}  "
              f"eps0 {g['pred_poly_total_mean'].median():.2f}  "
              f"Ehull {g[sc].median():.4f}  score {g['predicted_property_rank_score'].median():.4f}")


if __name__ == "__main__":
    main()
