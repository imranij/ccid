"""Surrogate 1: composition-only stability and band-gap predictors, trained on
dgx/outputs_dgx/composition_cvae_training_records.csv (25,148 rows) using the
SAME deterministic 80/10/10 split the CVAE used (split_indices, seed=42), so
results are comparable to the CVAE's own reported test metrics.

Key handling of the stability trap (see the audit brief, corroborated below):
`stability` pools Castelli formation energy (e_form, eV/atom, referenced to
elemental phases) with Wolverton energy-above-hull (e_hull, eV/atom, referenced
to the convex hull) — these are numerically and physically different quantities.
A single pooled regressor would silently absorb this as noise. We therefore:
  (a) show the pooled (naive) fit as a documented negative-example baseline,
  (b) train and report PER-SOURCE stability models (Castelli-only, Wolverton-only)
      as the primary, trustworthy result,
  (c) train stability only on rows with stability_known==1 (drops the
      double_perovskites_gap rows, whose "stability"==0 is a placeholder, not data).

Band gap is trained as a single composition-only model across all three sources
(gap is the same physical quantity, eV, in all three, albeit computed with
different DFT flavours: GLLB-SC for Castelli/double-perovskites, PBE for
Wolverton) with source_dataset supplied as a one-hot feature. Overall AND
non-zero-gap-subset metrics are reported separately, because ~88% of rows have
gap==0 exactly and a trivial "always predict 0" model would look deceptively
good on the pooled metric.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import functools

from a5_common import (
    DGX_OUT, RESULTS_DIR, split_indices, BootstrapEnsemble, regression_metrics, save_json,
    rf_factory_default,
)

RNG_SEED = 42
N_ENSEMBLE = 6

rf_factory = functools.partial(rf_factory_default, n_estimators=80, max_depth=20, min_samples_leaf=2)


def load_data():
    df = pd.read_csv(DGX_OUT / "composition_cvae_training_records.csv")
    feats = np.load(_CCID_ROOT + "/analysis/a5_surrogate/a5_training_magpie_features.npz", allow_pickle=True)
    X = feats["X"]
    ok = feats["ok"]
    assert ok.all(), "some formulas failed featurization; cannot proceed silently"
    splits = split_indices(len(df), seed=RNG_SEED)
    return df, X, splits


def parity_plot(y_true, y_pred, y_std, title, path, unit=""):
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    sc = ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ms=3, alpha=0.35,
                      ecolor="lightgray", elinewidth=0.5, capsize=0, color="steelblue")
    ax.set_xlabel(f"True {title}{unit}")
    ax.set_ylabel(f"Predicted {title}{unit}")
    ax.set_title(f"Parity: {title} (test set)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    df, X, splits = load_data()
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
    # val is folded into train for these tree ensembles (no early-stopping needed);
    # test is held out untouched, exactly as the CVAE's own test split.
    fit_idx = np.concatenate([train_idx, val_idx])

    results = {}

    # ---------------------------------------------------------------
    # A) STABILITY — naive pooled baseline (documented negative example)
    # ---------------------------------------------------------------
    known = df["stability_known"].astype(bool).to_numpy()
    fit_known = np.array([i for i in fit_idx if known[i]])
    test_known = np.array([i for i in test_idx if known[i]])

    naive = BootstrapEnsemble(rf_factory, n_members=N_ENSEMBLE, random_state=RNG_SEED)
    naive.fit(X[fit_known], df["stability"].to_numpy()[fit_known])
    pred_mean, pred_std = naive.predict(X[test_known])
    naive_metrics = regression_metrics(df["stability"].to_numpy()[test_known], pred_mean)
    results["stability_pooled_naive"] = naive_metrics
    print("Pooled naive stability (documented as a cautionary baseline):", naive_metrics)

    # cross-source offset check (corroborates the audit brief's flagged 2.37 eV/atom claim
    # using only the rows actually in this table)
    means = df[known].groupby("source_dataset")["stability"].mean()
    offset_report = means.to_dict()
    results["stability_source_means_eV_per_atom"] = offset_report
    print("Per-source mean stability (documents the pooling trap):", offset_report)

    # ---------------------------------------------------------------
    # B) STABILITY — per-source models (primary result)
    # ---------------------------------------------------------------
    per_source_models = {}
    per_source_metrics = {}
    for source in ["castelli_perovskites", "wolverton_oxides"]:
        src_mask = (df["source_dataset"] == source).to_numpy() & known
        f_idx = np.array([i for i in fit_idx if src_mask[i]])
        t_idx = np.array([i for i in test_idx if src_mask[i]])
        model = BootstrapEnsemble(rf_factory, n_members=N_ENSEMBLE, random_state=RNG_SEED)
        model.fit(X[f_idx], df["stability"].to_numpy()[f_idx])
        mean, std = model.predict(X[t_idx])
        y_test = df["stability"].to_numpy()[t_idx]
        m = regression_metrics(y_test, mean)
        m["mean_ensemble_std"] = float(np.mean(std))
        per_source_metrics[source] = m
        per_source_models[source] = model
        unit = " (eV/atom, e_form)" if source == "castelli_perovskites" else " (eV/atom, e_hull)"
        parity_plot(y_test, mean, std, f"stability [{source}]",
                    RESULTS_DIR / f"a5_parity_stability_{source}.pdf", unit=unit)
        print(f"Per-source stability [{source}]:", m)
    results["stability_per_source"] = per_source_metrics

    import pickle
    with open(_CCID_ROOT + "/analysis/a5_surrogate/a5_stability_models.pkl", "wb") as f:
        pickle.dump(per_source_models, f)

    # ---------------------------------------------------------------
    # C) BAND GAP — single composition-only model, source one-hot appended
    # ---------------------------------------------------------------
    source_dummies = pd.get_dummies(df["source_dataset"], prefix="src").to_numpy(dtype=np.float64)
    X_gap = np.hstack([X, source_dummies])

    gap_model = BootstrapEnsemble(rf_factory, n_members=N_ENSEMBLE, random_state=RNG_SEED)
    gap_model.fit(X_gap[fit_idx], df["gap"].to_numpy()[fit_idx])
    mean, std = gap_model.predict(X_gap[test_idx])
    y_test = df["gap"].to_numpy()[test_idx]
    overall_metrics = regression_metrics(y_test, mean)
    overall_metrics["mean_ensemble_std"] = float(np.mean(std))
    results["gap_overall"] = overall_metrics
    print("Band gap, ALL test rows (dominated by the exact-zero subset):", overall_metrics)

    nz_mask_test = y_test != 0.0
    nz_metrics = regression_metrics(y_test[nz_mask_test], mean[nz_mask_test])
    nz_metrics["mean_ensemble_std"] = float(np.mean(std[nz_mask_test]))
    results["gap_nonzero_subset"] = nz_metrics
    print(f"Band gap, NON-ZERO-only test rows (n={nz_mask_test.sum()} of {len(y_test)}):", nz_metrics)

    zero_frac_train = float((df["gap"].to_numpy()[fit_idx] == 0).mean())
    zero_frac_test = float((y_test == 0).mean())
    results["gap_zero_fraction"] = {"train_val": zero_frac_train, "test": zero_frac_test}

    with open(_CCID_ROOT + "/analysis/a5_surrogate/a5_gap_model.pkl", "wb") as f:
        pickle.dump(gap_model, f)

    parity_plot(y_test, mean, std, "band gap (all test rows)",
                RESULTS_DIR / "a5_parity_gap_all.pdf", unit=" (eV)")
    parity_plot(y_test[nz_mask_test], mean[nz_mask_test], std[nz_mask_test],
                "band gap (non-zero-gap test subset)",
                RESULTS_DIR / "a5_parity_gap_nonzero.pdf", unit=" (eV)")

    # ---------------------------------------------------------------
    # Save everything
    # ---------------------------------------------------------------
    save_json(results, RESULTS_DIR / "a5_stability_gap_metrics.json")

    # LaTeX table (booktabs)
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Composition-only surrogate performance for stability and band gap "
               r"(test split, 80/10/10 seed-42 split matching the CVAE). Stability is trained "
               r"per source because Castelli $e_{\mathrm{form}}$ and Wolverton $e_{\mathrm{hull}}$ "
               r"are different physical quantities; the pooled fit is shown only as a cautionary "
               r"baseline. Band gap is reported both over all test rows and over the subset with "
               r"a non-zero reference gap, since %.0f\%% of rows have gap $=0$ exactly.}" % (100*zero_frac_test))
    tex.append(r"\label{tab:a5_stability_gap}")
    tex.append(r"\begin{tabular}{lrrrr}")
    tex.append(r"\toprule")
    tex.append(r"Target & $n_{\mathrm{test}}$ & MAE & RMSE & $R^2$ \\")
    tex.append(r"\midrule")
    def row(label, m):
        return f"{label} & {m['n']} & {m['mae']:.4f} & {m['rmse']:.4f} & {m['r2']:.4f} \\\\"
    tex.append(row("Stability, pooled (naive, not recommended)", naive_metrics))
    tex.append(row(r"Stability, Castelli only ($e_{\mathrm{form}}$)", per_source_metrics["castelli_perovskites"]))
    tex.append(row(r"Stability, Wolverton only ($e_{\mathrm{hull}}$)", per_source_metrics["wolverton_oxides"]))
    tex.append(r"\midrule")
    tex.append(row("Band gap, all test rows", overall_metrics))
    tex.append(row("Band gap, non-zero-gap subset", nz_metrics))
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(RESULTS_DIR / "a5_table_stability_gap.tex", "w") as f:
        f.write("\n".join(tex) + "\n")

    print("Done. Results written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
