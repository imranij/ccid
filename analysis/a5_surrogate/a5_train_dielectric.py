"""Surrogate 2: composition-only dielectric-permittivity predictor.

Primary data: Materials Project dielectric_constant.json.gz (1,056 records: formula,
band_gap, poly_electronic, poly_total, n). Composition-only means we use ONLY the
`formula` column (no `structure`, no `cif`, no `poscar`) — none of that survives to
the generated candidates, which have no structure at all.

Auxiliary data: shared_datasets/matminer/phonon_dielectric_mp.json.gz (1,296 records,
523 of which share an mp-id with the primary set — i.e. largely the same underlying
MP calculations, repackaged with a phonon DOS peak). We therefore only add its
NON-overlapping ~773 rows as weak auxiliary training data, flagged by source. Its
eps_total/eps_electronic values include physically implausible outliers (up to
~3e6) that come from near-zero phonon-mode frequencies (structural instabilities/
soft modes) blowing up the ionic contribution — not the everyday "prioritise a
candidate" regime this surrogate is meant to serve. We winsorize by dropping rows
with eps > 1000 before log-transforming.

This is a SMALL dataset (1,056 primary rows). We do NOT hold out a single test
split; the low sample count would make one lucky/unlucky split misleading. Instead:
  - report 10-fold cross-validated MAE/RMSE/R2 on the log-transformed targets
    (log1p, then reported both in log space and back-transformed eV/dimensionless
    space via expm1 of predictions),
  - report an ensemble (bagging) uncertainty as elsewhere,
  - explicitly state, based on the CV R2, whether the surrogate is good enough to
    RANK/prioritise candidates or only good enough to REJECT obviously bad ones.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import functools

from a5_common import (
    PROJECT_ROOT, RESULTS_DIR, magpie_features, BootstrapEnsemble, regression_metrics, save_json,
    rf_factory_default,
)

RNG_SEED = 42
N_ENSEMBLE = 8
N_FOLDS = 10

rf_factory = functools.partial(rf_factory_default, n_estimators=200, max_depth=12, min_samples_leaf=2)


def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and {"data", "columns", "index"} <= set(raw):
        return pd.DataFrame(raw["data"], columns=raw["columns"], index=raw["index"])
    return pd.DataFrame(raw)


def main():
    primary_path = PROJECT_ROOT / "Paper_1_Uncertainty_Calibrated_Data_Repair_for_Ceramic_Materials" / "data" / "dielectric_constant.json.gz"
    phonon_path = PROJECT_ROOT / "shared_datasets" / "matminer" / "phonon_dielectric_mp.json.gz"

    primary = load_json_gz(primary_path)[["material_id", "formula", "band_gap", "poly_electronic", "poly_total", "n"]].copy()
    primary["source"] = "mp_dielectric_primary"
    print(f"Primary dielectric set: {len(primary)} rows")

    phonon = load_json_gz(phonon_path)[["mpid", "formula", "eps_electronic", "eps_total"]].copy()
    overlap_ids = set(primary["material_id"]) & set(phonon["mpid"])
    print(f"Phonon set: {len(phonon)} rows, {len(overlap_ids)} overlap with primary by mp-id -> dropping overlap")
    phonon_aux = phonon[~phonon["mpid"].isin(overlap_ids)].copy()
    phonon_aux = phonon_aux.rename(columns={"eps_electronic": "poly_electronic", "eps_total": "poly_total"})
    phonon_aux["source"] = "phonon_aux"
    n_before = len(phonon_aux)
    phonon_aux = phonon_aux[(phonon_aux["poly_electronic"] < 1000) & (phonon_aux["poly_total"] < 1000)]
    print(f"Phonon aux after winsorizing eps<1000: {len(phonon_aux)} of {n_before} (dropped {n_before-len(phonon_aux)} soft-mode outliers)")

    # We use the auxiliary phonon rows ONLY for poly_total/poly_electronic, not band_gap/n
    # (not provided by that file).
    combined = pd.concat([
        primary[["formula", "poly_electronic", "poly_total", "source"]],
        phonon_aux[["formula", "poly_electronic", "poly_total", "source"]],
    ], ignore_index=True)

    print("Featurizing formulas (Magpie)...")
    X_all, ok = magpie_features(combined["formula"].tolist())
    print(f"{ok.sum()}/{len(combined)} formulas featurized OK")
    combined = combined[ok].reset_index(drop=True)
    X_all = X_all[ok]

    is_primary = (combined["source"] == "mp_dielectric_primary").to_numpy()
    src_dummy = (combined["source"] == "phonon_aux").to_numpy(dtype=np.float64).reshape(-1, 1)
    X_full = np.hstack([X_all, src_dummy])  # 1 extra column flags aux rows

    results = {}
    fold_curves = {}

    for target in ["poly_electronic", "poly_total"]:
        y_log_full = np.log1p(combined[target].to_numpy(dtype=np.float64))

        # ---- Cross-validation, but ONLY evaluate on primary rows (aux rows may
        # only ever be used for training, never scored, since we don't fully trust
        # their comparability) ----
        primary_idx = np.where(is_primary)[0]
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
        oof_pred = np.full(len(primary_idx), np.nan)
        oof_std = np.full(len(primary_idx), np.nan)

        for fold, (tr_local, te_local) in enumerate(kf.split(primary_idx)):
            te_global = primary_idx[te_local]
            tr_primary_global = primary_idx[tr_local]
            # training set = this fold's primary-train rows + ALL aux rows
            aux_idx = np.where(~is_primary)[0]
            tr_global = np.concatenate([tr_primary_global, aux_idx])

            model = BootstrapEnsemble(rf_factory, n_members=N_ENSEMBLE, random_state=RNG_SEED + fold)
            model.fit(X_full[tr_global], y_log_full[tr_global])
            mean, std = model.predict(X_full[te_global])
            oof_pred[te_local] = mean
            oof_std[te_local] = std

        y_true_log = y_log_full[primary_idx]
        y_true_lin = np.expm1(y_true_log)
        y_pred_lin = np.expm1(oof_pred)

        m_log = regression_metrics(y_true_log, oof_pred)
        m_lin = regression_metrics(y_true_lin, y_pred_lin)
        m_lin["mean_ensemble_std_log"] = float(np.mean(oof_std))
        results[target] = {"log_space": m_log, "linear_space": m_lin}
        print(f"[{target}] CV (log space): {m_log}")
        print(f"[{target}] CV (back-transformed, linear space): {m_lin}")

        fig, ax = plt.subplots(figsize=(5, 5))
        lo, hi = min(y_true_lin.min(), y_pred_lin.min()), max(y_true_lin.max(), y_pred_lin.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
        ax.errorbar(y_true_lin, y_pred_lin, yerr=np.expm1(oof_std) * y_pred_lin, fmt="o", ms=3,
                    alpha=0.4, ecolor="lightgray", elinewidth=0.5, color="darkorange")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"10-fold CV parity: {target} (log-log)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"a5_parity_dielectric_{target}.pdf")
        plt.close(fig)

    # ---------------------------------------------------------------
    # Fit FINAL models on ALL data (primary + aux) for downstream candidate scoring
    # ---------------------------------------------------------------
    final_models = {}
    for target in ["poly_electronic", "poly_total"]:
        y_log_full = np.log1p(combined[target].to_numpy(dtype=np.float64))
        model = BootstrapEnsemble(rf_factory, n_members=N_ENSEMBLE, random_state=RNG_SEED)
        model.fit(X_full, y_log_full)
        final_models[target] = model

    import pickle
    with open(_CCID_ROOT + "/analysis/a5_surrogate/a5_dielectric_models.pkl", "wb") as f:
        pickle.dump(final_models, f)

    save_json(results, RESULTS_DIR / "a5_dielectric_metrics.json")

    # verdict
    r2_elec = results["poly_electronic"]["linear_space"]["r2"]
    r2_total = results["poly_total"]["linear_space"]["r2"]
    verdict = {
        "poly_electronic_r2": r2_elec,
        "poly_total_r2": r2_total,
        "verdict": (
            "GOOD ENOUGH TO PRIORITISE/RANK candidates" if min(r2_elec, r2_total) > 0.5
            else "ONLY GOOD ENOUGH TO REJECT OBVIOUSLY BAD CANDIDATES (screen), NOT to rank finely"
        ),
    }
    save_json(verdict, RESULTS_DIR / "a5_dielectric_verdict.json")
    print("VERDICT:", verdict)

    # LaTeX table
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Composition-only dielectric surrogate, 10-fold cross-validated on the "
               r"1,056-record Materials Project dielectric set (targets log1p-transformed; linear-space "
               r"metrics are back-transformed via expm1). Non-overlapping rows of the phonon dielectric "
               r"set (773 of 1,296, after dropping mp-id overlap and winsorizing $\varepsilon>1000$ "
               r"soft-mode outliers) were added as auxiliary training-only rows, never scored.}")
    tex.append(r"\label{tab:a5_dielectric}")
    tex.append(r"\begin{tabular}{lrrrr}")
    tex.append(r"\toprule")
    tex.append(r"Target (linear space) & $n$ & MAE & RMSE & $R^2$ \\")
    tex.append(r"\midrule")
    for target, label in [("poly_electronic", r"$\varepsilon_{\infty}$ (electronic)"),
                          ("poly_total", r"$\varepsilon_{0}$ (total)")]:
        m = results[target]["linear_space"]
        tex.append(f"{label} & {m['n']} & {m['mae']:.3f} & {m['rmse']:.3f} & {m['r2']:.3f} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(RESULTS_DIR / "a5_table_dielectric.tex", "w") as f:
        f.write("\n".join(tex) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
