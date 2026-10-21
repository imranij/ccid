"""Step 3: apply the composition-only surrogates (stability, band gap, dielectric)
to the 66 unique valid generated candidates (valid==True rows of
composition_cvae_generated_candidates.csv, deduplicated on canonical_formula), and
build a ranking by PREDICTED PROPERTIES to replace the meaningless
target-proximity ("candidate_score") ranking.

For stability we report BOTH per-source predictions (Castelli-style e_form-like
and Wolverton-style e_hull-like) since the candidates have no ground-truth source
label — a candidate composition is not "from" either dataset. We surface both so
the reader can see whether the two stability notions actually disagree.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import pickle
import numpy as np
import pandas as pd

from a5_common import DGX_OUT, RESULTS_DIR, magpie_features, HIGH_RISK_ELEMENTS

BANDGAP_INSULATOR_THRESHOLD = 1.0  # eV; matches the paper's Eg >= 1.0 eV dielectric filter (Stage 1)


def main():
    cand = pd.read_csv(DGX_OUT / "composition_cvae_generated_candidates.csv")
    valid = cand[cand["valid"] == True].drop_duplicates(subset="canonical_formula").reset_index(drop=True)
    print(f"{len(valid)} unique valid candidates")

    X, ok = magpie_features(valid["formula"].tolist())
    if not ok.all():
        print("WARNING: some candidates failed featurization:", valid.loc[~ok, "formula"].tolist())
    valid = valid[ok].reset_index(drop=True)
    X = X[ok]

    # ---- Stability (per-source models) ----
    with open(_CCID_ROOT + "/analysis/a5_surrogate/a5_stability_models.pkl", "rb") as f:
        stab_models = pickle.load(f)
    for source, model in stab_models.items():
        mean, std = model.predict(X)
        valid[f"pred_stability_{source}_mean"] = mean
        valid[f"pred_stability_{source}_std"] = std

    # ---- Band gap ----
    with open(_CCID_ROOT + "/analysis/a5_surrogate/a5_gap_model.pkl", "rb") as f:
        gap_model = pickle.load(f)
    # gap model was trained with a one-hot source_dataset feature appended (3 dummies,
    # alphabetical: castelli_perovskites, double_perovskites_gap, wolverton_oxides).
    # Candidates have no known source; we average over the three one-hot settings to
    # avoid arbitrarily picking one, and report the spread as part of the uncertainty.
    n_src = 3
    preds = []
    stds = []
    for i in range(n_src):
        dummy = np.zeros((len(X), n_src))
        dummy[:, i] = 1.0
        X_gap = np.hstack([X, dummy])
        mean, std = gap_model.predict(X_gap)
        preds.append(mean)
        stds.append(std)
    preds = np.stack(preds, axis=0)
    valid["pred_band_gap_eV_mean"] = preds.mean(axis=0)
    # total uncertainty = ensemble std (within-source) blended with across-source-assumption spread
    within_source_std = np.stack(stds, axis=0).mean(axis=0)
    across_source_std = preds.std(axis=0)
    valid["pred_band_gap_eV_std"] = np.sqrt(within_source_std**2 + across_source_std**2)
    valid["pred_band_gap_source_sensitivity"] = across_source_std

    # ---- Dielectric ----
    with open(_CCID_ROOT + "/analysis/a5_surrogate/a5_dielectric_models.pkl", "rb") as f:
        diel_models = pickle.load(f)
    # dielectric model expects an extra "is_phonon_aux" column; candidates are
    # neither MP-primary nor phonon-aux, so we set it to 0 (use the primary-style
    # calibration, which is the trustworthy one).
    X_diel = np.hstack([X, np.zeros((len(X), 1))])
    for target in ["poly_electronic", "poly_total"]:
        mean_log, std_log = diel_models[target].predict(X_diel)
        valid[f"pred_{target}_mean"] = np.expm1(mean_log)
        # propagate log-space uncertainty to linear space (asymmetric; report both bounds)
        valid[f"pred_{target}_lower"] = np.expm1(mean_log - std_log)
        valid[f"pred_{target}_upper"] = np.expm1(mean_log + std_log)

    # ---- flags ----
    # The point estimate alone is not enough: several candidates have a predicted
    # band gap only slightly above 1.0 eV with an ensemble std comparable to or
    # larger than the mean (band-gap surrogate is trained mostly on gap==0 rows,
    # see a5_REPORT.md), so we flag both the point estimate AND a "mean minus one
    # std falls under threshold" low-confidence case, surfaced separately.
    valid["flag_low_bandgap_not_insulating"] = valid["pred_band_gap_eV_mean"] < BANDGAP_INSULATOR_THRESHOLD
    valid["flag_uncertain_insulator"] = (
        (valid["pred_band_gap_eV_mean"] - valid["pred_band_gap_eV_std"]) < BANDGAP_INSULATOR_THRESHOLD
    ) & (~valid["flag_low_bandgap_not_insulating"])
    valid["has_open_shell_risk_element"] = valid["elements"].fillna("").apply(
        lambda s: any(e in {"Ti", "Sn"} for e in s.split())
    )

    # ---- ranking by predicted properties (replaces meaningless target-proximity score) ----
    # Use the Wolverton-style (e_hull-like) stability model as the primary stability axis
    # since e_hull (distance above convex hull) is the more standard, target-agnostic
    # thermodynamic stability metric for screening (lower = more stable, so we invert).
    stab_col = "pred_stability_wolverton_oxides_mean"
    stab_norm = 1.0 - (valid[stab_col] - valid[stab_col].min()) / (valid[stab_col].max() - valid[stab_col].min() + 1e-12)
    gap_norm = np.clip(valid["pred_band_gap_eV_mean"] / 4.0, 0, 1)  # saturate around a wide-gap target of ~4 eV
    diel_norm = (np.log1p(valid["pred_poly_total_mean"]) - np.log1p(valid["pred_poly_total_mean"]).min()) / (
        np.log1p(valid["pred_poly_total_mean"]).max() - np.log1p(valid["pred_poly_total_mean"]).min() + 1e-12
    )
    valid["predicted_property_rank_score"] = 0.4 * stab_norm + 0.3 * gap_norm + 0.3 * diel_norm
    valid = valid.sort_values("predicted_property_rank_score", ascending=False).reset_index(drop=True)
    valid.insert(0, "predicted_rank", np.arange(1, len(valid) + 1))

    out_cols = [
        "predicted_rank", "sample_id", "formula", "canonical_formula", "elements",
        "target_stability_utility", "target_gap_utility", "candidate_score",  # the OLD meaningless score, kept for comparison
        "pred_stability_castelli_perovskites_mean", "pred_stability_castelli_perovskites_std",
        "pred_stability_wolverton_oxides_mean", "pred_stability_wolverton_oxides_std",
        "pred_band_gap_eV_mean", "pred_band_gap_eV_std", "pred_band_gap_source_sensitivity",
        "pred_poly_electronic_mean", "pred_poly_electronic_lower", "pred_poly_electronic_upper",
        "pred_poly_total_mean", "pred_poly_total_lower", "pred_poly_total_upper",
        "flag_low_bandgap_not_insulating", "flag_uncertain_insulator", "has_open_shell_risk_element",
        "predicted_property_rank_score",
    ]
    valid[out_cols].to_csv(RESULTS_DIR / "a5_candidate_predictions.csv", index=False)

    n_low_gap = int(valid["flag_low_bandgap_not_insulating"].sum())
    n_uncertain = int(valid["flag_uncertain_insulator"].sum())
    n_open_shell = int(valid["has_open_shell_risk_element"].sum())
    print(f"Wrote {len(valid)} scored candidates to a5_candidate_predictions.csv")
    print(f"{n_low_gap}/{len(valid)} candidates predicted band gap < {BANDGAP_INSULATOR_THRESHOLD} eV (flagged, not buried)")
    print(f"{n_uncertain}/{len(valid)} more candidates have mean-std below {BANDGAP_INSULATOR_THRESHOLD} eV (uncertain insulator status)")
    print(f"{n_open_shell}/{len(valid)} candidates contain Ti or Sn (open-shell-capable risk elements)")

    # LaTeX table: top 15 by predicted-property rank
    top = valid.head(15)
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Top 15 of the 66 unique valid generated candidates, ranked by "
           r"predicted properties (stability, band gap, dielectric permittivity) from the "
           r"composition-only surrogates, replacing the target-proximity score (which "
           r"carried no candidate-specific information; see Sec.~X). $^\dagger$flags a "
           r"predicted band gap below 1.0~eV (unsuitable as a wide-gap insulator).}",
           r"\label{tab:a5_candidate_ranking}",
           r"\begin{tabular}{rlrrrrr}", r"\toprule",
           r"Rank & Formula & $\hat{E}_{\mathrm{hull}}$ (eV/at) & $\hat{E}_g$ (eV) & $\hat\varepsilon_{\infty}$ & $\hat\varepsilon_{0}$ & Flag \\",
           r"\midrule"]
    for _, r in top.iterrows():
        flag = r"$\dagger$" if r["flag_low_bandgap_not_insulating"] else ""
        tex.append(
            f"{int(r['predicted_rank'])} & {r['canonical_formula']} & "
            f"{r['pred_stability_wolverton_oxides_mean']:.3f} & {r['pred_band_gap_eV_mean']:.3f} & "
            f"{r['pred_poly_electronic_mean']:.2f} & {r['pred_poly_total_mean']:.2f} & {flag} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(RESULTS_DIR / "a5_table_candidate_ranking.tex", "w") as f:
        f.write("\n".join(tex) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
