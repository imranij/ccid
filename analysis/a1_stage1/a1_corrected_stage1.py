"""Corrected Stage-1 screen for MTCOMM-D-26-04679 (agent a1).

Produces, in results/ with prefix a1_ :
  * an exact re-scoring of the legacy pipeline (audit / before tables)
  * a filters-only variant  (Eg>=1 eV on every pool + harmonised element exclusion)
  * the corrected variant   (filters + harmonised scoring + commensurate stability metric)
  * an imputation sensitivity sweep over the double-perovskite stability utility

Run:  python a1_corrected_stage1.py
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_common import (  # noqa: E402
    EXCLUDED_ELEMENTS, RESULTS, brute_force_pareto, candidate_pool_summary,
    diverse_acquisition_batch, elements_of, gate_summary, load_dielectric,
    load_perovskite_pool, minmax, objective_sensitivity, pareto_front,
    score_dielectric, validation_gate,
)

RESULTS.mkdir(parents=True, exist_ok=True)
OUT = RESULTS


def score_pool(
    pool: pd.DataFrame,
    *,
    gap_filter: bool,
    element_filter: bool,
    stability_field: str,      # "legacy" | "e_form"
    stability_norm: str,       # "per_dataset" | "pooled"
    gap_norm: str,             # "per_dataset" | "pooled"
    weighting: str,            # "legacy" | "harmonised"
    rho: float = 0.5,
) -> pd.DataFrame:
    df = pool.copy()
    if element_filter:
        df = df[df["excluded_elements"] == ""].copy()
    if gap_filter:
        df = df[df["band_gap"] >= 1.0].copy()
    if df.empty:
        return df

    # --- stability metric -----------------------------------------------
    if stability_field == "legacy":
        metric = np.where(
            df["source_dataset"] == "wolverton_oxides", df["e_hull"], df["e_form"]
        )
        metric_name = np.where(
            df["source_dataset"] == "wolverton_oxides", "e_hull",
            np.where(df["source_dataset"] == "castelli_perovskites", "e_form", "imputed"),
        )
    else:
        metric = df["e_form"].to_numpy(float)
        metric_name = np.where(df["stability_known"] == 1, "e_form", "imputed")
    df["stability_metric"] = metric
    df["stability_metric_type"] = metric_name
    # legacy pipeline dropped records with a missing stability metric via dropna()
    known = df["stability_known"] == 1
    df = df[~(known & df["stability_metric"].isna())].copy()
    known = df["stability_known"] == 1

    # --- utilities -------------------------------------------------------
    su = pd.Series(float(rho), index=df.index)
    gu = pd.Series(np.nan, index=df.index)
    if gap_norm == "per_dataset":
        for _ds, idx in df.groupby("source_dataset").groups.items():
            gu.loc[idx] = minmax(df.loc[idx, "band_gap"]).values
    else:
        gu.loc[:] = minmax(df["band_gap"]).values
    if stability_norm == "per_dataset":
        for ds, idx in df.groupby("source_dataset").groups.items():
            if ds != "double_perovskites_gap":
                su.loc[idx] = minmax(df.loc[idx, "stability_metric"], invert=True).values
    elif known.any():
        su.loc[known[known].index] = minmax(
            df.loc[known, "stability_metric"], invert=True
        ).values
    df["stability_utility"] = su
    df["gap_utility"] = gu
    df["direct_gap_utility"] = df["direct_gap"].astype(float)
    # legacy code zeroes the direct-gap utility for the two datasets that do not report it
    df.loc[df["source_dataset"] != "castelli_perovskites", "direct_gap_utility"] = 0.0

    # --- scalarisation ---------------------------------------------------
    if weighting == "legacy":
        s = pd.Series(np.nan, index=df.index)
        m = df["source_dataset"] == "castelli_perovskites"
        s[m] = (0.45 * df.loc[m, "stability_utility"] + 0.35 * df.loc[m, "gap_utility"]
                + 0.20 * df.loc[m, "direct_gap_utility"])
        m = df["source_dataset"] == "wolverton_oxides"
        s[m] = 0.65 * df.loc[m, "stability_utility"] + 0.35 * df.loc[m, "gap_utility"]
        m = df["source_dataset"] == "double_perovskites_gap"
        s[m] = df.loc[m, "gap_utility"]
        df["candidate_score"] = s
    else:  # harmonised - the equation actually printed in the manuscript
        df["candidate_score"] = 0.5 * df["stability_utility"] + 0.5 * df["gap_utility"]

    cols = ["source_dataset", "formula", "stability_metric", "stability_metric_type",
            "band_gap", "direct_gap", "stability_known", "stability_utility",
            "gap_utility", "direct_gap_utility", "candidate_score",
            "excluded_elements", "process_risk_elements"]
    return df[cols].sort_values("candidate_score", ascending=False).reset_index(drop=True)


def full_bundle(ranked: pd.DataFrame, tag: str, write: bool = True) -> dict:
    par = pareto_front(ranked)
    bf = brute_force_pareto(ranked)
    summ = candidate_pool_summary(ranked)
    sens = objective_sensitivity(ranked)
    acq = diverse_acquisition_batch(ranked, batch_size=20)
    rev = validation_gate(acq)
    gates = gate_summary(rev)
    bundle = dict(ranked=ranked, pareto=par, pareto_bruteforce=bf, pool_summary=summ,
                  sensitivity=sens, acquisition=acq, review=rev, gates=gates)
    if write:
        ranked.head(200).to_csv(OUT / f"a1_{tag}_top200.csv", index=False)
        par.to_csv(OUT / f"a1_{tag}_pareto.csv", index=False)
        summ.to_csv(OUT / f"a1_{tag}_pool_summary.csv", index=False)
        sens.to_csv(OUT / f"a1_{tag}_objective_sensitivity.csv", index=False)
        acq.to_csv(OUT / f"a1_{tag}_acquisition_batch.csv", index=False)
        rev.to_csv(OUT / f"a1_{tag}_expert_review_pack.csv", index=False)
        gates.to_csv(OUT / f"a1_{tag}_validation_gate_summary.csv", index=False)
    return bundle


def main() -> None:
    pool = load_perovskite_pool()
    print(f"raw pooled perovskite/oxide records: {len(pool)}")
    n_excl = (pool["excluded_elements"] != "").sum()
    print(f"records containing an excluded element: {n_excl}")
    print(f"records with Eg >= 1.0 eV: {(pool['band_gap'] >= 1.0).sum()}")

    dup = set(pool.loc[pool.source_dataset == "castelli_perovskites", "formula"]) & set(
        pool.loc[pool.source_dataset == "wolverton_oxides", "formula"])
    print(f"formulas present in BOTH castelli and wolverton: {len(dup)}")
    print(f"unique formulas in pooled 25,148-record table: {pool['formula'].nunique()}")

    # ---------------------------------------------------------------- V0
    v0 = score_pool(pool, gap_filter=False, element_filter=False,
                    stability_field="legacy", stability_norm="per_dataset",
                    gap_norm="per_dataset", weighting="legacy", rho=0.5)
    b0 = full_bundle(v0, "legacy")
    print(f"\n[V0 legacy] {len(v0)} records, pareto {len(b0['pareto'])} "
          f"(brute-force {len(b0['pareto_bruteforce'])})")

    # ---------------------------------------------------------------- V1
    v1 = score_pool(pool, gap_filter=True, element_filter=True,
                    stability_field="legacy", stability_norm="per_dataset",
                    gap_norm="per_dataset", weighting="legacy", rho=0.5)
    b1 = full_bundle(v1, "filtersonly")
    print(f"[V1 filters only] {len(v1)} records, pareto {len(b1['pareto'])} "
          f"(brute-force {len(b1['pareto_bruteforce'])})")

    # ---------------------------------------------------------------- V2
    v2 = score_pool(pool, gap_filter=True, element_filter=True,
                    stability_field="legacy", stability_norm="per_dataset",
                    gap_norm="pooled", weighting="harmonised", rho=0.5)
    b2 = full_bundle(v2, "corrected")
    print(f"[V2 corrected rho=0.5] {len(v2)} records, pareto {len(b2['pareto'])} "
          f"(brute-force {len(b2['pareto_bruteforce'])})")

    # ------------------------- V3: stability-metric harmonisation probe
    v3 = score_pool(pool, gap_filter=True, element_filter=True,
                    stability_field="e_form", stability_norm="pooled",
                    gap_norm="pooled", weighting="harmonised", rho=0.5)
    b3 = full_bundle(v3, "eformprobe")
    print(f"[V3 e_form probe] {len(v3)} records, pareto {len(b3['pareto'])}")
    overlap = len(set(v2.head(20)['formula']) & set(v3.head(20)['formula']))
    print(f"  top-20 overlap between V2 and V3: {overlap}/20")

    # ------------------------------------------------- imputation sweep
    sweep_rows, batch_rows = [], []
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = score_pool(pool, gap_filter=True, element_filter=True,
                       stability_field="legacy", stability_norm="per_dataset",
                       gap_norm="pooled", weighting="harmonised", rho=rho)
        acq = diverse_acquisition_batch(r, batch_size=20)
        dp = r["source_dataset"] == "double_perovskites_gap"
        casi = r.index[r["formula"] == "CaSiYAlO6"]
        sweep_rows.append({
            "rho": rho,
            "top_formula": r.iloc[0]["formula"],
            "top_source_dataset": r.iloc[0]["source_dataset"],
            "top_score": r.iloc[0]["candidate_score"],
            "rank_CaSiYAlO6": int(casi[0]) + 1 if len(casi) else -1,
            "score_CaSiYAlO6": float(r.loc[casi[0], "candidate_score"]) if len(casi) else np.nan,
            "double_perovskites_in_top20": int(dp.head(20).sum()),
            "double_perovskites_in_top50": int(dp.head(50).sum()),
            "double_perovskites_in_batch20": int((acq["source_dataset"] == "double_perovskites_gap").sum()),
            "mean_score_double": float(r.loc[dp, "candidate_score"].mean()),
            "mean_score_known": float(r.loc[~dp, "candidate_score"].mean()),
        })
        for _, row in acq.iterrows():
            batch_rows.append({"rho": rho, "acquisition_rank": row["acquisition_rank"],
                               "source_dataset": row["source_dataset"], "formula": row["formula"],
                               "band_gap": row["band_gap"],
                               "stability_utility": row["stability_utility"],
                               "gap_utility": row["gap_utility"],
                               "candidate_score": row["candidate_score"]})
        r.head(50).to_csv(OUT / f"a1_corrected_rho{rho:.2f}_top50.csv", index=False)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT / "a1_imputation_sensitivity.csv", index=False)
    pd.DataFrame(batch_rows).to_csv(OUT / "a1_imputation_acquisition_batches.csv", index=False)
    print("\n[imputation sweep]")
    print(sweep.to_string(index=False))

    # ------------------------------------------------------- dielectric
    die = load_dielectric()
    die["excluded_elements"] = die["formula"].map(
        lambda f: " ".join(sorted(elements_of(f) & EXCLUDED_ELEMENTS)))
    legacy_die = score_dielectric(die)
    legacy_die.to_csv(OUT / "a1_dielectric_legacy_ranked.csv", index=False)
    corrected_die = score_dielectric(die[die["excluded_elements"] == ""])
    corrected_die.to_csv(OUT / "a1_dielectric_corrected_ranked.csv", index=False)
    # oxide-only supplementary view
    oxide_only = corrected_die[corrected_die["formula"].map(lambda f: "O" in elements_of(f))]
    oxide_only.to_csv(OUT / "a1_dielectric_corrected_oxideonly.csv", index=False)
    print(f"\n[dielectric] 1056 raw -> {len(legacy_die)} after Eg>=1 (legacy) "
          f"-> {len(corrected_die)} after Eg>=1 + element exclusion; "
          f"{len(oxide_only)} of those are oxides")
    print(legacy_die.head(5)[["material_id", "formula", "inverse_design_score"]].to_string(index=False))
    print(corrected_die.head(5)[["material_id", "formula", "inverse_design_score"]].to_string(index=False))

    # what got removed, for the before/after narrative
    removed = legacy_die[legacy_die["formula"].map(
        lambda f: bool(elements_of(f) & EXCLUDED_ELEMENTS))]
    removed.head(50).to_csv(OUT / "a1_dielectric_removed_by_exclusion.csv", index=False)
    print(f"dielectric records removed by element exclusion: {len(removed)}")


if __name__ == "__main__":
    main()
