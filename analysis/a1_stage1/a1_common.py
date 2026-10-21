"""Shared helpers for the corrected Stage-1 screen (agent a1).

Everything here is computed from the datasets shipped in the project tree:
  shared_datasets/matminer/{castelli_perovskites,wolverton_oxides,double_perovskites_gap}.json.gz
  Paper_1_Uncertainty_Calibrated_Data_Repair_for_Ceramic_Materials/data/dielectric_constant.json.gz
No network access is used.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

PROJECT = Path(_CCID_ROOT + "")
MATMINER = PROJECT / "shared_datasets" / "matminer"
DIELECTRIC = (
    PROJECT
    / "Paper_1_Uncertainty_Calibrated_Data_Repair_for_Ceramic_Materials"
    / "data"
    / "dielectric_constant.json.gz"
)
RESULTS = Path(_CCID_ROOT + "/results")

ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")

# ---------------------------------------------------------------------------
# Harmonised element-exclusion list (Stage 1 == Stage 2).
# Group 1: no stable isotope / radioactive.
RADIOACTIVE = {
    "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
    "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}
# Group 2: RoHS / REACH-restricted toxic elements (already in the Stage-2 HIGH_RISK list).
TOXIC_RESTRICTED = {"Pb", "Cd", "Hg", "Tl", "As", "Be"}
# Group 3: noble gases - cannot occupy a ceramic lattice site.
NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}

EXCLUDED_ELEMENTS = RADIOACTIVE | TOXIC_RESTRICTED | NOBLE_GASES

# Informational only - flagged, not excluded (matches Stage-2 PROCESS_RISK_ELEMENTS).
PROCESS_RISK_ELEMENTS = {"F", "Cl", "Br", "I", "Se", "Te", "S"}

# Legacy Stage-2 list, for the before/after comparison.
LEGACY_HIGH_RISK = {"Pb", "Cd", "Hg", "As", "Tl", "Be", "Po", "Ra", "Th", "U", "Pu"}


def elements_of(formula: str) -> set[str]:
    return set(ELEMENT_RE.findall(str(formula)))


def excluded_elements_in(formula: str) -> str:
    return " ".join(sorted(elements_of(formula) & EXCLUDED_ELEMENTS))


def process_risk_in(formula: str) -> str:
    return " ".join(sorted(elements_of(formula) & PROCESS_RISK_ELEMENTS))


def load_json_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict) and {"data", "columns", "index"} <= set(raw):
        return pd.DataFrame(raw["data"], columns=raw["columns"], index=raw["index"])
    return pd.DataFrame(raw)


def minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    """Reproduces inverse_design_baseline.safe_scaled exactly."""
    values = pd.to_numeric(series, errors="coerce")
    if invert:
        values = -values
    if values.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index)
    return pd.Series(
        MinMaxScaler().fit_transform(values.to_numpy().reshape(-1, 1)).reshape(-1),
        index=series.index,
    )


# ---------------------------------------------------------------------------
def load_dielectric() -> pd.DataFrame:
    raw = json.load(gzip.open(DIELECTRIC, "rt", encoding="utf-8"))
    df = pd.DataFrame(raw["data"], columns=raw["columns"], index=raw["index"])
    cols = [
        "material_id", "formula", "nsites", "space_group", "volume", "band_gap",
        "n", "poly_electronic", "poly_total", "pot_ferroelectric",
    ]
    df = df[cols].copy()
    df["pot_ferroelectric"] = df["pot_ferroelectric"].astype(int)
    numeric = [c for c in cols if c not in ("material_id", "formula")]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df.dropna()


def score_dielectric(df: pd.DataFrame) -> pd.DataFrame:
    """Exact re-implementation of inverse_design_baseline.score_candidates,
    applied to whatever population is passed in (so the min-max reference
    population follows the filters)."""
    df = df[df["band_gap"] >= 1.0].copy()
    objectives = ["n", "poly_total", "band_gap"]
    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(df[objectives]), columns=objectives, index=df.index
    )
    desirability = 0.45 * scaled["poly_total"] + 0.35 * scaled["n"] + 0.20 * scaled["band_gap"]
    feature_cols = ["nsites", "space_group", "volume", "band_gap", "n", "poly_total"]
    features = MinMaxScaler().fit_transform(df[feature_cols])
    distances = pairwise_distances(features)
    np.fill_diagonal(distances, np.nan)
    novelty = np.nanmean(np.sort(distances, axis=1)[:, :10], axis=1)
    novelty = MinMaxScaler().fit_transform(novelty.reshape(-1, 1)).reshape(-1)
    out = df.copy()
    out["desirability_score"] = desirability
    out["local_novelty_score"] = novelty
    out["inverse_design_score"] = 0.80 * out["desirability_score"] + 0.20 * out["local_novelty_score"]
    return out.sort_values("inverse_design_score", ascending=False)


# ---------------------------------------------------------------------------
def load_perovskite_pool() -> pd.DataFrame:
    """Raw pooled perovskite/oxide records with BOTH stability metrics kept."""
    rows = []
    df = load_json_gz(MATMINER / "castelli_perovskites.json.gz")
    rows.append(pd.DataFrame({
        "source_dataset": "castelli_perovskites",
        "formula": df["formula"].astype(str),
        "e_form": pd.to_numeric(df["e_form"], errors="coerce"),
        "e_hull": np.nan,
        "band_gap": pd.to_numeric(df["gap gllbsc"], errors="coerce"),
        "direct_gap": df["gap is direct"].astype(int),
        "stability_known": 1,
    }))
    df = load_json_gz(MATMINER / "wolverton_oxides.json.gz")
    rows.append(pd.DataFrame({
        "source_dataset": "wolverton_oxides",
        "formula": df["formula"].astype(str),
        "e_form": pd.to_numeric(df["e_form"], errors="coerce"),
        "e_hull": pd.to_numeric(df["e_hull"], errors="coerce"),
        "band_gap": pd.to_numeric(df["gap pbe"], errors="coerce"),
        "direct_gap": 0,
        "stability_known": 1,
    }))
    df = load_json_gz(MATMINER / "double_perovskites_gap.json.gz")
    rows.append(pd.DataFrame({
        "source_dataset": "double_perovskites_gap",
        "formula": df["formula"].astype(str),
        "e_form": np.nan,
        "e_hull": np.nan,
        "band_gap": pd.to_numeric(df["gap gllbsc"], errors="coerce"),
        "direct_gap": 0,
        "stability_known": 0,
    }))
    pool = pd.concat(rows, ignore_index=True)
    pool = pool[pool["band_gap"].notna()].reset_index(drop=True)
    pool["excluded_elements"] = pool["formula"].map(excluded_elements_in)
    pool["process_risk_elements"] = pool["formula"].map(process_risk_in)
    return pool


# ---------------------------------------------------------------------------
def pareto_front(ranked: pd.DataFrame) -> pd.DataFrame:
    eligible = ranked[ranked["stability_known"] == 1].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible = eligible.sort_values(["stability_utility", "gap_utility"], ascending=[False, False])
    g = eligible["gap_utility"].to_numpy(float)
    prev_max = np.concatenate(([-np.inf], np.maximum.accumulate(g)[:-1]))
    return eligible[g > prev_max].sort_values("candidate_score", ascending=False)


def brute_force_pareto(ranked: pd.DataFrame) -> pd.DataFrame:
    """Independent non-dominated set, used only to audit pareto_front()."""
    e = ranked[ranked["stability_known"] == 1]
    if e.empty:
        return e
    pts = e[["stability_utility", "gap_utility"]].drop_duplicates()
    s = pts["stability_utility"].to_numpy(float)
    g = pts["gap_utility"].to_numpy(float)
    keep = np.ones(len(pts), bool)
    for i in range(len(pts)):
        keep[i] = not (((s >= s[i]) & (g >= g[i]) & ((s > s[i]) | (g > g[i]))).any())
    nd = pts[keep]
    mask = e.set_index(["stability_utility", "gap_utility"]).index.isin(
        nd.set_index(["stability_utility", "gap_utility"]).index)
    return e[mask]


def candidate_pool_summary(ranked: pd.DataFrame) -> pd.DataFrame:
    return (
        ranked.groupby("source_dataset", as_index=False)
        .agg(
            candidates=("formula", "count"),
            stability_known=("stability_known", "sum"),
            mean_band_gap=("band_gap", "mean"),
            max_band_gap=("band_gap", "max"),
            mean_candidate_score=("candidate_score", "mean"),
            max_candidate_score=("candidate_score", "max"),
        )
        .sort_values("candidates", ascending=False)
    )


WEIGHT_SETS = {
    "balanced": {"stability": 0.45, "gap": 0.35, "direct": 0.20},
    "stability_first": {"stability": 0.70, "gap": 0.20, "direct": 0.10},
    "gap_first": {"stability": 0.25, "gap": 0.65, "direct": 0.10},
    "direct_gap_bonus": {"stability": 0.35, "gap": 0.35, "direct": 0.30},
    "risk_averse": {"stability": 0.80, "gap": 0.15, "direct": 0.05},
}


def objective_sensitivity(ranked: pd.DataFrame) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame()
    base_top = set(ranked.sort_values("candidate_score", ascending=False).head(50)["formula"])
    rows = []
    for name, w in WEIGHT_SETS.items():
        scored = ranked.copy()
        scored["sensitivity_score"] = (
            w["stability"] * scored["stability_utility"]
            + w["gap"] * scored["gap_utility"]
            + w["direct"] * scored["direct_gap_utility"]
        )
        scored = scored.sort_values("sensitivity_score", ascending=False)
        top = scored.iloc[0]
        rows.append({
            "objective_profile": name,
            "stability_weight": w["stability"], "gap_weight": w["gap"],
            "direct_gap_weight": w["direct"],
            "top_formula": top["formula"], "top_source_dataset": top["source_dataset"],
            "top_band_gap": top["band_gap"],
            "top_stability_utility": top["stability_utility"],
            "top_gap_utility": top["gap_utility"],
            "top_score": top["sensitivity_score"],
            "top50_overlap_with_balanced": len(base_top & set(scored.head(50)["formula"])),
        })
    return pd.DataFrame(rows)


def diverse_acquisition_batch(ranked: pd.DataFrame, batch_size: int = 20) -> pd.DataFrame:
    """Identical acquisition rule to inverse_design_baseline.diverse_acquisition_batch."""
    if ranked.empty:
        return pd.DataFrame()
    pool = ranked.sort_values("candidate_score", ascending=False).head(min(1500, len(ranked))).copy()
    feature_cols = ["stability_utility", "gap_utility", "direct_gap_utility", "candidate_score"]
    features = pool[feature_cols].fillna(0.0).to_numpy()
    # Vectorised but semantically identical to the published greedy loop:
    # acquisition = 0.65*utility + 0.35*min-distance-to-already-selected.
    # Ties are broken by lowest pool index (the published loop's tie-break is
    # CPython set-iteration order; exact ties occur only for duplicate rows).
    D = pairwise_distances(features)
    utility = pool["candidate_score"].to_numpy(float)
    selected = [0]
    mind = D[:, 0].copy()
    alive = np.ones(len(pool), bool)
    alive[0] = False
    while alive.any() and len(selected) < batch_size:
        acq = 0.65 * utility + 0.35 * mind
        acq[~alive] = -np.inf
        best_idx = int(np.argmax(acq))
        selected.append(best_idx)
        alive[best_idx] = False
        mind = np.minimum(mind, D[:, best_idx])
    batch = pool.iloc[selected].copy()
    batch.insert(0, "acquisition_rank", range(1, len(batch) + 1))
    return batch


# ---------------------------------------------------------------------------
def validation_gate(batch: pd.DataFrame) -> pd.DataFrame:
    """Corrected expert-review gate.

    Fixes two defects of candidate_validation_pack.py:
      * no merge back onto the full ranked table (that merge duplicated rows
        whose formula is not unique, inflating the published gate counts);
      * uses the harmonised exclusion list rather than the short legacy one.
    """
    rows = []
    for _, row in batch.iterrows():
        elems = elements_of(row["formula"])
        excluded = sorted(elems & EXCLUDED_ELEMENTS)
        process = sorted(elems & PROCESS_RISK_ELEMENTS)
        stability_known = int(row["stability_known"])
        su = float(row["stability_utility"])
        gu = float(row["gap_utility"])
        score = float(row["candidate_score"])
        if excluded:
            gate, why = "reject_or_special_handling", "contains excluded element(s): " + ", ".join(excluded)
        elif stability_known == 0:
            gate, why = "computational_validation_required", "stability metric imputed, not present in the source record"
        elif su < 0.60:
            gate, why = "low_priority", "known-stability utility below conservative review threshold"
        elif process:
            gate, why = "expert_review_required", "contains volatile/process-sensitive element(s): " + ", ".join(process)
        elif score >= 0.80 and gu >= 0.50:
            gate, why = "priority_expert_review", "high score with acceptable band-gap utility and no hazard flag"
        else:
            gate, why = "standard_review", "no hazard flag, but candidate needs expert assessment"
        rows.append({
            "formula": row["formula"], "source_dataset": row["source_dataset"],
            "band_gap": row["band_gap"], "stability_known": stability_known,
            "stability_utility": su, "gap_utility": gu, "candidate_score": score,
            "excluded_elements": " ".join(excluded), "process_risk_elements": " ".join(process),
            "validation_gate": gate, "review_rationale": why,
        })
    return pd.DataFrame(rows)


def gate_summary(review: pd.DataFrame) -> pd.DataFrame:
    return (
        review.groupby("validation_gate", as_index=False)
        .agg(
            candidates=("formula", "count"),
            mean_candidate_score=("candidate_score", "mean"),
            mean_stability_utility=("stability_utility", "mean"),
            mean_gap_utility=("gap_utility", "mean"),
        )
        .sort_values("candidates", ascending=False)
    )
