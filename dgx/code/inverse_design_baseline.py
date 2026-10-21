
from __future__ import annotations

import gzip
import json
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

DATA_URL = "https://ndownloader.figshare.com/files/13213475"

def ensure_dataset() -> Path:
    path = Path(__file__).resolve().parents[2] / "Paper_1_Uncertainty_Calibrated_Data_Repair_for_Ceramic_Materials" / "data" / "dielectric_constant.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(DATA_URL, path)
    return path

def load_candidates() -> pd.DataFrame:
    with gzip.open(ensure_dataset(), "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    df = pd.DataFrame(raw["data"], columns=raw["columns"], index=raw["index"])
    cols = ["material_id", "formula", "nsites", "space_group", "volume", "band_gap", "n", "poly_electronic", "poly_total", "pot_ferroelectric"]
    df = df[cols].copy()
    df["pot_ferroelectric"] = df["pot_ferroelectric"].astype(int)
    numeric = ["nsites", "space_group", "volume", "band_gap", "n", "poly_electronic", "poly_total", "pot_ferroelectric"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df.dropna()

def load_json_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict) and {"data", "columns", "index"} <= set(raw):
        return pd.DataFrame(raw["data"], columns=raw["columns"], index=raw["index"])
    return pd.DataFrame(raw)

def safe_scaled(series: pd.Series, invert: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if invert:
        values = -values
    if values.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index)
    return pd.Series(MinMaxScaler().fit_transform(values.to_numpy().reshape(-1, 1)).reshape(-1), index=series.index)

def score_candidates(df: pd.DataFrame) -> pd.DataFrame:

    df = df[df["band_gap"] >= 1.0].copy()
    objectives = ["n", "poly_total", "band_gap"]
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(df[objectives]), columns=objectives, index=df.index)

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

def score_perovskite_candidates(root: Path) -> pd.DataFrame:
    shared = root.parent / "shared_datasets" / "matminer"
    rows: list[pd.DataFrame] = []

    castelli_path = shared / "castelli_perovskites.json.gz"
    if castelli_path.exists():
        df = load_json_gz(castelli_path)
        out = pd.DataFrame(
            {
                "source_dataset": "castelli_perovskites",
                "formula": df["formula"],
                "stability_metric": pd.to_numeric(df["e_form"], errors="coerce"),
                "band_gap": pd.to_numeric(df["gap gllbsc"], errors="coerce"),
                "direct_gap": df["gap is direct"].astype(int),
                "stability_known": 1,
            }
        ).dropna()
        out["stability_utility"] = safe_scaled(out["stability_metric"], invert=True)
        out["gap_utility"] = safe_scaled(out["band_gap"])
        out["direct_gap_utility"] = out["direct_gap"].astype(float)
        out["candidate_score"] = (
            0.45 * out["stability_utility"]
            + 0.35 * out["gap_utility"]
            + 0.20 * out["direct_gap_utility"]
        )
        rows.append(out)

    wolverton_path = shared / "wolverton_oxides.json.gz"
    if wolverton_path.exists():
        df = load_json_gz(wolverton_path)
        out = pd.DataFrame(
            {
                "source_dataset": "wolverton_oxides",
                "formula": df["formula"],
                "stability_metric": pd.to_numeric(df["e_hull"], errors="coerce"),
                "band_gap": pd.to_numeric(df["gap pbe"], errors="coerce"),
                "direct_gap": 0,
                "stability_known": 1,
            }
        ).dropna()
        out["stability_utility"] = safe_scaled(out["stability_metric"], invert=True)
        out["gap_utility"] = safe_scaled(out["band_gap"])
        out["direct_gap_utility"] = 0.0
        out["candidate_score"] = 0.65 * out["stability_utility"] + 0.35 * out["gap_utility"]
        rows.append(out)

    double_path = shared / "double_perovskites_gap.json.gz"
    if double_path.exists():
        df = load_json_gz(double_path)
        out = pd.DataFrame(
            {
                "source_dataset": "double_perovskites_gap",
                "formula": df["formula"],
                "stability_metric": 0.0,
                "band_gap": pd.to_numeric(df["gap gllbsc"], errors="coerce"),
                "direct_gap": 0,
                "stability_known": 0,
            }
        ).dropna()
        out["stability_utility"] = 0.5
        out["gap_utility"] = safe_scaled(out["band_gap"])
        out["direct_gap_utility"] = 0.0
        out["candidate_score"] = out["gap_utility"]
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    ranked = pd.concat(rows, ignore_index=True).sort_values("candidate_score", ascending=False)
    return ranked

def pareto_front(ranked: pd.DataFrame) -> pd.DataFrame:

    eligible = ranked[ranked["stability_known"] == 1].copy()
    if eligible.empty:
        return pd.DataFrame()
    eligible = eligible.sort_values(["stability_utility", "gap_utility"], ascending=[False, False])
    best_gap = -np.inf
    pareto_rows = []
    for _, row in eligible.iterrows():
        gap = float(row["gap_utility"])
        if gap > best_gap:
            pareto_rows.append(row)
            best_gap = gap
    front = pd.DataFrame(pareto_rows)
    return front.sort_values("candidate_score", ascending=False)

def candidate_pool_summary(ranked: pd.DataFrame) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame()
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

def objective_sensitivity(ranked: pd.DataFrame) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame()
    weight_sets = {
        "balanced": {"stability": 0.45, "gap": 0.35, "direct": 0.20},
        "stability_first": {"stability": 0.70, "gap": 0.20, "direct": 0.10},
        "gap_first": {"stability": 0.25, "gap": 0.65, "direct": 0.10},
        "direct_gap_bonus": {"stability": 0.35, "gap": 0.35, "direct": 0.30},
        "risk_averse": {"stability": 0.80, "gap": 0.15, "direct": 0.05},
    }
    base_top = set(ranked.sort_values("candidate_score", ascending=False).head(50)["formula"])
    rows: list[dict[str, object]] = []
    for name, weights in weight_sets.items():
        scored = ranked.copy()
        scored["sensitivity_score"] = (
            weights["stability"] * scored["stability_utility"]
            + weights["gap"] * scored["gap_utility"]
            + weights["direct"] * scored["direct_gap_utility"]
        )
        scored = scored.sort_values("sensitivity_score", ascending=False)
        top50 = set(scored.head(50)["formula"])
        top = scored.iloc[0]
        rows.append(
            {
                "objective_profile": name,
                "stability_weight": weights["stability"],
                "gap_weight": weights["gap"],
                "direct_gap_weight": weights["direct"],
                "top_formula": top["formula"],
                "top_source_dataset": top["source_dataset"],
                "top_band_gap": top["band_gap"],
                "top_stability_utility": top["stability_utility"],
                "top_gap_utility": top["gap_utility"],
                "top_score": top["sensitivity_score"],
                "top50_overlap_with_balanced": len(base_top & top50),
            }
        )
    return pd.DataFrame(rows)

def diverse_acquisition_batch(ranked: pd.DataFrame, batch_size: int = 20) -> pd.DataFrame:

    if ranked.empty:
        return pd.DataFrame()
    pool = ranked.sort_values("candidate_score", ascending=False).head(min(1500, len(ranked))).copy()
    feature_cols = ["stability_utility", "gap_utility", "direct_gap_utility", "candidate_score"]
    features = pool[feature_cols].fillna(0.0).to_numpy()
    selected: list[int] = [0]
    remaining = set(range(1, len(pool)))
    while remaining and len(selected) < batch_size:
        selected_features = features[selected]
        best_idx = None
        best_value = -np.inf
        for idx in remaining:
            diversity = pairwise_distances(features[idx : idx + 1], selected_features).min()
            utility = float(pool.iloc[idx]["candidate_score"])
            acquisition = 0.65 * utility + 0.35 * diversity
            if acquisition > best_value:
                best_value = acquisition
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    batch = pool.iloc[selected].copy()
    batch["acquisition_rank"] = range(1, len(batch) + 1)
    return batch[
        [
            "acquisition_rank",
            "source_dataset",
            "formula",
            "stability_metric",
            "band_gap",
            "stability_utility",
            "gap_utility",
            "candidate_score",
        ]
    ]

def save_figures(ranked: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    plt.figure(figsize=(7.2, 4.6))
    scatter = plt.scatter(
        ranked["band_gap"],
        ranked["poly_total"],
        c=ranked["inverse_design_score"],
        cmap="viridis",
        s=36,
        edgecolor="black",
        linewidth=0.25,
    )
    plt.colorbar(scatter, label="Inverse-design score")
    plt.xlabel("Band gap (eV)")
    plt.ylabel("Total dielectric constant")
    plt.title("Real Candidate Ranking for Ceramic Inverse Design")
    plt.tight_layout()
    plt.savefig(figures_dir / "ceramic_inverse_design_candidate_map_origin_style.png", dpi=300)
    plt.close()

    top = ranked.head(15)
    plt.figure(figsize=(7.2, 4.6))
    sns.barplot(data=top, x="inverse_design_score", y="formula", color="#1B4D89")
    plt.xlabel("Inverse-design score")
    plt.ylabel("Formula")
    plt.title("Top Real Candidates Before Generative Expansion")
    plt.tight_layout()
    plt.savefig(figures_dir / "ceramic_top_candidates_origin_style.png", dpi=300)
    plt.close()

def save_perovskite_figures(ranked: pd.DataFrame, figures_dir: Path) -> None:
    if ranked.empty:
        return
    top = ranked.head(25).copy()
    plt.figure(figsize=(7.6, 5.0))
    sns.scatterplot(
        data=ranked.sample(min(len(ranked), 5000), random_state=2026),
        x="band_gap",
        y="candidate_score",
        hue="source_dataset",
        s=22,
        linewidth=0,
        alpha=0.75,
    )
    plt.xlabel("Band gap (eV)")
    plt.ylabel("Perovskite candidate score")
    plt.title("Perovskite/Oxide Candidate Space from Real Open Datasets")
    plt.tight_layout()
    plt.savefig(figures_dir / "ceramic_perovskite_candidate_space_origin_style.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.6, 5.2))
    sns.barplot(data=top, x="candidate_score", y="formula", hue="source_dataset", dodge=False)
    plt.xlabel("Candidate score")
    plt.ylabel("Formula")
    plt.title("Top Perovskite/Oxide Candidates from Real Datasets")
    plt.tight_layout()
    plt.savefig(figures_dir / "ceramic_top_perovskite_candidates_origin_style.png", dpi=300)
    plt.close()

def save_pareto_figures(ranked: pd.DataFrame, pareto: pd.DataFrame, figures_dir: Path) -> None:
    if ranked.empty or pareto.empty:
        return
    known = ranked[ranked["stability_known"] == 1].copy()
    plt.figure(figsize=(7.6, 5.0))
    sns.scatterplot(
        data=known.sample(min(len(known), 6000), random_state=2026),
        x="stability_utility",
        y="gap_utility",
        hue="source_dataset",
        s=24,
        linewidth=0,
        alpha=0.55,
    )
    plt.scatter(
        pareto["stability_utility"],
        pareto["gap_utility"],
        s=76,
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        label="Pareto front",
    )
    plt.xlabel("Normalized stability utility")
    plt.ylabel("Normalized band-gap utility")
    plt.title("Pareto Front for Known-Stability Perovskite/Oxide Candidates")
    plt.legend(fontsize=7, title_fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "ceramic_pareto_front_origin_style.png", dpi=300)
    plt.close()

    top = pareto.head(20).copy()
    plt.figure(figsize=(7.6, 5.2))
    sns.barplot(data=top, x="candidate_score", y="formula", hue="source_dataset", dodge=False)
    plt.xlabel("Candidate score")
    plt.ylabel("Pareto-front formula")
    plt.title("Top Pareto-Front Candidates")
    plt.tight_layout()
    plt.savefig(figures_dir / "ceramic_top_pareto_candidates_origin_style.png", dpi=300)
    plt.close()

def save_acquisition_figures(
    ranked: pd.DataFrame,
    sensitivity: pd.DataFrame,
    acquisition: pd.DataFrame,
    summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    if ranked.empty:
        return
    if not sensitivity.empty:
        plt.figure(figsize=(7.6, 4.6))
        sns.barplot(data=sensitivity, x="top_score", y="objective_profile", hue="top_source_dataset", dodge=False)
        plt.xlabel("Top candidate score under objective profile")
        plt.ylabel("Objective profile")
        plt.title("Objective Sensitivity of Inverse-Design Ranking")
        plt.tight_layout()
        plt.savefig(figures_dir / "ceramic_objective_sensitivity_origin_style.png", dpi=300)
        plt.close()

    if not acquisition.empty:
        plt.figure(figsize=(7.6, 5.0))
        sns.scatterplot(
            data=ranked.sample(min(len(ranked), 6000), random_state=2026),
            x="stability_utility",
            y="gap_utility",
            hue="source_dataset",
            s=22,
            linewidth=0,
            alpha=0.45,
        )
        plt.scatter(
            acquisition["stability_utility"],
            acquisition["gap_utility"],
            s=88,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label="Diverse acquisition batch",
        )
        plt.xlabel("Normalized stability utility")
        plt.ylabel("Normalized band-gap utility")
        plt.title("Diverse Next-Review Batch from Real Candidate Pool")
        plt.legend(fontsize=7, title_fontsize=8)
        plt.tight_layout()
        plt.savefig(figures_dir / "ceramic_diverse_acquisition_batch_origin_style.png", dpi=300)
        plt.close()

    if not summary.empty:
        plt.figure(figsize=(7.4, 4.4))
        sns.barplot(data=summary, x="candidates", y="source_dataset", color="#1B4D89")
        plt.xlabel("Real candidates in pool")
        plt.ylabel("Source dataset")
        plt.title("Candidate Pool Composition")
        plt.tight_layout()
        plt.savefig(figures_dir / "ceramic_candidate_pool_summary_origin_style.png", dpi=300)
        plt.close()

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "data" / "processed_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    ranked = score_candidates(candidates)
    ranked.to_csv(results_dir / "ceramic_ranked_real_candidates.csv", index=False)
    perovskite_ranked = score_perovskite_candidates(root)
    if not perovskite_ranked.empty:
        perovskite_ranked.to_csv(results_dir / "ceramic_ranked_perovskite_candidates.csv", index=False)
        pareto = pareto_front(perovskite_ranked)
        pareto.to_csv(results_dir / "ceramic_pareto_perovskite_candidates.csv", index=False)
        pareto_summary = (
            pareto.groupby("source_dataset", as_index=False)
            .agg(
                pareto_count=("formula", "count"),
                mean_candidate_score=("candidate_score", "mean"),
                max_candidate_score=("candidate_score", "max"),
            )
            .sort_values("pareto_count", ascending=False)
        )
        pareto_summary.to_csv(results_dir / "ceramic_pareto_summary.csv", index=False)
        pool_summary = candidate_pool_summary(perovskite_ranked)
        pool_summary.to_csv(results_dir / "ceramic_candidate_pool_summary.csv", index=False)
        sensitivity = objective_sensitivity(perovskite_ranked)
        sensitivity.to_csv(results_dir / "ceramic_objective_sensitivity.csv", index=False)
        acquisition = diverse_acquisition_batch(perovskite_ranked, batch_size=20)
        acquisition.to_csv(results_dir / "ceramic_diverse_acquisition_batch.csv", index=False)
    else:
        pareto = pd.DataFrame()
        pool_summary = pd.DataFrame()
        sensitivity = pd.DataFrame()
        acquisition = pd.DataFrame()
    save_figures(ranked, root / "figures")
    save_perovskite_figures(perovskite_ranked, root / "figures")
    save_pareto_figures(perovskite_ranked, pareto, root / "figures")
    save_acquisition_figures(perovskite_ranked, sensitivity, acquisition, pool_summary, root / "figures")
    print(f"Candidates ranked: {len(ranked)}")
    print(f"Perovskite/oxide candidates ranked: {len(perovskite_ranked)}")
    print(f"Known-stability Pareto candidates: {len(pareto)}")
    print(ranked[["material_id", "formula", "inverse_design_score"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()