
from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
HIGH_RISK_ELEMENTS = {"Pb", "Cd", "Hg", "As", "Tl", "Be", "Po", "Ra", "Th", "U", "Pu"}
VOLATILE_OR_PROCESS_RISK = {"F", "Cl", "Br", "I", "Se", "Te", "S"}
COMMON_CERAMIC_ELEMENTS = {
    "O",
    "Ba",
    "Sr",
    "Ca",
    "Ti",
    "Zr",
    "Hf",
    "Nb",
    "Ta",
    "Mg",
    "Zn",
    "Al",
    "Si",
    "Y",
    "La",
    "Ce",
    "Bi",
    "Na",
    "K",
    "Li",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "W",
    "Mo",
    "Sn",
}

def elements(formula: str) -> set[str]:
    return set(ELEMENT_RE.findall(str(formula)))

def classify_candidate(row: pd.Series) -> dict[str, object]:
    elems = elements(row.get("formula", ""))
    high_risk = sorted(elems & HIGH_RISK_ELEMENTS)
    process_risk = sorted(elems & VOLATILE_OR_PROCESS_RISK)
    unusual = sorted(elems - COMMON_CERAMIC_ELEMENTS)
    stability_known = int(row.get("stability_known", 0)) if "stability_known" in row else 0
    stability_utility = float(row.get("stability_utility", 0.0))
    gap_utility = float(row.get("gap_utility", 0.0))
    score = float(row.get("candidate_score", 0.0))

    if high_risk:
        gate = "reject_or_special_handling"
        rationale = "contains high-risk element(s): " + ", ".join(high_risk)
    elif stability_known == 0:
        gate = "computational_validation_required"
        rationale = "stability metric is not available in the source record"
    elif stability_utility < 0.60:
        gate = "low_priority"
        rationale = "known-stability utility is below conservative review threshold"
    elif process_risk:
        gate = "expert_review_required"
        rationale = "contains volatile/process-sensitive element(s): " + ", ".join(process_risk)
    elif unusual:
        gate = "expert_review_required"
        rationale = "contains less common ceramic screening element(s): " + ", ".join(unusual)
    elif score >= 0.80 and gap_utility >= 0.50:
        gate = "priority_expert_review"
        rationale = "high score with acceptable band-gap utility and no simple hazard flag"
    else:
        gate = "standard_review"
        rationale = "no simple hazard flag, but candidate needs expert assessment"

    return {
        "formula": row.get("formula", ""),
        "source_dataset": row.get("source_dataset", ""),
        "stability_metric": row.get("stability_metric", ""),
        "band_gap": row.get("band_gap", ""),
        "stability_known": stability_known,
        "stability_utility": stability_utility,
        "gap_utility": gap_utility,
        "candidate_score": score,
        "elements": " ".join(sorted(elems)),
        "high_risk_elements": " ".join(high_risk),
        "process_risk_elements": " ".join(process_risk),
        "unusual_elements": " ".join(unusual),
        "validation_gate": gate,
        "review_rationale": rationale,
    }

def load_candidates(results_dir: Path) -> pd.DataFrame:
    acquisition = results_dir / "ceramic_diverse_acquisition_batch.csv"
    ranked = results_dir / "ceramic_ranked_perovskite_candidates.csv"
    if acquisition.exists():
        candidates = pd.read_csv(acquisition)
        if ranked.exists():
            ranked_df = pd.read_csv(ranked)
            candidates = candidates.merge(
                ranked_df[["formula", "direct_gap", "stability_known", "direct_gap_utility"]],
                on="formula",
                how="left",
            )
        return candidates
    if ranked.exists():
        return pd.read_csv(ranked).head(50)
    raise FileNotFoundError("Run inverse_design_baseline.py before creating the validation pack.")

def save_figure(review: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    order = [
        "priority_expert_review",
        "standard_review",
        "expert_review_required",
        "computational_validation_required",
        "low_priority",
        "reject_or_special_handling",
    ]
    counts = review["validation_gate"].value_counts().reindex(order).dropna().reset_index()
    counts.columns = ["validation_gate", "count"]
    plt.figure(figsize=(7.4, 4.4))
    sns.barplot(data=counts, x="count", y="validation_gate", color="#2F5D8C")
    plt.xlabel("Candidate count")
    plt.ylabel("Validation gate")
    plt.title("Expert-Review Gate Summary for Inverse-Design Candidates")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def write_markdown(review: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    priority = review[review["validation_gate"] == "priority_expert_review"].head(10)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Expert Review Pack for Ceramic Candidates\n\n")
        handle.write("This file is a computational pre-screen for domain-expert review. ")
        handle.write("It is not experimental validation and should not be cited as proof of synthesizability.\n\n")
        handle.write("## Gate Summary\n\n")
        handle.write(summary.to_markdown(index=False))
        handle.write("\n\n")
        handle.write("## Priority Candidates\n\n")
        if priority.empty:
            handle.write("No candidate passed the priority gate under the conservative thresholds.\n")
        else:
            cols = ["formula", "source_dataset", "band_gap", "stability_utility", "gap_utility", "candidate_score", "review_rationale"]
            handle.write(priority[cols].to_markdown(index=False))
            handle.write("\n")
        handle.write("\n## Required Expert Checks\n\n")
        handle.write("- Confirm phase plausibility and charge/valence balance.\n")
        handle.write("- Confirm synthesis route and temperature feasibility.\n")
        handle.write("- Check toxicity, volatility, and handling constraints.\n")
        handle.write("- Verify source-dataset stability meaning before ranking candidates experimentally.\n")
        handle.write("- Use DFT, uMLIP, Materials Project lookup, or laboratory synthesis before making discovery claims.\n")

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "data" / "processed_results"
    figures_dir = root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(results_dir)
    review = pd.DataFrame([classify_candidate(row) for _, row in candidates.iterrows()])
    summary = (
        review.groupby("validation_gate", as_index=False)
        .agg(
            candidates=("formula", "count"),
            mean_candidate_score=("candidate_score", "mean"),
            mean_stability_utility=("stability_utility", "mean"),
            mean_gap_utility=("gap_utility", "mean"),
        )
        .sort_values("candidates", ascending=False)
    )
    review.to_csv(results_dir / "ceramic_expert_review_pack.csv", index=False)
    summary.to_csv(results_dir / "ceramic_validation_gate_summary.csv", index=False)
    save_figure(review, figures_dir / "ceramic_expert_review_gate_origin_style.png")
    write_markdown(review, summary, results_dir / "ceramic_expert_review_pack.md")
    print("Expert-review pack complete.")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()