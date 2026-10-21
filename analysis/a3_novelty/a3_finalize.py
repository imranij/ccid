"""
a3_finalize.py -- assemble deliverables: per-candidate CSV, summary CSV/JSON,
LaTeX (booktabs) tables, from the intermediate a3_stage_nn.csv / a3_families.csv.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(_CCID_ROOT + "/results")

nn = pd.read_csv(RESULTS / "a3_stage_nn.csv")
fam = pd.read_csv(RESULTS / "a3_families.csv")[["canonical_formula", "family_id", "family_key"]]

df = nn.merge(fam, on="canonical_formula", how="left")

# External-database columns: honestly marked "not_verifiable" for every provider
# we could not successfully query (see a3_REPORT.md, Task 1, for the full
# per-provider connectivity log). No fabricated hits are recorded.
for col in [
    "found_materials_project_optimade",
    "found_oqmd",
    "found_aflow",
    "found_cod",
    "found_alexandria",
]:
    df[col] = "not_verifiable"
df["external_db_entry_id"] = ""
df["external_db_check_note"] = (
    "OPTIMADE cross-check could not be completed for this candidate: every provider "
    "tried was either blocked by the sandbox egress policy (direct HTTPS), blocked by "
    "robots.txt (OQMD, COD) or returned a provider-side error on any filtered query "
    "(AFLOW /structures, Alexandria /structures) even though the same providers' /info "
    "endpoints were reachable. See a3_REPORT.md Task 1 for the full connectivity log."
)

df["novelty_level_i_absent_from_training_exact_match"] = df["absent_from_training_exact"]
df["novelty_level_ii_absent_from_external_db"] = "not_verifiable"
df["novelty_level_iii_no_entry_in_chemical_system"] = ~df["same_chem_system_in_training"]

cols = [
    "canonical_formula", "candidate_score", "novel",
    "novelty_level_i_absent_from_training_exact_match",
    "novelty_level_ii_absent_from_external_db",
    "novelty_level_iii_no_entry_in_chemical_system",
    "same_chem_system_in_training",
    "elmd_nn_distance", "elmd_nn_train_formula",
    "euclid_nn_distance", "euclid_nn_train_formula",
    "family_id", "family_key",
    "found_materials_project_optimade", "found_oqmd", "found_aflow", "found_cod",
    "found_alexandria", "external_db_entry_id", "external_db_check_note",
]
out = df[cols].sort_values("candidate_score", ascending=False).reset_index(drop=True)
out.to_csv(RESULTS / "a3_per_candidate_novelty.csv", index=False)
print(f"wrote a3_per_candidate_novelty.csv ({len(out)} rows)")

# ---------------- summary ----------------
n = len(out)
n_novel_paper = int(out["novel"].sum())
n_level_i = int(out["novelty_level_i_absent_from_training_exact_match"].sum())
n_level_iii_novel_system = int(out["novelty_level_iii_no_entry_in_chemical_system"].sum())

summary = {
    "n_valid_rows_20000_sample_archive": 6768,
    "n_unique_valid_formulas": n,
    "n_flagged_novel_by_papers_exact_match_test": n_novel_paper,
    "novelty_rate_papers_definition_pct": round(100 * n_novel_paper / n, 2),
    "n_absent_from_training_by_chemical_system": n_level_iii_novel_system,
    "chemical_system_novelty_rate_pct": round(100 * n_level_iii_novel_system / n, 2),
    "external_database_cross_check": "NOT COMPLETED -- see a3_REPORT.md Task 1 for the full "
                                      "per-provider connectivity log. No provider could be "
                                      "queried for candidate-specific matches from this "
                                      "sandbox; treat level-(ii) and level-(iii)-across-all-"
                                      "external-DBs novelty as unverified, not as confirmed.",
    "elmd_nearest_neighbour_distance_magpie": {
        "min": float(out["elmd_nn_distance"].min()),
        "p25": float(out["elmd_nn_distance"].quantile(0.25)),
        "median": float(out["elmd_nn_distance"].median()),
        "p75": float(out["elmd_nn_distance"].quantile(0.75)),
        "max": float(out["elmd_nn_distance"].max()),
        "mean": float(out["elmd_nn_distance"].mean()),
        "n_exact_zero_distance": int((out["elmd_nn_distance"] == 0).sum()),
    },
    "euclid_nearest_neighbour_distance_element_fraction_space": {
        "min": float(out["euclid_nn_distance"].min()),
        "p25": float(out["euclid_nn_distance"].quantile(0.25)),
        "median": float(out["euclid_nn_distance"].median()),
        "p75": float(out["euclid_nn_distance"].quantile(0.75)),
        "max": float(out["euclid_nn_distance"].max()),
        "mean": float(out["euclid_nn_distance"].mean()),
    },
    "n_distinct_substitution_families": int(fam["family_id"].nunique()),
    "n_unique_formulas_for_family_count": len(fam),
}
json.dump(summary, open(RESULTS / "a3_summary.json", "w"), indent=2)
pd.DataFrame([summary]).to_csv(RESULTS / "a3_summary_flat.csv", index=False)
print(json.dumps(summary, indent=2))

# ---------------- top-10 table for LaTeX ----------------
top10 = out.head(10).copy()

def esc(s):
    return str(s).replace("_", r"\_")

lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Top-10 Stage-2 candidates by \texttt{candidate\_score}: exact-match novelty "
              r"(the paper's definition) versus nearest-neighbour distance to the training set and "
              r"substitution-family membership. External-database status is not verifiable from this "
              r"sandbox (see text) and is omitted from ranking.}")
lines.append(r"\label{tab:a3_top10}")
lines.append(r"\begin{tabular}{lccccc}")
lines.append(r"\toprule")
lines.append(r"Formula & Score & Novel (exact) & ElMD NN dist. & Nearest training formula & Family \\")
lines.append(r"\midrule")
for _, r in top10.iterrows():
    lines.append(
        f"{esc(r['canonical_formula'])} & {r['candidate_score']:.3f} & "
        f"{'Yes' if r['novel'] else 'No'} & {r['elmd_nn_distance']:.2f} & "
        f"{esc(r['elmd_nn_train_formula'])} & {esc(r['family_id'])} \\\\"
    )
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
(RESULTS / "a3_table_top10.tex").write_text("\n".join(lines) + "\n")

# ---------------- family summary table ----------------
fam_sizes = fam.groupby("family_id")["canonical_formula"].apply(list).sort_values(key=lambda s: -s.str.len())
lines2 = []
lines2.append(r"\begin{table}[htbp]")
lines2.append(r"\centering")
lines2.append(r"\caption{Substitution families among the 66 unique valid Stage-2 candidates. "
              r"A family shares the same anonymised stoichiometry (atom-count pattern including "
              r"oxygen) and the same multiset of periodic-table groups on the cation sites; only "
              r"families with $\geq 2$ members are listed individually, singleton families are "
              r"aggregated in the last row.}")
lines2.append(r"\label{tab:a3_families}")
lines2.append(r"\begin{tabular}{clc}")
lines2.append(r"\toprule")
lines2.append(r"Family & Member formulas & Size \\")
lines2.append(r"\midrule")
n_singletons = 0
for fid, members in fam_sizes.items():
    if len(members) == 1:
        n_singletons += 1
        continue
    lines2.append(f"{esc(fid)} & {', '.join(esc(m) for m in members)} & {len(members)} \\\\")
lines2.append(f"(singletons) & {n_singletons} distinct one-member families & {n_singletons} \\\\")
lines2.append(r"\bottomrule")
lines2.append(r"\end{tabular}")
lines2.append(r"\end{table}")
(RESULTS / "a3_table_families.tex").write_text("\n".join(lines2) + "\n")

# ---------------- novelty definitions summary table ----------------
lines3 = []
lines3.append(r"\begin{table}[htbp]")
lines3.append(r"\centering")
lines3.append(r"\caption{Novelty of the 66 unique valid Stage-2 candidates under progressively "
              r"stricter definitions.}")
lines3.append(r"\label{tab:a3_novelty_levels}")
lines3.append(r"\begin{tabular}{lcc}")
lines3.append(r"\toprule")
lines3.append(r"Novelty definition & N (of 66) & Rate \\")
lines3.append(r"\midrule")
lines3.append(f"(i) Absent by exact formula match, training set only & {n_level_i} & {100*n_level_i/n:.1f}\\% \\\\")
lines3.append(r"(ii) Absent from external databases queried & \multicolumn{2}{c}{not verifiable -- see Task 1} \\")
lines3.append(f"(iii) No entry in the same chemical system (element set), training set only & {n_level_iii_novel_system} & {100*n_level_iii_novel_system/n:.1f}\\% \\\\")
lines3.append(r"\bottomrule")
lines3.append(r"\end{tabular}")
lines3.append(r"\end{table}")
(RESULTS / "a3_table_novelty_levels.tex").write_text("\n".join(lines3) + "\n")

print("wrote a3_table_top10.tex, a3_table_families.tex, a3_table_novelty_levels.tex")
