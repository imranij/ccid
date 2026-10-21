"""a7_compare.py -- the quality comparison: CVAE candidates vs random-baseline candidates.

Inputs (all produced earlier in this agent's run):
  a7_surrogate_predictions.csv   a5 surrogates applied to both populations
  a7_chemical_sanity.csv         oxidation-state / open-shell / distance-to-training
Outputs:
  a7_compare_continuous.csv      every continuous-axis comparison with effect sizes
  a7_compare_proportions.csv     every proportion comparison
  a7_steelman.csv                the deliberately CVAE-favourable tests
  a7_compare_summary.json
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import json, sys
import numpy as np, pandas as pd
from pymatgen.core import Composition

sys.path.insert(0, _CCID_ROOT + "/analysis/a7_quality")
from a7_stats import compare_continuous, compare_proportions

RESULTS = _CCID_ROOT + "/results"

# Chemistries the Stage-1 real-data screen independently favoured (audit / a1):
STAGE1_GOOD = {"Hf", "Zr", "Al"}

CONT_AXES = [
    ("pred_band_gap_eV_mean", "predicted band gap (eV)", "higher is better"),
    ("pred_poly_total_mean", "predicted eps_0 (total)", "higher is better"),
    ("pred_poly_electronic_mean", "predicted eps_inf (electronic)", "higher is better"),
    ("pred_stability_wolverton_oxides_mean", "predicted E_hull-like (eV/atom)", "lower is better"),
    ("pred_stability_castelli_perovskites_mean", "predicted E_form-like (eV/atom)", "lower is better"),
    ("predicted_property_rank_score", "composite predicted-property score", "higher is better"),
]


def load():
    s = pd.read_csv(f"{RESULTS}/a7_surrogate_predictions.csv")
    c = pd.read_csv(f"{RESULTS}/a7_chemical_sanity.csv")
    c["population"] = c["population"].replace({"random_uniform41": "random", "random_freq41": "random_freq", "random_dirichlet41": "random_dirichlet"})
    c = c.drop_duplicates(subset=["population", "canonical_formula"])
    d = s.merge(c[["population", "canonical_formula", "requires_reduced_cation",
                   "has_open_shell", "n_open_shell", "n_forced_below_max",
                   "open_shell_cations", "forced_below_max_state",
                   "nn_train_distance", "nn_train_formula", "in_training_set",
                   "balance_assignable", "n_cations"]],
                on=["population", "canonical_formula"], how="left")
    # chemistry annotations
    def cats(f):
        return {str(e) for e in Composition(f).elements if str(e) != "O"}
    d["cation_set"] = d["canonical_formula"].map(cats)
    d["has_stage1_good"] = d["cation_set"].map(lambda s: bool(s & STAGE1_GOOD))
    d["n_stage1_good"] = d["cation_set"].map(lambda s: len(s & STAGE1_GOOD))
    d["all_stage1_good"] = d["cation_set"].map(lambda s: s <= STAGE1_GOOD)
    # stoichiometry family: n_cation_atoms : n_O atoms, reduced
    def fam(f):
        c = Composition(f).reduced_composition
        no = int(round(c["O"])); nc = int(round(c.num_atoms)) - no
        return f"{nc}:{no}"
    d["stoich_family"] = d["canonical_formula"].map(fam)
    # combined wide-gap-dielectric sanity gate
    d["passes_quality_gate"] = ((d["pred_band_gap_eV_mean"] >= 1.0) &
                                (~d["has_open_shell"].fillna(True)) &
                                (~d["requires_reduced_cation"].fillna(True)))
    return d


def run_block(d, a_name, b_name, tag, rows_c, rows_p):
    A = d[d["population"] == a_name]
    B = d[d["population"] == b_name]
    for col, label, direction in CONT_AXES:
        r = compare_continuous(A[col], B[col], a_name, b_name)
        r.update(comparison=tag, axis=col, axis_label=label, direction=direction)
        rows_c.append(r)
    r = compare_continuous(A["nn_train_distance"], B["nn_train_distance"], a_name, b_name)
    r.update(comparison=tag, axis="nn_train_distance",
             axis_label="Euclidean distance to nearest training composition",
             direction="higher = less like the training set")
    rows_c.append(r)
    for col, label in [("requires_reduced_cation", "requires Ti(3+)/Sn(2+) to charge balance"),
                       ("has_open_shell", "contains an open-shell d/f cation"),
                       ("passes_quality_gate", "passes combined wide-gap-dielectric gate"),
                       ("has_stage1_good", "contains Hf, Zr or Al (Stage-1 favoured)"),
                       ("all_stage1_good", "all cations in {Hf, Zr, Al}")]:
        aa = A[col].dropna().astype(bool); bb = B[col].dropna().astype(bool)
        r = compare_proportions(int(aa.sum()), len(aa), int(bb.sum()), len(bb), a_name, b_name)
        r.update(comparison=tag, axis=col, axis_label=label)
        rows_p.append(r)


def steelman(d):
    """Tests deliberately constructed to give the CVAE its best chance."""
    out = []
    A = d[d["population"] == "cvae"]
    B = d[d["population"] == "random"]

    # 1. top-N of each population by its OWN predicted composite score
    for N in (5, 10, 20, 66):
        a = A.nlargest(N, "predicted_property_rank_score")
        b = B.nlargest(N, "predicted_property_rank_score")
        for col, label, _ in CONT_AXES:
            r = compare_continuous(a[col], b[col], f"cvae_top{N}", f"random_top{N}")
            r.update(test=f"top-{N} of each population by composite score",
                     axis=col, axis_label=label)
            out.append(r)

    # 2. conditioned on chemical family (stoichiometry class), CVAE vs random
    for famname, g in d.groupby("stoich_family"):
        a = g[g["population"] == "cvae"]; b = g[g["population"] == "random"]
        if len(a) < 5 or len(b) < 5:
            continue
        for col in ("pred_band_gap_eV_mean", "predicted_property_rank_score"):
            r = compare_continuous(a[col], b[col], "cvae", "random")
            r.update(test=f"within stoichiometry family {famname}", axis=col,
                     axis_label=dict(CONT_AXES and [(c, l) for c, l, _ in CONT_AXES])[col])
            out.append(r)

    # 3. conditioned on containing a Stage-1-favoured cation
    for flag in (True, False):
        a = A[A["has_stage1_good"] == flag]; b = B[B["has_stage1_good"] == flag]
        if len(a) < 5 or len(b) < 5:
            continue
        for col, label, _ in CONT_AXES:
            r = compare_continuous(a[col], b[col], "cvae", "random")
            r.update(test=f"restricted to candidates {'with' if flag else 'without'} Hf/Zr/Al",
                     axis=col, axis_label=label)
            out.append(r)
    return out


def main():
    d = load()
    d.to_csv(f"{RESULTS}/a7_merged_candidates.csv", index=False)
    rows_c, rows_p = [], []
    run_block(d, "cvae", "random", "cvae vs full random pool", rows_c, rows_p)
    run_block(d, "cvae", "random_matched", "cvae vs size-matched random (n=66)", rows_c, rows_p)
    run_block(d, "cvae", "random_freq", "cvae vs training-frequency-weighted random", rows_c, rows_p)

    pd.DataFrame(rows_c).to_csv(f"{RESULTS}/a7_compare_continuous.csv", index=False)
    pd.DataFrame(rows_p).to_csv(f"{RESULTS}/a7_compare_proportions.csv", index=False)
    pd.DataFrame(steelman(d)).to_csv(f"{RESULTS}/a7_steelman.csv", index=False)

    # enrichment of Stage-1 chemistries
    enr = {}
    for el in sorted(STAGE1_GOOD):
        a = d[d["population"] == "cvae"]["cation_set"].map(lambda s, e=el: e in s)
        b = d[d["population"] == "random"]["cation_set"].map(lambda s, e=el: e in s)
        enr[el] = compare_proportions(int(a.sum()), len(a), int(b.sum()), len(b), "cvae", "random")
    json.dump(enr, open(f"{RESULTS}/a7_stage1_enrichment.json", "w"), indent=2)

    summ = {}
    for n, g in d.groupby("population"):
        summ[n] = dict(
            n=len(g),
            median_band_gap=float(g["pred_band_gap_eV_mean"].median()),
            median_eps0=float(g["pred_poly_total_mean"].median()),
            median_ehull_like=float(g["pred_stability_wolverton_oxides_mean"].median()),
            median_composite=float(g["predicted_property_rank_score"].median()),
            frac_reduced_cation=float(g["requires_reduced_cation"].mean()),
            frac_open_shell=float(g["has_open_shell"].mean()),
            frac_pass_gate=float(g["passes_quality_gate"].mean()),
            frac_Hf_Zr_Al=float(g["has_stage1_good"].mean()),
            median_nn_train_distance=float(g["nn_train_distance"].median()),
            n_in_training_set=int(g["in_training_set"].sum()),
        )
    json.dump(summ, open(f"{RESULTS}/a7_compare_summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
