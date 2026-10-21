"""a9_tables.py -- booktabs LaTeX tables for the four-arm comparison.
Reads a9_counts_summary.json, a9_compare_summary.json, a9_compare_proportions.csv,
a9_compare_continuous.csv, a9_steelman_size_controlled.csv.
Writes results/a9_table_counts.tex, a9_table_chemistry.tex, a9_table_surrogates.tex
(and the same numbers as a9_TABLE_*.csv).
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys
sys.path.insert(0, _CCID_ROOT + "/analysis/a9_fourarm")
import a9_common as K  # noqa: E402

import ast
import json

import pandas as pd

R = K.RESULTS
NAME = {"A": r"A\quad Random (uniform)", "B": r"B\quad Element statistics",
        "C": r"C\quad Ion substitution", "D": r"D\quad Learned (corrected CVAE)",
        "A_dirichlet": r"A$'$\ Random, model-free fractions (sensitivity)"}


def ci(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def pfmt(p):
    return "<10^{-300}" if p == 0 else (f"{p:.2g}" if p >= 1e-3 else f"{p:.0e}".replace("e-0", "e-").replace("e-", r"\times10^{-") + "}")


def counts():
    s = json.loads((R / "a9_counts_summary.json").read_text())
    rows, out = [], []
    for a in ("A", "B", "C", "D", "A_dirichlet"):
        d = s[a]
        m = lambda k, nd: f"{d[k]['mean']:.{nd}f} $\\pm$ {d[k]['std']:.{nd}f}"
        rows.append(dict(arm=a, validity=d["validity_rate"]["mean"], validity_sd=d["validity_rate"]["std"],
                         unique_valid=d["unique_valid_count"]["mean"], unique_valid_sd=d["unique_valid_count"]["std"],
                         uniqueness=d["uniqueness_rate"]["mean"], novelty=d["novelty_rate"]["mean"],
                         mpd=d["mean_pairwise_composition_distance"]["mean"],
                         families=d["n_families_unique_valid"]["mean"],
                         pooled_unique_valid=d["pooled_unique_valid"],
                         pooled_novel=d["pooled_novel_unique"], pooled_families=d["pooled_n_families"]))
        if a == "A_dirichlet":
            out.append(r"\midrule")
        out.append(" & ".join([NAME[a], m("validity_rate", 4),
                               f"{d['unique_valid_count']['mean']:,.0f} $\\pm$ {d['unique_valid_count']['std']:.0f}".replace(",", "{,}"),
                               m("uniqueness_rate", 3), m("novelty_rate", 3),
                               m("mean_pairwise_composition_distance", 3),
                               f"{d['n_families_unique_valid']['mean']:,.0f}".replace(",", "{,}"),
                               f"{d['pooled_unique_valid']:,}".replace(",", "{,}"),
                               f"{d['pooled_novel_unique']:,}".replace(",", "{,}"),
                               f"{d['pooled_n_families']:,}".replace(",", "{,}")]) + r" \\")
    pd.DataFrame(rows).to_csv(R / "a9_TABLE_counts.csv", index=False)
    t = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\caption{Four composition-generation strategies through the identical "
         r"\texttt{decode\_composition\_vector} filter. 20{,}000 proposals per run, mean $\pm$ s.d. "
         r"over seeds 0--4. Uniqueness = unique valid / valid; novelty = fraction of unique valid "
         r"formulas absent (exact canonical formula) from the 25{,}148-record training table; mean "
         r"pairwise distance over novel unique valid formulas in the 77-element fraction space "
         r"(5{,}000-pair estimate); families = distinct substitution families (same anonymised "
         r"stoichiometry and cation periodic-group multiset). Pooled columns: union over the five "
         r"seeds. Arms A and B keep each sample's fraction profile from the deposited CVAE and "
         r"randomise element identities (the a2 design); row A$'$ draws fractions from a Dirichlet "
         r"and uses no model input.}",
         r"\label{tab:a9_counts}",
         r"\begin{tabular}{lccccccccc}", r"\toprule",
         r" & & & & & Mean pairwise & Families & \multicolumn{3}{c}{Pooled over 5 seeds} \\",
         r"\cmidrule(lr){8-10}",
         r"Arm & Validity & Unique valid & Uniqueness & Novelty & distance & per run & Unique valid & Novel & Families \\",
         r"\midrule"] + out + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (R / "a9_table_counts.tex").write_text("\n".join(t) + "\n")


def chemistry():
    s = json.loads((R / "a9_compare_summary.json").read_text())
    p = pd.read_csv(R / "a9_compare_proportions.csv")
    axes = [("has_open_shell", "frac_open_shell"), ("requires_reduced_cation", "frac_reduced_cation"),
            ("has_stage1_good", "frac_Hf_Zr_Al"), ("passes_quality_gate", "frac_pass_gate")]
    out, rows = [], []
    for a in ("A", "B", "C", "D"):
        out.append(" & ".join([NAME[a], f"{s[a]['n']:,}".replace(",", "{,}")] +
                              [f"{100*s[a][k]:.1f}" for _, k in axes]) + r" \\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{6}{l}{\emph{Difference D $-$ X, percentage points [Newcombe 95\% CI]}} \\")
    for other in ("A", "B", "C"):
        cells = []
        for ax, _ in axes:
            r = p[(p["comparison"] == f"D vs {other}") & (p["axis"] == ax)].iloc[0]
            lo, hi = ci(r["diff_ci95"])
            star = "" if r["fisher_p"] < 0.05 else r"$^{\mathrm{ns}}$"
            cells.append(f"{100*r['diff']:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]{star}".replace("-", "$-$"))
            rows.append(dict(comparison=f"D vs {other}", axis=ax, diff_pp=100 * r["diff"],
                             ci_lo_pp=100 * lo, ci_hi_pp=100 * hi, fisher_p=r["fisher_p"]))
        out.append(" & ".join([f"D vs {other}", ""] + cells) + r" \\")
    pd.DataFrame(rows).to_csv(R / "a9_TABLE_chemistry.csv", index=False)
    maxp = max(r["fisher_p"] for r in rows)
    t = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\caption{Chemistry of each arm's pooled unique valid set (seeds 0--4). Open-shell = at "
         r"least one partially filled d/f cation at the assigned oxidation state; Ti$^{3+}$/Sn$^{2+}$ "
         r"= charge balance impossible without reducing Ti or Sn; gate = predicted $E_g \geq 1$ eV "
         r"and no open-shell cation and no forced Ti$^{3+}$/Sn$^{2+}$. Percentages of the arm's "
         r"unique valid formulas. Every D $-$ X difference shown has Fisher $p \leq " +
         f"{maxp:.0e}".replace("e-0", "e-").replace("e-", r"\times10^{-") + r"}$ "
         r"($^{\mathrm{ns}}$ would mark $p \geq 0.05$).}",
         r"\label{tab:a9_chemistry}",
         r"\begin{tabular}{lrcccc}", r"\toprule",
         r"Arm & $n$ & Open-shell (\%) & Ti$^{3+}$/Sn$^{2+}$ (\%) & Hf/Zr/Al (\%) & Gate (\%) \\",
         r"\midrule"] + out + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (R / "a9_table_chemistry.tex").write_text("\n".join(t) + "\n")


def surrogates():
    s = json.loads((R / "a9_compare_summary.json").read_text())
    c = pd.read_csv(R / "a9_compare_continuous.csv")
    st = pd.read_csv(R / "a9_steelman_size_controlled.csv")
    axes = [("pred_band_gap_eV_mean", "median_band_gap", 2), ("pred_poly_total_mean", "median_eps0", 1),
            ("pred_stability_wolverton_oxides_mean", "median_ehull_like", 3),
            ("predicted_property_rank_score", "median_composite", 3)]
    out, rows = [], []
    for a in ("A", "B", "C", "D"):
        stl = []
        for N in (5, 20):
            if a == "A":
                stl.append("--")
            else:
                r = st[(st["arm"] == a) & (st["axis"] == "predicted_property_rank_score") & (st["top_N"] == N)].iloc[0]
                pv = r['p_armA_draw_at_least_as_good']
                stl.append(f"{pv:.3f}" if pv > 0 else r"$<$0.0005")
        out.append(" & ".join([NAME[a], f"{s[a]['n']:,}".replace(",", "{,}")] +
                              [f"{s[a][k]:.{nd}f}" for _, k, nd in axes] + stl) + r" \\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{8}{l}{\emph{Cliff's $\delta$, D vs X [bootstrap 95\% CI]; positive = D larger (for $\hat E_{\mathrm{hull}}$ larger is worse)}} \\")
    for other in ("A", "B", "C"):
        cells = []
        for ax, _, _ in axes:
            r = c[(c["comparison"] == f"D vs {other}") & (c["axis"] == ax)].iloc[0]
            lo, hi = ci(r["cliffs_delta_ci95"])
            cells.append(f"{r['cliffs_delta']:+.2f} [{lo:+.2f}, {hi:+.2f}]".replace("-", "$-$"))
            rows.append(dict(comparison=f"D vs {other}", axis=ax, cliffs_delta=r["cliffs_delta"],
                             ci_lo=lo, ci_hi=hi, mannwhitney_p=r["p_value"],
                             median_D=r["median_a"], median_other=r["median_b"]))
        out.append(" & ".join([f"D vs {other}", ""] + cells + ["", ""]) + r" \\")
    pd.DataFrame(rows).to_csv(R / "a9_TABLE_surrogates.csv", index=False)
    nov = []
    for a in ("B", "C", "D"):
        q = [st[(st["arm"] == a + "_novel") & (st["axis"] == "predicted_property_rank_score")
                & (st["top_N"] == N)].iloc[0]["p_armA_draw_at_least_as_good"] for N in (5, 20)]
        nov.append(f"{a} {q[0]:.3f}/{q[1]:.3f}")
    novtxt = (r" Restricted to novel formulas (not in the training table, on which the surrogates "
              r"were fitted), the same $p$ values ($N=5$/$N=20$) are " + "; ".join(nov) + ".")
    t = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\caption{Composition-only surrogate predictions (a5 models, retrained) on each arm's "
         r"pooled unique valid set: medians of predicted band gap $\hat E_g$ (eV), static "
         r"permittivity $\hat\varepsilon_0$, Wolverton-head $\hat E_{\mathrm{hull}}$-like stability "
         r"(eV/atom, lower is better) and the a7 composite score (normalised over the pooled four "
         r"arms). Last two columns: size-controlled steelman, fraction of 2{,}000 equal-budget draws "
         r"from arm A whose top-$N$ median composite is at least the arm's own top-$N$ median "
         r"(budget = the arm's pooled unique valid count)." + novtxt + "}",
         r"\label{tab:a9_surrogates}",
         r"\begin{tabular}{lrcccccc}", r"\toprule",
         r" & & & & & & \multicolumn{2}{c}{$p$(A $\geq$ arm), top-$N$} \\",
         r"\cmidrule(lr){7-8}",
         r"Arm & $n$ & $\hat E_g$ & $\hat\varepsilon_0$ & $\hat E_{\mathrm{hull}}$ & Composite & $N=5$ & $N=20$ \\",
         r"\midrule"] + out + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (R / "a9_table_surrogates.tex").write_text("\n".join(t) + "\n")


if __name__ == "__main__":
    counts()
    if (R / "a9_compare_summary.json").exists():
        chemistry()
        surrogates()
    print("tables written")
