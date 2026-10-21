"""Build publication-ready booktabs table bodies from the a2 result files."""
from __future__ import annotations

import json

import pandas as pd

import a2_common as C

R = C.RESULTS_DIR


def fmt(x, d=3):
    return f"{x:.{d}f}"


def pm(m, s, d=3):
    return f"{m:.{d}f} $\\pm$ {s:.{d}f}"


def table1() -> str:
    d = json.loads((R / "a2_reproduce_summary.json").read_text())
    a = d["archived_metrics_json"]
    ms = d["manuscript_reported"]
    lines = [
        r"% Table A2-1: reproduction of the deposited Composition-CVAE generation run",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Source & Valid / 20{,}000 & Validity & Unique valid & Novel unique & Mean pairwise dist. \\",
        r"\midrule",
        f"Manuscript (as printed) & 6{{,}}577 & {ms['validity_rate']:.3f} & 71 & 69 & 0.278 \\\\",
        f"Deposited metrics JSON & 6{{,}}768 & {a['validity_rate']:.4f} & {a['unique_valid_count']} & "
        f"{a['novel_unique_count']} & {a['mean_pairwise_composition_distance']:.4f} \\\\",
        r"\midrule",
        r"\multicolumn{6}{l}{\emph{This work, deposited checkpoint, 8 seeds (mean $\pm$ s.d.)}} \\",
    ]
    for j, label in [("0.08", r"\quad script default, jitter $=0.08$"),
                     ("0.3", r"\quad archived setting, jitter $=0.30$")]:
        g = d["by_jitter"][j]
        lines.append(
            f"{label} & {g['valid_count']['mean']:.0f} $\\pm$ {g['valid_count']['std']:.0f} & "
            f"{pm(g['validity_rate']['mean'], g['validity_rate']['std'], 4)} & "
            f"{pm(g['unique_valid_count']['mean'], g['unique_valid_count']['std'], 1)} & "
            f"{pm(g['novel_unique_count']['mean'], g['novel_unique_count']['std'], 1)} & "
            f"{pm(g['mean_pairwise_composition_distance']['mean'], g['mean_pairwise_composition_distance']['std'], 4)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table2() -> str:
    d = json.loads((R / "a2_random_baseline_summary.json").read_text())
    order = ["cvae", "rand_uniform41", "rand_freq41", "rand_dirichlet41",
             "rand_uniform77", "rand_freq77"]
    label = {
        "cvae": "Composition-CVAE (deposited checkpoint)",
        "rand_uniform41": "Random: uniform over 41-element whitelist",
        "rand_freq41": "Random: training-frequency weighted, whitelist",
        "rand_dirichlet41": "Random: whitelist + random fractions (no model input)",
        "rand_uniform77": "Random: uniform over 77-element vocabulary",
        "rand_freq77": "Random: training-frequency weighted, full vocabulary",
    }
    lines = [
        r"% Table A2-2: random element-selection controls through the identical filter path",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Generator & Validity & Unique valid & Novel unique & Uniqueness & Mean pairwise dist. \\",
        r"\midrule",
    ]
    for k in order:
        g = d[k]
        lines.append(
            f"{label[k]} & {pm(g['validity_rate']['mean'], g['validity_rate']['std'], 4)} & "
            f"{g['unique_valid_count']['mean']:,.0f} & {g['novel_unique_count']['mean']:,.0f} & "
            f"{pm(g['uniqueness_rate']['mean'], g['uniqueness_rate']['std'], 4)} & "
            f"{pm(g['mean_pairwise_composition_distance']['mean'], g['mean_pairwise_composition_distance']['std'], 4)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines).replace("_", r"\_")


def table3() -> str:
    d = json.loads((R / "a2_unconditional_summary.json").read_text())
    label = {
        "conditioned": r"Conditioned, archived path ($c\sim N(0.9,0.30),N(0.8,0.30)$)",
        "shuffled": "Conditions shuffled across samples",
        "uniform_random": r"Conditions $\sim U(0,1)^2$",
        "const_mid": r"Condition held at $(0.5,0.5)$",
        "const_target": r"Condition held at $(0.9,0.8)$",
        "const_train_mean": "Condition held at the training-set mean",
    }
    lines = [
        r"% Table A2-3: condition-destroyed controls (same checkpoint, same latent draw)",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Condition signal & Validity & Unique valid & Distinct formulas & Modal share & Mean pairwise dist. \\",
        r"\midrule",
    ]
    for k in ["conditioned", "shuffled", "uniform_random", "const_mid",
              "const_target", "const_train_mean"]:
        g = d[k]
        lines.append(
            f"{label[k]} & {fmt(g['validity_rate']['mean'], 4)} & "
            f"{g['unique_valid_count']['mean']:.1f} & "
            f"{g['unique_formula_count_all']['mean']:.1f} & "
            f"{fmt(g['modal_formula_share_all']['mean'], 3)} & "
            f"{fmt(g['mean_pairwise_composition_distance']['mean'], 4)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table4() -> str:
    d = json.loads((R / "a2_condition_sweep_summary.json").read_text())
    cells = pd.DataFrame(d["per_cell"])
    modal = d["modal_formula_per_cell"]
    lines = [
        r"% Table A2-4: condition grid, shared fixed latent draw, 4000 samples per cell",
        r"\begin{tabular}{cccccl}",
        r"\toprule",
        r"$U_{\mathrm{stab}}$ & $U_{\mathrm{gap}}$ & Validity & Distinct formulas & Modal share & Modal decoded formula \\",
        r"\midrule",
    ]
    for _, r_ in cells.iterrows():
        lines.append(
            f"{r_['stability_utility']:.1f} & {r_['gap_utility']:.1f} & "
            f"{r_['validity_rate']:.3f} & {int(r_['unique_formula_count_all'])} & "
            f"{r_['modal_formula_share_all']:.3f} & "
            f"\\ce{{{modal[r_['cell']]}}} \\\\"
        )
    mi = d["mutual_information_bits"]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.3em}\footnotesize{Effect size of conditioning: "
        f"$I(\\text{{cell}};\\text{{formula}}) = {mi['cell_vs_formula']:.2f}$ bits "
        f"(label-permutation null {mi['cell_vs_formula_permutation_null_mean']:.3f} bits; "
        f"maximum possible {mi['max_possible_bits_12_cells']:.2f} bits), "
        f"normalised MI $= {mi['normalised_MI_formula']:.2f}$; "
        f"mean Jensen--Shannon distance between per-cell element-frequency distributions "
        f"{d['js_distance_between_cells']['mean_offdiag']:.2f}; "
        f"mean Jaccard overlap of decoded formula sets {d['jaccard_formula_sets']['mean_offdiag']:.3f}; "
        f"paired formula-switch rate {d['paired_formula_switch_rate']['mean']:.3f}." + r"}",
    ]
    return "\n".join(lines)


def table5() -> str:
    df = pd.read_csv(R / "a2_diversity_sweeps.csv")
    reach = json.loads((R / "a2_decoder_reachability.json").read_text())
    lines = [
        r"% Table A2-5: diversity / mode-concentration sweeps",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Sweep & Setting & Validity & Unique valid & per 1{,}000 valid & Element entropy (bits) & Stoich. classes \\",
        r"\midrule",
    ]
    for sweep, key, name in [("latent_sigma", "latent_sigma", r"Latent $\sigma$"),
                             ("condition_jitter", "condition_jitter", "Condition jitter")]:
        sub = df[df["sweep"] == sweep]
        for val, g in sub.groupby(key):
            lines.append(
                f"{name} & {val:g} & {g['validity_rate'].mean():.4f} & "
                f"{g['unique_valid_count'].mean():.1f} & "
                f"{g['unique_valid_per_1000_valid'].mean():.1f} & "
                f"{g['element_entropy_bits_valid'].mean():.2f} & "
                f"{g['n_stoich_classes_valid'].mean():.1f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}",
        r"\vspace{0.3em}\footnotesize{Decoder reachability: the top-3-cation / "
        r"max-count-4 / oxygen-1--12 decoder can emit "
        f"{reach['total_reachable_valid_formulas']:,} distinct charge-balanced whitelist "
        f"formulas. The CVAE reaches {reach['cvae_valid_formulas_20k']} of them "
        f"({100*reach['coverage_fraction_of_reachable_space']:.2f}\\%) in 20{{,}}000 samples; "
        f"uniform random element selection through the same decoder reaches "
        f"{reach['rand_uniform41_valid_formulas_20k']:,} "
        f"({100*reach['rand_uniform41_coverage_fraction']:.1f}\\%)." + r"}",
    ]
    return "\n".join(lines)


def main() -> None:
    out = {
        "a2_table1_reproduction.tex": table1(),
        "a2_table2_random_baseline.tex": table2(),
        "a2_table3_unconditional.tex": table3(),
        "a2_table4_condition_sweep.tex": table4(),
        "a2_table5_diversity.tex": table5(),
    }
    for name, body in out.items():
        (R / name).write_text(body + "\n", encoding="utf-8")
        print("=" * 70, name, sep="\n")
        print(body)


if __name__ == "__main__":
    main()
