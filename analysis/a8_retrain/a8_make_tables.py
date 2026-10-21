"""a8 — booktabs LaTeX tables + the headline comparison CSV."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import a8_common as A

R = A.RESULTS_DIR
PRIMARY = "fb0.5_b0.01"
DIVERSE = "fb1.0_b0.01"


def esc(s):
    return str(s).replace("_", r"\_")


def wrap(body, caption, label, colspec, header, note=None):
    L = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}",
         r"\label{" + label + "}", r"\begin{tabular}{" + colspec + "}", r"\toprule",
         header + r" \\", r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}"]
    if note:
        L.append(r"\vspace{0.3em}\footnotesize{" + note + "}")
    L += [r"\end{table}", ""]
    return "\n".join(L)


def ms(g, c, fmt="{:.4f}"):
    m, s = g[c].mean(), g[c].std(ddof=1) if len(g) > 1 else 0.0
    return (fmt + r" $\pm$ " + fmt).format(m, s)


def main() -> None:
    sweep = pd.read_csv(R / "a8_sweep_all.csv")
    rnd = json.loads((R / "a2_random_baseline_summary.json").read_text())
    genP = pd.read_csv(R / f"a8_downstream_{PRIMARY}_generation_runs.csv")
    genD = pd.read_csv(R / f"a8_downstream_{DIVERSE}_generation_runs.csv")
    a2runs = pd.read_csv(R / "a2_random_baseline_runs.csv")
    dep = a2runs[a2runs["arm"] == "cvae"]

    # ---- Table 1: headline before / after / random -------------------------------
    rows = [
        ("Deposited checkpoint (buggy free bits)", dep, None),
        (f"Retrained, per-dim free bits $\\lambda=0.5$, $\\beta_{{\\max}}=0.01$", genP, None),
        (f"Retrained, per-dim free bits $\\lambda=1.0$, $\\beta_{{\\max}}=0.01$", genD, None),
    ]
    body = []
    for name, g, _ in rows:
        body.append(" & ".join([name, ms(g, "validity_rate"),
                                ms(g, "unique_valid_count", "{:.0f}"),
                                ms(g, "novelty_rate"),
                                ms(g, "mean_pairwise_composition_distance"),
                                ms(g, "unique_formula_count_all", "{:.0f}")]) + r" \\")
    body.append(r"\midrule")
    for arm, label in [("rand_uniform41", "Random cations, uniform over the 41-element whitelist"),
                       ("rand_dirichlet41", "Random, no model input at all (Dirichlet)"),
                       ("rand_freq41", "Random, training-frequency weighted (whitelist)")]:
        g = a2runs[a2runs["arm"] == arm]
        body.append(" & ".join([label, ms(g, "validity_rate"),
                                ms(g, "unique_valid_count", "{:.0f}"),
                                ms(g, "novelty_rate"),
                                ms(g, "mean_pairwise_composition_distance"),
                                ms(g, "unique_formula_count_all", "{:.0f}")]) + r" \\")
    t1 = wrap(body,
              "Composition-CVAE before and after the free-bits correction, against the "
              "random-selection controls. All arms pass through the identical "
              "\\texttt{decode\\_composition\\_vector} charge-balance and whitelist filter; "
              "20{,}000 samples per run; mean $\\pm$ s.d. over independent seeds "
              "(5 for the CVAE arms, 5 for the random controls).",
              "tab:a8_headline", "lccccc",
              "Generator & Validity & Unique valid & Novelty & Mean pairwise dist. & "
              "Distinct decoded")

    # ---- Table 2: the trade-off curve --------------------------------------------
    s = sweep.copy()
    s = s[s["loss"] != "buggy"]
    s = s[s["beta_max"].isin([0.01, 0.1])]
    s = s.sort_values(["beta_max", "free_bits_per_dim"])
    body = []
    prev_beta = None
    for _, r in s.iterrows():
        if prev_beta is not None and r["beta_max"] != prev_beta:
            body.append(r"\midrule")
        prev_beta = r["beta_max"]
        body.append(" & ".join([
            f"{r['beta_max']:g}", f"{r['free_bits_per_dim']:g}",
            f"{r['test_composition_mse']:.5f}", f"{r['total_kl_nats']:.2f}",
            f"{int(r['active_units_kl_gt_0.01'])}",
            f"{r['gen_validity_rate_mean']:.4f}",
            f"{r['gen_unique_valid_count_mean']:.0f}",
            f"{r['gen_mean_pairwise_composition_distance_mean']:.4f}",
            f"{int(r['fixed_cond_unique_valid'])}"]) + r" \\")
    dr = sweep[sweep["loss"] == "deposited"].iloc[0]
    body = [" & ".join(["--", "buggy", "0.12786", f"{dr['total_kl_nats']:.2f}", "1",
                        f"{dr['gen_validity_rate_mean']:.4f}",
                        f"{dr['gen_unique_valid_count_mean']:.0f}",
                        f"{dr['gen_mean_pairwise_composition_distance_mean']:.4f}",
                        f"{int(dr['fixed_cond_unique_valid'])}"]) + r" \\", r"\midrule"] + body
    t2 = wrap(body,
              "Free-bits sweep. $\\lambda$ is the per-dimension KL floor in nats; "
              "reconstruction is the test-split composition MSE per sample; active units "
              "are latent dimensions with KL $>0.01$ nats over the full 25{,}148-record "
              "table; generation metrics are 20{,}000 samples at the archived settings "
              "(seed 123). The last column is the number of distinct valid formulas "
              "obtained from 20{,}000 latent draws with the condition pinned at "
              "$(0.9, 0.8)$ --- the direct measure of whether the latent space does "
              "anything. The first row is the deposited checkpoint.",
              "tab:a8_freebits_sweep", "llccccccc",
              r"$\beta_{\max}$ & $\lambda$ & Recon. MSE & Total KL & Active & Validity & "
              r"Unique valid & Pairwise dist. & Fixed-cond.\ unique")

    # ---- Table 3: unconditional controls on the retrained model -------------------
    uc = pd.read_csv(R / f"a8_downstream_{PRIMARY}_unconditional_runs.csv")
    a2uc = pd.read_csv(R / "a2_unconditional_runs.csv")
    order = ["conditioned", "shuffled", "const_mid", "const_target", "const_train_mean",
             "uniform_random"]
    pretty = {"conditioned": "Archived conditioning $\\mathcal{N}(0.9,0.30),\\mathcal{N}(0.8,0.30)$",
              "shuffled": "Same conditions, permuted across samples",
              "const_mid": "Constant $(0.5,0.5)$", "const_target": "Constant $(0.9,0.8)$",
              "const_train_mean": "Constant training mean", "uniform_random": "Uniform $(0,1)^2$"}
    body = []
    for arm in order:
        g1, g0 = uc[uc["arm"] == arm], a2uc[a2uc["arm"] == arm]
        body.append(" & ".join([pretty[arm],
                                ms(g0, "validity_rate"), ms(g0, "unique_valid_count", "{:.0f}"),
                                ms(g1, "validity_rate"), ms(g1, "unique_valid_count", "{:.0f}")])
                    + r" \\")
    t3 = wrap(body,
              "Condition-ablation control, deposited checkpoint versus the retrained model "
              "($\\lambda=0.5$, $\\beta_{\\max}=0.01$). The latent draw is shared within each "
              "seed, so every difference is attributable to the condition vector alone. "
              "20{,}000 samples per cell; mean $\\pm$ s.d. over seeds (5 deposited, 3 retrained).",
              "tab:a8_unconditional", "lcccc",
              r"& \multicolumn{2}{c}{Deposited} & \multicolumn{2}{c}{Retrained} \\"
              "\n" + r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + "\n"
              r"Condition arm & Validity & Unique valid & Validity & Unique valid")

    # ---- Table 4: decoder capacity (top_cations) ---------------------------------
    body = []
    frames = {"Deposited": pd.read_csv(R / "a8_downstream_deposited_top_cations.csv"),
              f"Retrained $\\lambda=0.5$": pd.read_csv(R / f"a8_downstream_{PRIMARY}_top_cations.csv"),
              f"Retrained $\\lambda=1.0$": pd.read_csv(R / f"a8_downstream_{DIVERSE}_top_cations.csv")}
    for name, df in frames.items():
        for t in sorted(df["top_cations"].unique()):
            g = df[df["top_cations"] == t]
            body.append(" & ".join([name if t == sorted(df["top_cations"].unique())[0] else "",
                                    str(int(t)), ms(g, "validity_rate"),
                                    ms(g, "unique_valid_count", "{:.0f}"),
                                    ms(g, "unique_formula_count_all", "{:.0f}")]) + r" \\")
        body.append(r"\midrule")
    body = body[:-1]
    t4 = wrap(body,
              "Decoder-capacity experiment. \\texttt{decode\\_composition\\_vector} keeps the "
              "\\texttt{top\\_cations} largest cation fractions; the deposited pipeline uses 3. "
              "20{,}000 samples, 3 seeds, mean $\\pm$ s.d.",
              "tab:a8_top_cations", "llccc",
              r"Model & \texttt{top\_cations} & Validity & Unique valid & Distinct decoded")

    (R / "a8_tables.tex").write_text("\n".join([t1, t2, t3, t4]), encoding="utf-8")

    # headline CSV
    head = []
    for name, g in [("deposited_cvae", dep), (f"retrained_{PRIMARY}", genP),
                    (f"retrained_{DIVERSE}", genD)] + \
                   [(a, a2runs[a2runs["arm"] == a]) for a in
                    ("rand_uniform41", "rand_dirichlet41", "rand_freq41",
                     "rand_uniform77", "rand_freq77")]:
        head.append({"arm": name, "n_seeds": len(g),
                     **{f"{c}_{k}": (g[c].mean() if k == "mean" else g[c].std(ddof=1))
                        for c in ("validity_rate", "unique_valid_count", "novelty_rate",
                                  "mean_pairwise_composition_distance",
                                  "unique_formula_count_all")
                        for k in ("mean", "std")}})
    pd.DataFrame(head).to_csv(R / "a8_headline_comparison.csv", index=False)
    print(pd.DataFrame(head)[["arm", "validity_rate_mean", "unique_valid_count_mean",
                              "novelty_rate_mean",
                              "mean_pairwise_composition_distance_mean"]].to_string(index=False))
    print("wrote", R / "a8_tables.tex")


if __name__ == "__main__":
    main()
