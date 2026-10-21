"""a7_tables.py -- booktabs LaTeX tables + tidy CSVs for the a7 quality comparison."""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import json, os
import numpy as np, pandas as pd

RESULTS = _CCID_ROOT + "/results"


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def fmt_ci(ci, nd=3):
    try:
        return f"[{float(ci[0]):.{nd}f}, {float(ci[1]):.{nd}f}]"
    except Exception:
        return "--"


def parse_ci(x):
    if isinstance(x, str):
        return json.loads(x)
    return x


LABEL = {"cvae": "CVAE", "random": "Random (uniform, 41-el.)",
         "random_matched": "Random, size-matched ($n=66$)",
         "random_freq": "Random, training-frequency weighted"}


def table_overview():
    s = json.load(open(f"{RESULTS}/a7_compare_summary.json"))
    order = ["cvae", "random_matched", "random_freq", "random"]
    rows = []
    for k in order:
        if k not in s:
            continue
        v = s[k]
        rows.append(dict(population=LABEL[k], n=v["n"],
                         Eg=v["median_band_gap"], eps0=v["median_eps0"],
                         Ehull=v["median_ehull_like"],
                         openshell=100 * v["frac_open_shell"],
                         reduced=100 * v["frac_reduced_cation"],
                         gate=100 * v["frac_pass_gate"],
                         hzal=100 * v["frac_Hf_Zr_Al"],
                         nn=v["median_nn_train_distance"]))
    d = pd.DataFrame(rows)
    d.to_csv(f"{RESULTS}/a7_TABLE_overview.csv", index=False)
    t = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Quality of the CVAE candidate population against random-baseline "
         r"populations produced by the identical rule-based filter. All values are medians "
         r"(properties) or percentages of the population. $\hat{E}_g$ and $\hat\varepsilon_0$ "
         r"are composition-only surrogate predictions; open-shell and reduced-cation "
         r"fractions are exact electron-counting results, not predictions.}",
         r"\label{tab:a7_overview}",
         r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Population & $n$ & $\hat{E}_g$ & $\hat\varepsilon_0$ & $\hat{E}_{\mathrm{hull}}$ "
         r"& open-shell & Ti$^{3+}$/Sn$^{2+}$ & gate & Hf/Zr/Al \\",
         r" & & (eV) & & (eV/at) & (\%) & (\%) & (\%) & (\%) \\", r"\midrule"]
    for _, r in d.iterrows():
        t.append(f"{esc(r['population'])} & {int(r['n'])} & {r['Eg']:.2f} & {r['eps0']:.1f} & "
                 f"{r['Ehull']:.3f} & {r['openshell']:.1f} & {r['reduced']:.1f} & "
                 f"{r['gate']:.1f} & {r['hzal']:.1f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(t)


def table_continuous():
    c = pd.read_csv(f"{RESULTS}/a7_compare_continuous.csv")
    t = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Distributional comparison of surrogate-predicted properties and "
         r"distance-to-training-data. $\delta$ is Cliff's delta (CVAE minus random; "
         r"$|\delta|<0.147$ negligible, $<0.33$ small, $<0.474$ medium, else large) with a "
         r"2{,}000-resample percentile bootstrap 95\,\% CI; $p$ is a two-sided "
         r"Mann--Whitney $U$ test.}",
         r"\label{tab:a7_continuous}",
         r"\begin{tabular}{llrrrrl}", r"\toprule",
         r"Comparison & Axis & med. CVAE & med. random & $\delta$ & 95\,\% CI & $p$ \\",
         r"\midrule"]
    last = None
    for _, r in c.iterrows():
        if r["comparison"] != last:
            if last is not None:
                t.append(r"\midrule")
            last = r["comparison"]
        t.append(f"{esc(r['comparison'])} & {esc(r['axis_label'])} & {r['median_a']:.3f} & "
                 f"{r['median_b']:.3f} & {r['cliffs_delta']:+.3f} & "
                 f"{fmt_ci(parse_ci(r['cliffs_delta_ci95']))} & "
                 f"{r['p_value']:.2g} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(t)


def table_proportions():
    p = pd.read_csv(f"{RESULTS}/a7_compare_proportions.csv")
    t = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Chemical-sanity comparison. Proportions with Wilson 95\,\% intervals; "
         r"the difference carries a Newcombe hybrid-score 95\,\% CI and a Fisher exact $p$.}",
         r"\label{tab:a7_proportions}",
         r"\begin{tabular}{llrrrl}", r"\toprule",
         r"Comparison & Property & CVAE (\%) & Random (\%) & $\Delta$ (pp) & 95\,\% CI, $p$ \\",
         r"\midrule"]
    last = None
    for _, r in p.iterrows():
        if r["comparison"] != last:
            if last is not None:
                t.append(r"\midrule")
            last = r["comparison"]
        ci = parse_ci(r["diff_ci95"])
        t.append(f"{esc(r['comparison'])} & {esc(r['axis_label'])} & "
                 f"{100*r['p_a']:.1f} & {100*r['p_b']:.1f} & {100*r['diff']:+.1f} & "
                 f"[{100*ci[0]:+.1f}, {100*ci[1]:+.1f}], $p={r['fisher_p']:.1g}$ \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(t)


def table_steelman():
    f = f"{RESULTS}/a7_steelman_size_controlled.csv"
    if not os.path.exists(f):
        return ""
    s = pd.read_csv(f)
    s = s[s["axis"] == "predicted_property_rank_score"]
    t = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Size-controlled steelman. For each of 2{,}000 random subsamples of the "
         r"same budget as the CVAE ($n=66$), the median composite predicted-property score "
         r"of the top $N$ is computed; $p$ is the fraction of those equal-budget random "
         r"draws whose top-$N$ is at least as good as the CVAE's.}",
         r"\label{tab:a7_steelman}",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Random pool & $N$ & CVAE top-$N$ & random top-$N$ & $p$ \\", r"\midrule"]
    for _, r in s.iterrows():
        t.append(f"{esc(r['pool'])} & {int(r['top_N'])} & {r['cvae_topN_median']:.3f} & "
                 f"{r['null_median_of_topN_median']:.3f} & {r['p_random_draw_at_least_as_good']:.4f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(t)


def table_energetics():
    f = f"{RESULTS}/a7_energetics_comparison.csv"
    if not os.path.exists(f):
        return ""
    e = pd.read_csv(f)
    t = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Decomposition energy into binary oxides, $\Delta E_{\mathrm{dec}}$, under "
         r"a single consistent potential (lower is better). \emph{full} uses every structure "
         r"hypothesis; \emph{pyxtal\_only} restricts both arms to the random symmetric search "
         r"that every stoichiometry receives, removing the prototype-availability bias.}",
         r"\label{tab:a7_energetics}",
         r"\begin{tabular}{llrrrrrrrl}", r"\toprule",
         r"Potential & Protocol & $n_{\mathrm{CVAE}}$ & $n_{\mathrm{rand}}$ & med. CVAE & "
         r"med. rand & $\delta$ & $p$ & \%$\le 0$ CVAE & \%$\le 0$ rand \\", r"\midrule"]
    pr = pd.read_csv(f"{RESULTS}/a7_energetics_proportions.csv") \
        if os.path.exists(f"{RESULTS}/a7_energetics_proportions.csv") else None
    for _, r in e.iterrows():
        pa = pb = float("nan")
        if pr is not None:
            m = pr[(pr["calc"] == r["calc"]) & (pr["mode"] == r["mode"])]
            if len(m):
                pa, pb = 100 * m.iloc[0]["p_a"], 100 * m.iloc[0]["p_b"]
        t.append(f"{esc(r['calc'])} & {esc(r['mode'])} & {int(r['n_cvae_with_reference'])} & "
                 f"{int(r['n_random_with_reference'])} & {r['median_a']:+.4f} & "
                 f"{r['median_b']:+.4f} & {r['cliffs_delta']:+.3f} & {r['p_value']:.2g} & "
                 f"{pa:.1f} & {pb:.1f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(t)


def main():
    parts = [table_overview(), table_continuous(), table_proportions(),
             table_steelman(), table_energetics()]
    with open(f"{RESULTS}/a7_tables.tex", "w") as f:
        f.write("\n\n".join(p for p in parts if p) + "\n")
    print("wrote a7_tables.tex")


if __name__ == "__main__":
    main()
