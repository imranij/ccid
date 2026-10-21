"""a9_energetics_analyze.py -- four-arm uMLIP energetics comparison.

Arms (a ladder of chemical knowledge; all through the byte-identical filter):
  A  random, uniform 41-element whitelist   -- a7's 50 relaxed candidates (a7_*), reused
  B  element statistics (rand_freq41)        -- a9, this protocol
  C  ion substitution (Hautier 2011)         -- a9, this protocol
  D  corrected CVAE (fb0.5_b0.01)            -- a9, this protocol
  deposited CVAE (reference row, not an arm) -- a4's relaxations of its 66 candidates

Physics: decomposition energy into binary oxides via a4_analyze.analyze(), called through
a7's own staging helper (a7_energetics_analyze.analyze_table) against a4's already-relaxed
references.  Nothing is re-relaxed here.  Two protocols exactly as a7:
  full        -- minimum over every hypothesis a4_hypotheses generates
  pyxtal_only -- minimum over the pyxtal random-symmetric hypotheses only (CHGNet); for
                 MACE this is the subset of formulas whose best CHGNet hypothesis was a
                 pyxtal one (MACE only refines that single structure, a4/a7 policy)
Completeness filter exactly as a7: an arm's formula is kept only if its CHGNet hypothesis
set is complete (number of finished tags >= len(all_hypotheses(f, 4, 0))).  As in a7 the
deposited-CVAE reference row is not filtered.

Statistics: a7_stats (Cliff's delta + 2,000-resample bootstrap CI, Mann-Whitney U;
Fisher exact + Wilson/Newcombe CIs), scipy Kruskal-Wallis across A-D, Holm adjustment of
the three arm-vs-D p-values within each (potential, protocol) cell.
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import json, os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, _CCID_ROOT + "/analysis/a4_umlip")
sys.path.insert(0, _CCID_ROOT + "/analysis/a7_quality")
sys.path.insert(0, _CCID_ROOT + "/analysis/a2_generative")
sys.path.insert(0, _CCID_ROOT + "/dgx/code")
from a7_energetics_analyze import analyze_table                      # noqa: E402
from a7_stats import compare_continuous, compare_proportions, _wilson  # noqa: E402
from a4_hypotheses import all_hypotheses                             # noqa: E402

RESULTS = _CCID_ROOT + "/results"
# Default = CORRECTED: hypotheses that a4's spinel builder made with the wrong composition
# (a9_energetics_hypothesis_audit.csv) are replaced by the correctly rebuilt and relaxed
# ones (a9_energetics_spinel_fix.py), and the MACE value of an affected formula is the
# refinement of its best VALID CHGNet hypothesis.  --as-run reproduces the uncorrected
# a7-identical numbers, written with suffix _asrun.
AS_RUN = "--as-run" in sys.argv
SFX = "_asrun" if AS_RUN else ""
ARM_ORDER = ["A", "B", "C", "D"]
ARMS = {
    "A": dict(label="A: random (uniform-41)", chg="a7_hypothesis_relaxations_chgnet.csv",
              mace="a7_hypothesis_relaxations_mace.csv",
              sample="a7_energetics_random_sample.csv",
              eligible="a7_energetics_random_eligible.csv", complete=True),
    "B": dict(label="B: element statistics (freq-41)",
              chg="a9_hypothesis_relaxations_B_chgnet.csv",
              mace="a9_hypothesis_relaxations_B_mace.csv",
              sample="a9_energetics_sample_B.csv", eligible="a9_energetics_eligible_B.csv",
              complete=True),
    "C": dict(label="C: ion substitution", chg="a9_hypothesis_relaxations_C_chgnet.csv",
              mace="a9_hypothesis_relaxations_C_mace.csv",
              sample="a9_energetics_sample_C.csv", eligible="a9_umlip_eligible_C.csv",
              complete=True),
    "D": dict(label="D: corrected CVAE", chg="a9_hypothesis_relaxations_D_chgnet.csv",
              mace="a9_hypothesis_relaxations_D_mace.csv",
              sample="a9_energetics_sample_D.csv", eligible="a9_umlip_eligible_D.csv",
              complete=True),
    "CVAE_dep": dict(label="deposited CVAE (reference)",
                     chg="a4_hypothesis_relaxations_chgnet.csv",
                     mace="a4_hypothesis_relaxations_mace.csv",
                     sample=None, eligible=None, complete=False),
}
MODES = {"full": lambda d: d,
         "pyxtal_only": lambda d: d[d["tag"].astype(str).str.startswith("pyxtal")]}


def _read(path):
    d = pd.read_csv(path)
    for c in ("energy_per_atom", "energy_eV", "volume_per_atom"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def load_arm(key):
    """Return (chgnet_raw, chgnet_ok, mace_raw, mace_ok, info) or None if not run."""
    a = ARMS[key]
    pc, pm = f"{RESULTS}/{a['chg']}", f"{RESULTS}/{a['mace']}"
    if not os.path.exists(pc):
        return None
    ch = _read(pc)
    ma = _read(pm) if os.path.exists(pm) else None
    if not AS_RUN:
        aud = pd.read_csv(f"{RESULTS}/a9_energetics_hypothesis_audit.csv")
        bad = aud[(aud["arm"] == key) & (aud["valid"] == False)]
        if len(bad):
            badset = set(zip(bad["formula"], bad["tag"]))
            ch = ch[[(f, t) not in badset for f, t in zip(ch["formula"], ch["tag"])]]
            fc = _read(f"{RESULTS}/a9_hypothesis_relaxations_fix_chgnet.csv")
            ch = pd.concat([ch, fc[fc["arm"] == key].drop(columns="arm")], ignore_index=True)
            if ma is not None:
                aff = set(bad["formula"])
                fm = _read(f"{RESULTS}/a9_hypothesis_relaxations_fix_mace.csv")
                ma = pd.concat([ma[~ma["formula"].isin(aff)],
                                fm[fm["arm"] == key].drop(columns="arm")], ignore_index=True)
            print(f"[{key}] corrected {len(badset)} mis-built hypotheses in {sorted(set(bad['formula']))}")
    info = dict(arm=key, label=a["label"])
    if a["sample"]:
        samp = pd.read_csv(f"{RESULTS}/{a['sample']}")["canonical_formula"].tolist()
        info["n_sampled"] = len(samp)
        info["n_eligible"] = len(pd.read_csv(f"{RESULTS}/{a['eligible']}"))
        ch = ch[ch["formula"].isin(samp)]
        if ma is not None:
            ma = ma[ma["formula"].isin(samp)]
    else:
        info["n_sampled"] = int(ch["formula"].nunique())
        info["n_eligible"] = np.nan
    ok = ch[np.isfinite(ch["energy_per_atom"])]
    info["n_chgnet_hypotheses"] = len(ch)
    info["n_chgnet_hyp_finite"] = len(ok)
    info["chgnet_hyp_converged_frac"] = float(ch["converged"].astype(bool).mean()) if len(ch) else np.nan
    if a["complete"]:
        have = ok.groupby("formula")["tag"].nunique()
        complete = []
        for f in have.index:
            try:
                n_exp = len(all_hypotheses(f, n_random=4, seed=0))
            except Exception:
                continue
            if have[f] >= n_exp:
                complete.append(f)
        info["n_complete"] = len(complete)
        info["n_incomplete_dropped"] = int(len(have) - len(complete))
        ok = ok[ok["formula"].isin(complete)]
        if ma is not None:
            ma = ma[ma["formula"].isin(complete)]
    else:
        info["n_complete"] = int(ok["formula"].nunique())
        info["n_incomplete_dropped"] = 0
    if ma is not None:
        info["n_mace_refined"] = len(ma)
        info["mace_converged_frac"] = float(ma["converged"].astype(bool).mean()) if len(ma) else np.nan
        ma_ok = ma[np.isfinite(ma["energy_per_atom"])]
    else:
        info["n_mace_refined"] = 0; info["mace_converged_frac"] = np.nan; ma_ok = None
    return ok, ma_ok, info


def main():
    import a2_common as C
    _, known = C.get_known()

    infos, frames = [], []
    for key in list(ARMS):
        r = load_arm(key)
        if r is None:
            print(f"[{key}] no relaxation table yet -- skipped")
            continue
        ch, ma, info = r
        infos.append(info)
        for calc, h in (("chgnet", ch), ("mace", ma)):
            if h is None or h.empty:
                continue
            for mode, filt in MODES.items():
                out = analyze_table(filt(h), calc, f"a9tmp_{key}_{mode}_{calc}")
                if out.empty:
                    continue
                out["arm"] = key; out["mode"] = mode; out["pot"] = calc
                frames.append(out)
    allf = pd.concat(frames, ignore_index=True)
    allf["in_training_table"] = allf["canonical_formula"].isin(known)
    # third protocol view: full protocol restricted to formulas NOT in the training table
    # (a known compound coming out stable is a rediscovery, not a discovery)
    from pymatgen.core import Composition as _Comp
    _cc = {f: _Comp(f).reduced_composition for f in allf["canonical_formula"].unique()}
    allf["n_cations"] = allf["canonical_formula"].map(lambda f: sum(1 for e in _cc[f] if str(e) != "O"))
    allf["atoms_per_fu"] = allf["canonical_formula"].map(lambda f: int(round(_cc[f].num_atoms)))
    # a single-cation formula (e.g. ZrO2) IS a binary reference oxide: never a discovery
    nov = allf[(allf["mode"] == "full") & (~allf["in_training_table"])
               & (allf["n_cations"] >= 2)].copy()
    nov["mode"] = "full_novel"
    allf = pd.concat([allf, nov], ignore_index=True)
    allf.to_csv(f"{RESULTS}/a9_energetics_all{SFX}.csv", index=False)
    pd.DataFrame(infos).to_csv(f"{RESULTS}/a9_energetics_run_status{SFX}.csv", index=False)
    # stratification by number of distinct cations (full protocol): substitution makes many
    # 2-cation ABO3 formulas, which have small cells and get a perovskite prototype
    strat = []
    for (key, pot, nc), g in allf[allf["mode"] == "full"].groupby(["arm", "pot", "n_cations"]):
        x = g["decomposition_energy_eV_per_atom"].dropna()
        strat.append(dict(arm=key, calc=pot, n_cations=nc, n=len(x),
                          median_dE=float(x.median()) if len(x) else np.nan,
                          k_le0=int((x <= 0).sum()), frac_le0=float((x <= 0).mean()) if len(x) else np.nan,
                          n_in_training=int(g["in_training_table"].sum()),
                          median_atoms_per_fu=float(g["atoms_per_fu"].median())))
    pd.DataFrame(strat).to_csv(f"{RESULTS}/a9_energetics_by_ncations{SFX}.csv", index=False)
    present = [k for k in ARMS if k in set(allf["arm"])]
    Y = "decomposition_energy_eV_per_atom"

    # ---------------- per-arm summary ----------------
    srows = []
    info_by = {i["arm"]: i for i in infos}
    for (key, pot, mode), g in allf.groupby(["arm", "pot", "mode"]):
        x = g[Y].dropna()
        k = int((x <= 0).sum()); n = len(x)
        lo, hi = _wilson(k, n)
        nov = g[~g["in_training_table"]][Y].dropna()
        kn = int((nov <= 0).sum())
        srows.append(dict(
            arm=key, label=ARMS[key]["label"], calc=pot, mode=mode,
            n_formulas_relaxed=int(g["canonical_formula"].nunique()), n_with_reference=n,
            median_dE=float(x.median()) if n else np.nan,
            q25=float(x.quantile(.25)) if n else np.nan, q75=float(x.quantile(.75)) if n else np.nan,
            mean_dE=float(x.mean()) if n else np.nan, min_dE=float(x.min()) if n else np.nan,
            k_le0=k, frac_le0=k / n if n else np.nan, frac_le0_wilson_lo=lo, frac_le0_wilson_hi=hi,
            n_in_training=int(g.loc[g[Y].notna(), "in_training_table"].sum()),
            k_le0_in_training=int(((g[Y] <= 0) & g["in_training_table"]).sum()),
            n_novel=len(nov), k_le0_novel=kn, frac_le0_novel=kn / len(nov) if len(nov) else np.nan,
            median_hypotheses=float(g["n_hypotheses"].median()),
            best_structure_converged_frac=float(g.loc[g[Y].notna(), "converged"].astype(bool).mean()) if n else np.nan,
            chgnet_all_hyp_converged_frac=info_by[key]["chgnet_hyp_converged_frac"],
            mace_refine_converged_frac=info_by[key]["mace_converged_frac"],
            n_sampled=info_by[key]["n_sampled"], n_eligible=info_by[key]["n_eligible"],
            n_incomplete_dropped=info_by[key]["n_incomplete_dropped"]))
    summ = pd.DataFrame(srows)
    summ["_o"] = summ["arm"].map({k: i for i, k in enumerate(list(ARMS))})
    summ = summ.sort_values(["calc", "mode", "_o"]).drop(columns="_o")
    summ.to_csv(f"{RESULTS}/a9_energetics_summary{SFX}.csv", index=False)

    # ---------------- cross-potential (full protocol) ----------------
    full = allf[allf["mode"] == "full"]
    piv = full.pivot_table(index=["arm", "canonical_formula"], columns="pot",
                           values=Y).dropna().reset_index()
    xrows = []
    if {"mace", "chgnet"} <= set(piv.columns):
        piv["dE_mace_minus_chgnet"] = piv["mace"] - piv["chgnet"]
        piv.to_csv(f"{RESULTS}/a9_energetics_cross_potential_per_candidate{SFX}.csv", index=False)
        for key, g in piv.groupby("arm"):
            d = g["dE_mace_minus_chgnet"]
            xrows.append(dict(arm=key, n=len(g), median_diff=float(d.median()),
                              MAD=float(d.abs().mean()), max_abs=float(d.abs().max()),
                              sign_agreement=float(((g["mace"] <= 0) == (g["chgnet"] <= 0)).mean()),
                              n_le0_both=int(((g["mace"] <= 0) & (g["chgnet"] <= 0)).sum())))
    pd.DataFrame(xrows).to_csv(f"{RESULTS}/a9_energetics_cross_potential{SFX}.csv", index=False)

    # ---------------- pairwise vs D, and Kruskal-Wallis ----------------
    prow, krow = [], []
    for (pot, mode), g in allf.groupby(["pot", "mode"]):
        vals = {k: g[g["arm"] == k][Y].dropna().values for k in present}
        if "D" in vals and len(vals["D"]) >= 2:
            cell = []
            for k in [a for a in present if a != "D"]:
                if len(vals[k]) < 2:
                    continue
                rc = compare_continuous(vals[k], vals["D"], k, "D")
                rp = compare_proportions(int((vals[k] <= 0).sum()), len(vals[k]),
                                         int((vals["D"] <= 0).sum()), len(vals["D"]), k, "D")
                cell.append(dict(
                    calc=pot, mode=mode, arm=k, vs="D", n_arm=rc["n_a"], n_D=rc["n_b"],
                    median_arm=rc["median_a"], median_D=rc["median_b"],
                    median_diff=rc["median_diff"],
                    median_diff_ci_lo=rc["median_diff_ci95"][0], median_diff_ci_hi=rc["median_diff_ci95"][1],
                    cliffs_delta=rc["cliffs_delta"], cliffs_delta_ci_lo=rc["cliffs_delta_ci95"][0],
                    cliffs_delta_ci_hi=rc["cliffs_delta_ci95"][1],
                    effect=rc["effect_size_interpretation"], mannwhitney_U=rc["mannwhitney_U"],
                    mannwhitney_p=rc["p_value"],
                    k_le0_arm=rp["k_a"], k_le0_D=rp["k_b"], frac_le0_arm=rp["p_a"], frac_le0_D=rp["p_b"],
                    frac_diff=rp["diff"], frac_diff_ci_lo=rp["diff_ci95"][0], frac_diff_ci_hi=rp["diff_ci95"][1],
                    fisher_p=rp["fisher_p"]))
            # Holm within the three arm-vs-D tests of this cell (deposited CVAE excluded)
            for col in ("mannwhitney_p", "fisher_p"):
                arms_only = [c for c in cell if c["arm"] in ARM_ORDER]
                ps = np.array([c[col] for c in arms_only])
                order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
                for rank, i in enumerate(order):
                    run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
                for c, a in zip(arms_only, adj):
                    c[col + "_holm"] = float(a)
            prow += cell
        use = [k for k in ARM_ORDER if k in vals and len(vals[k]) >= 2]
        four = [vals[k] for k in use]
        if len(four) >= 2:
            H, p = stats.kruskal(*four)
            tab = np.array([[int((vals[k] <= 0).sum()), int((vals[k] > 0).sum())] for k in use])
            try:
                chi2, pchi, dof, _ = stats.chi2_contingency(tab)
            except ValueError:
                chi2 = pchi = np.nan; dof = 0; _ = np.array([np.nan])
            krow.append(dict(calc=pot, mode=mode, arms=" ".join(use),
                             ns=" ".join(str(len(vals[k])) for k in use),
                             kruskal_H=float(H), kruskal_p=float(p),
                             le0_chi2=float(chi2), le0_chi2_dof=int(dof), le0_chi2_p=float(pchi),
                             le0_min_expected=float(_.min())))
            abd = [k for k in ("A", "B", "D") if k in use]
            if len(abd) == 3:
                H3, p3 = stats.kruskal(*[vals[k] for k in abd])
                krow[-1].update(kruskal_ABD_H=float(H3), kruskal_ABD_p=float(p3))
    pd.DataFrame(prow).to_csv(f"{RESULTS}/a9_energetics_pairwise_vs_D{SFX}.csv", index=False)
    pd.DataFrame(krow).to_csv(f"{RESULTS}/a9_energetics_kruskal{SFX}.csv", index=False)

    # ---------------- most favourable compositions per arm ----------------
    ch_full = full[full["pot"] == "chgnet"][["arm", "formula", "canonical_formula", Y,
                                             "best_prototype", "relaxed_spacegroup",
                                             "n_hypotheses", "n_cations", "in_training_table"]]
    ma_full = full[full["pot"] == "mace"][["arm", "canonical_formula", Y]].rename(
        columns={Y: "dE_mace"})
    top = ch_full.rename(columns={Y: "dE_chgnet"}).merge(ma_full, on=["arm", "canonical_formula"], how="left")
    top = top.dropna(subset=["dE_chgnet"]).sort_values(["arm", "dE_chgnet"])
    # which other arms produced the same formula
    arms_of = top.groupby("canonical_formula")["arm"].apply(lambda s: " ".join(sorted(set(s))))
    top["also_in_arms"] = [" ".join(a for a in arms_of[f].split() if a != arm)
                           for f, arm in zip(top["canonical_formula"], top["arm"])]
    top["rank_in_arm"] = top.groupby("arm").cumcount() + 1
    # yield per relaxed candidate of NOVEL (not in training, >=2 cations) compositions that
    # are <= 0 under BOTH potentials -- the discovery-relevant count
    yrows = []
    nD = None
    for key in present:
        g = top[top["arm"] == key]
        n = len(g)
        k = int(((g["dE_chgnet"] <= 0) & (g["dE_mace"] <= 0) & (~g["in_training_table"])
                 & (g["n_cations"] >= 2)).sum())
        k_any = int(((g["dE_chgnet"] <= 0) & (g["dE_mace"] <= 0)).sum())
        lo, hi = _wilson(k, n)
        yrows.append(dict(arm=key, n_relaxed_with_reference=n, k_le0_both_any=k_any,
                          k_le0_both_novel=k, yield_novel=k / n if n else np.nan,
                          yield_novel_wilson_lo=lo, yield_novel_wilson_hi=hi))
    yd = pd.DataFrame(yrows)
    if "D" in set(yd["arm"]):
        dd = yd[yd["arm"] == "D"].iloc[0]
        yd["fisher_p_vs_D"] = [stats.fisher_exact([[r.k_le0_both_novel, r.n_relaxed_with_reference - r.k_le0_both_novel],
                                                   [dd.k_le0_both_novel, dd.n_relaxed_with_reference - dd.k_le0_both_novel]])[1]
                               if r.arm != "D" else np.nan for r in yd.itertuples()]
    yd.to_csv(f"{RESULTS}/a9_energetics_novel_yield{SFX}.csv", index=False)
    print(yd.to_string())
    top.to_csv(f"{RESULTS}/a9_energetics_ranked_candidates{SFX}.csv", index=False)
    top[top["rank_in_arm"] <= 5].to_csv(f"{RESULTS}/a9_energetics_top5_per_arm{SFX}.csv", index=False)

    # ---------------- overlap of sampled formulas between arms ----------------
    ov = []
    sets = {k: set(allf[allf["arm"] == k]["canonical_formula"]) for k in present}
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            ov.append(dict(arm_a=a, arm_b=b, n_a=len(sets[a]), n_b=len(sets[b]),
                           n_shared=len(sets[a] & sets[b]),
                           shared=" ".join(sorted(sets[a] & sets[b]))))
    pd.DataFrame(ov).to_csv(f"{RESULTS}/a9_energetics_overlap{SFX}.csv", index=False)

    write_latex(summ, pd.DataFrame(xrows), pd.DataFrame(prow))
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summ[["arm", "calc", "mode", "n_with_reference", "median_dE", "k_le0", "frac_le0",
                    "frac_le0_wilson_lo", "frac_le0_wilson_hi", "n_in_training", "k_le0_in_training",
                    "best_structure_converged_frac", "chgnet_all_hyp_converged_frac",
                    "mace_refine_converged_frac", "n_incomplete_dropped"]].to_string())
        print(pd.DataFrame(xrows).to_string())
        if prow:
            print(pd.DataFrame(prow)[["calc", "mode", "arm", "n_arm", "n_D", "median_diff",
                                      "cliffs_delta", "cliffs_delta_ci_lo", "cliffs_delta_ci_hi",
                                      "mannwhitney_p", "frac_le0_arm", "frac_le0_D", "fisher_p"]].to_string())
        print(pd.DataFrame(krow).to_string())
        print(top[top["rank_in_arm"] <= 5][["arm", "rank_in_arm", "formula", "dE_chgnet", "dE_mace",
                                           "best_prototype", "in_training_table", "also_in_arms"]].to_string())


def _f(x, nd=3, sign=True):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def write_latex(summ, cross, pw):
    """One booktabs tabular body: rows = arms (+ deposited reference), two panels
    (full protocol, PyXtal-only), both potentials."""
    lab = {"A": "A Random", "B": "B Element stats", "C": "C Substitution",
           "D": "D Corrected CVAE", "CVAE_dep": "Deposited CVAE$^\\dagger$"}
    cx = cross.set_index("arm") if len(cross) else pd.DataFrame()
    L = [r"\begin{tabular}{lrrrrrrrrl}", r"\toprule",
         r" & & \multicolumn{3}{c}{CHGNet} & \multicolumn{2}{c}{MACE-MP-0} & & & \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
         r"Arm & $n$ & med.\ $\Delta E_\mathrm{dec}$ & $\le 0$ (\%) [95\,\% CI] & $\delta$ vs D [95\,\% CI] "
         r"& med.\ $\Delta E_\mathrm{dec}$ & $\le 0$ (\%) & MAD & sign agr. & $p_\mathrm{MW}$ / $p_\mathrm{F}$ vs D \\",
         r"\midrule"]
    for mode, title in (("full", "Full protocol (prototypes + PyXtal)"),
                        ("pyxtal_only", "PyXtal-only (CHGNet); best-PyXtal subset (MACE)"),
                        ("full_novel", "Full protocol, novel only (not in training table, $\\geq$2 cations)")):
        L.append(rf"\multicolumn{{10}}{{l}}{{\emph{{{title}}}}} \\")
        for key in ["A", "B", "C", "D", "CVAE_dep"]:
            c = summ[(summ.arm == key) & (summ.calc == "chgnet") & (summ["mode"] == mode)]
            m = summ[(summ.arm == key) & (summ.calc == "mace") & (summ["mode"] == mode)]
            if c.empty:
                continue
            c = c.iloc[0]
            p = pw[(pw.arm == key) & (pw.calc == "chgnet") & (pw["mode"] == mode)] if len(pw) else pd.DataFrame()
            if key == "D":
                dl, pl = "ref.", "--"
            elif len(p):
                p = p.iloc[0]
                dl = f"{p.cliffs_delta:+.2f} [{p.cliffs_delta_ci_lo:+.2f}, {p.cliffs_delta_ci_hi:+.2f}]"
                pl = f"{p.mannwhitney_p:.2g} / {p.fisher_p:.2g}"
            else:
                dl, pl = "--", "--"
            n_txt = f"{int(c.n_with_reference)}"
            mm = m.iloc[0] if len(m) else None
            if mm is not None and mode == "pyxtal_only":
                n_txt += f"/{int(mm.n_with_reference)}"
            mad = sa = "--"
            if mode == "full" and len(cx) and key in cx.index:
                mad = f"{cx.loc[key, 'MAD']:.3f}"; sa = f"{100*cx.loc[key, 'sign_agreement']:.0f}\\%"
            m_med = _f(mm.median_dE) if mm is not None else "--"
            m_fr = f"{100*mm.frac_le0:.1f}" if mm is not None and np.isfinite(mm.frac_le0) else "--"
            L.append(f"{lab[key]} & {n_txt} & {_f(c.median_dE)} & "
                     f"{100*c.frac_le0:.1f} [{100*c.frac_le0_wilson_lo:.1f}, {100*c.frac_le0_wilson_hi:.1f}] & "
                     f"{dl} & {m_med} & {m_fr} & {mad} & {sa} & {pl} \\\\")
        L.append(r"\bottomrule" if mode == "full_novel" else r"\midrule")
    L.append(r"\end{tabular}")
    L.append("% dagger: deposited (buggy) CVAE, diagnostic reference row, not an arm; "
             "a4 relaxations, no completeness filter (as in a7).")
    L.append("% delta = Cliff's delta (arm minus D; negative = arm lower/better), 2000-resample "
             "bootstrap CI; p_MW two-sided Mann-Whitney on CHGNet dE_dec; p_F Fisher exact on "
             "the CHGNet <=0 fraction. MAD and sign agreement: MACE vs CHGNet, full protocol.")
    open(f"{RESULTS}/a9_energetics_table{SFX}.tex", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
