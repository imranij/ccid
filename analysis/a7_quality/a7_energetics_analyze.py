"""a7_energetics_analyze.py -- decomposition energies for the random sample, and the
CVAE-vs-random comparison of the DeltaE_dec distributions.

The physics is a4's, unchanged: a4_analyze.analyze() is imported and called on the a7
relaxation table, against a4's OWN already-relaxed binary-oxide references (the files are
copied to the names a4_analyze expects; no energy is recomputed).

Two comparisons are reported:
  (i)  FULL PROTOCOL -- lowest energy over every hypothesis a4_hypotheses generates.
       Confounded: a4's CVAE candidates got 2-16 hypotheses (median 4) because textbook
       prototypes exist for their stoichiometries, so their energy bound is tighter.
  (ii) PROTOCOL-MATCHED -- both arms restricted to the pyxtal random-symmetric-search
       hypotheses only (4 per candidate for 64 of 66 CVAE candidates), which every
       composition receives regardless of stoichiometry. This removes the bias.
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import json, os, shutil, sys
import numpy as np, pandas as pd

sys.path.insert(0, _CCID_ROOT + "/analysis/a4_umlip")
sys.path.insert(0, _CCID_ROOT + "/analysis/a7_quality")
import a4_analyze                                        # noqa: E402
from a7_stats import compare_continuous, compare_proportions   # noqa: E402

RESULTS = _CCID_ROOT + "/results"


def analyze_table(hyp: pd.DataFrame, calc_src: str, tag: str) -> pd.DataFrame:
    """Run a4_analyze.analyze on an arbitrary hypothesis table by staging it under the
    filename a4_analyze expects, with a4's own reference-oxide file."""
    hyp = hyp.copy()
    for col in ("energy_per_atom", "energy_eV", "volume_per_atom"):
        if col in hyp:
            hyp[col] = pd.to_numeric(hyp[col], errors="coerce")
    hyp = hyp[np.isfinite(hyp["energy_per_atom"])]
    hyp.to_csv(f"{RESULTS}/a4_hypothesis_relaxations_{tag}.csv", index=False)
    shutil.copy(f"{RESULTS}/a4_reference_oxides_{calc_src}.csv",
                f"{RESULTS}/a4_reference_oxides_{tag}.csv")
    out = a4_analyze.analyze(tag)
    os.remove(f"{RESULTS}/a4_hypothesis_relaxations_{tag}.csv")
    os.remove(f"{RESULTS}/a4_reference_oxides_{tag}.csv")
    return out


def main():
    rows = []
    frames = {}
    for calc in ("chgnet", "mace"):
        cv_csv = f"{RESULTS}/a4_hypothesis_relaxations_{calc}.csv"
        rn_csv = f"{RESULTS}/a7_hypothesis_relaxations_{calc}.csv"
        if not (os.path.exists(cv_csv) and os.path.exists(rn_csv)):
            print(f"[{calc}] skipped (missing relaxation table)")
            continue
        cv_h = pd.read_csv(cv_csv)
        rn_h = pd.read_csv(rn_csv)
        for _d in (cv_h, rn_h):
            _d["energy_per_atom"] = pd.to_numeric(_d["energy_per_atom"], errors="coerce")
        cv_h = cv_h[np.isfinite(cv_h["energy_per_atom"])]
        rn_h = rn_h[np.isfinite(rn_h["energy_per_atom"])]
        # Keep only candidates whose FULL hypothesis set finished, so that every energy
        # is a bound over the same search effort. The expected tag set is regenerated
        # from a4_hypotheses (deterministic, seed 0).
        from a4_hypotheses import all_hypotheses
        import warnings; warnings.filterwarnings("ignore")
        # completeness is judged on the CHGNet search (MACE only refines the single best
        # CHGNet prototype per formula, by a4's policy)
        ref_h = pd.read_csv(f"{RESULTS}/a7_hypothesis_relaxations_chgnet.csv")
        ref_h["energy_per_atom"] = pd.to_numeric(ref_h["energy_per_atom"], errors="coerce")
        ref_h = ref_h[np.isfinite(ref_h["energy_per_atom"])]
        have = ref_h.groupby("formula")["tag"].nunique()
        complete = []
        for f in have.index:
            try:
                n_exp = len(all_hypotheses(f, n_random=4, seed=0))
            except Exception:
                continue
            if have[f] >= n_exp:
                complete.append(f)
        complete = [f for f in complete if f in set(rn_h["formula"])]
        print(f"[{calc}] {len(complete)}/{len(have)} random candidates have a complete "
              f"hypothesis set; incomplete ones dropped")
        rn_h = rn_h[rn_h["formula"].isin(complete)]

        for mode, filt in (("full", lambda d: d),
                           ("pyxtal_only", lambda d: d[d["tag"].str.startswith("pyxtal")])):
            cv = analyze_table(filt(cv_h), calc, f"a7cv_{mode}_{calc}")
            rn = analyze_table(filt(rn_h), calc, f"a7rn_{mode}_{calc}")
            cv["population"] = "cvae"; rn["population"] = "random"
            cv["mode"] = rn["mode"] = mode
            both = pd.concat([cv, rn], ignore_index=True)
            frames[(calc, mode)] = both
            a = cv["decomposition_energy_eV_per_atom"].dropna()
            b = rn["decomposition_energy_eV_per_atom"].dropna()
            r = compare_continuous(a, b, "cvae", "random")
            r.update(calc=calc, mode=mode, axis="decomposition_energy_eV_per_atom",
                     axis_label="DeltaE_dec into binary oxides (eV/atom), lower = better",
                     n_cvae_relaxed=int(cv["formula"].nunique()),
                     n_random_relaxed=int(rn["formula"].nunique()),
                     n_cvae_with_reference=int(len(a)), n_random_with_reference=int(len(b)),
                     frac_cvae_negative=float((a <= 0).mean()) if len(a) else np.nan,
                     frac_random_negative=float((b <= 0).mean()) if len(b) else np.nan,
                     median_hypotheses_cvae=float(cv["n_hypotheses"].median()),
                     median_hypotheses_random=float(rn["n_hypotheses"].median()))
            rows.append(r)
            print(f"[{calc}/{mode}] cvae n={len(a)} med={a.median() if len(a) else np.nan:.4f} | "
                  f"random n={len(b)} med={b.median() if len(b) else np.nan:.4f} | "
                  f"p={r.get('p_value')} delta={r.get('cliffs_delta')}")

    # proportion of each arm that is stable against binary-oxide decomposition, and the
    # cross-potential (MACE vs CHGNet) agreement per arm
    prop_rows, cross_rows = [], []
    if frames:
        allf = pd.concat(frames.values(), ignore_index=True)
        allf["pot"] = np.where(allf["calc"].str.endswith("mace"), "mace", "chgnet")
        for (pot, mode), g in allf.groupby(["pot", "mode"]):
            a = g[g["population"] == "cvae"]["decomposition_energy_eV_per_atom"].dropna()
            b = g[g["population"] == "random"]["decomposition_energy_eV_per_atom"].dropna()
            if len(a) and len(b):
                r = compare_proportions(int((a <= 0).sum()), len(a), int((b <= 0).sum()), len(b),
                                        "cvae", "random")
                r.update(calc=pot, mode=mode, axis="frac_dE_dec_le_0",
                         axis_label="fraction stable against decomposition into binary oxides")
                prop_rows.append(r)
        piv = allf[allf["mode"] == "full"].pivot_table(
            index=["population", "canonical_formula"], columns="pot",
            values="decomposition_energy_eV_per_atom").dropna().reset_index()
        if "mace" in piv and "chgnet" in piv:
            piv["dE_mace_minus_chgnet"] = piv["mace"] - piv["chgnet"]
            piv.to_csv(f"{RESULTS}/a7_cross_potential_check.csv", index=False)
            for pop, g in piv.groupby("population"):
                cross_rows.append(dict(population=pop, n=len(g),
                                       median_diff=float(g["dE_mace_minus_chgnet"].median()),
                                       MAD=float(g["dE_mace_minus_chgnet"].abs().mean()),
                                       max_abs=float(g["dE_mace_minus_chgnet"].abs().max()),
                                       sign_agreement=float(((g["mace"] <= 0) == (g["chgnet"] <= 0)).mean())))
        pd.DataFrame(prop_rows).to_csv(f"{RESULTS}/a7_energetics_proportions.csv", index=False)
        pd.DataFrame(cross_rows).to_csv(f"{RESULTS}/a7_energetics_cross_potential.csv", index=False)
        print(pd.DataFrame(cross_rows).to_string())
        print(pd.DataFrame(prop_rows)[["calc","mode","p_a","p_b","diff","fisher_p"]].to_string())

    if frames:
        pd.concat(frames.values(), ignore_index=True).to_csv(
            f"{RESULTS}/a7_energetics_all.csv", index=False)
        pd.DataFrame(rows).to_csv(f"{RESULTS}/a7_energetics_comparison.csv", index=False)
    else:
        print("no energetics available yet")


if __name__ == "__main__":
    main()
