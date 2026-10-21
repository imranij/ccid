"""
a4_analyze.py -- turn relaxed uMLIP energies into a defensible stability statement.

WHAT IS AND IS NOT COMPUTED HERE
--------------------------------
* NOT an energy above the convex hull. No structure/thermodynamics database was
  reachable from this environment (every OPTIMADE endpoint tried -- Materials Project,
  Alexandria, OQMD, COD, AFLOW, NOMAD, Materials Cloud -- was refused by the proxy),
  so the competing-phase set needed for a hull could not be obtained.

* WHAT IS computed: the DECOMPOSITION ENERGY of each candidate into its constituent
  BINARY OXIDES, with candidate and references relaxed by the SAME potential, so the
  comparison is internally consistent and free of the O2-reference problem that makes
  uMLIP elemental formation energies unreliable.

      dE_decomp = E(candidate) - sum_i n_i E(binary oxide i)      [eV per atom]

LOGIC OF THE RESULT (this is the part the manuscript must state correctly)
-------------------------------------------------------------------------
Let E_cand be our lowest energy over a FINITE set of structure hypotheses. Then
E_cand >= E_true, so our dE_decomp is an UPPER BOUND on the true decomposition energy.
Also, the true convex hull lies at or below the binary-oxide tie-line, so the true
decomposition energy is a LOWER BOUND on the true energy above hull. Therefore:

  * dE_decomp <= 0  ==>  the candidate is genuinely stable against decomposition into
                         these binary oxides. This is real positive evidence.
  * dE_decomp >  0  ==>  INCONCLUSIVE. It may be unstable, or our structure search may
                         simply have missed its ground state. It is NOT proof of
                         instability, and must not be reported as such.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import json
import os
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
from pymatgen.core import Composition

warnings.filterwarnings("ignore")

RESULTS = _CCID_ROOT + "/results"
from a4_prototypes import OXIDE_OF, OXIDE_OF_STATE, MISSING_BINARY_REFERENCES
from a4_descriptors import reduction_requirement


def decomposition_recipe(formula: str):
    """Moles of each binary oxide per formula unit of the candidate, plus the oxygen
    residual. Returns (recipe, o_residual, ok).

    The binary oxide chosen for each cation follows the charge-balanced OXIDATION STATE
    assignment of that composition, so a candidate that requires Ti(3+) is referenced
    against Ti2O3 rather than TiO2. This makes the oxygen balance exact and avoids
    needing an O2 reference energy."""
    c = Composition(formula).reduced_composition
    d = {str(e): float(c[e]) for e in c}
    n_o = d.pop("O", 0.0)
    _, assign, _ = reduction_requirement(formula)
    recipe, o_used = {}, 0.0
    for el, n in d.items():
        ox = None
        if assign and el in assign:
            ox = OXIDE_OF_STATE.get((el, assign[el]))
        if ox is None:
            ox = OXIDE_OF.get(el)
        if ox is None:
            return None, None, False
        oc = Composition(ox)
        a = float(oc[el])
        b = float(oc["O"])
        recipe[ox] = recipe.get(ox, 0.0) + n / a
        o_used += (n / a) * b
    return recipe, n_o - o_used, True


def analyze(calc_name: str) -> pd.DataFrame:
    hyp_csv = os.path.join(RESULTS, f"a4_hypothesis_relaxations_{calc_name}.csv")
    ref_csv = os.path.join(RESULTS, f"a4_reference_oxides_{calc_name}.csv")
    if not os.path.exists(hyp_csv):
        return pd.DataFrame()
    h = pd.read_csv(hyp_csv)
    h = h[np.isfinite(h["energy_per_atom"])]
    refs = {}
    if os.path.exists(ref_csv):
        r = pd.read_csv(ref_csv)
        refs = {row["oxide"]: row["energy_per_fu"] for _, row in r.iterrows()
                if np.isfinite(row["energy_per_fu"])}

    rows = []
    for f, g in h.groupby("formula"):
        best = g.loc[g["energy_per_atom"].idxmin()]
        comp = Composition(f).reduced_composition
        n_at = float(comp.num_atoms)
        e_fu = best["energy_per_atom"] * n_at

        recipe, o_res, ok = decomposition_recipe(f)
        missing = [ox for ox in (recipe or {}) if ox not in refs]
        if ok and not missing and abs(o_res) < 1e-6:
            e_ref = sum(n * refs[ox] for ox, n in recipe.items())
            dE_fu = e_fu - e_ref
            dE_at = dE_fu / n_at
            note = "decomposition into binary oxides, same potential"
        else:
            dE_fu = dE_at = np.nan
            if not ok:
                note = "no binary-oxide reference defined for one or more cations"
            elif missing:
                note = "missing binary reference(s): " + ", ".join(sorted(missing)) + \
                       (" (structure not reconstructible offline)"
                        if set(missing) & MISSING_BINARY_REFERENCES else "")
            else:
                note = f"oxygen does not balance against binary oxides (residual {o_res:+.3f} O/f.u.)"

        rows.append(dict(
            formula=Composition(f).reduced_formula,
            canonical_formula=f,
            calc=calc_name,
            n_hypotheses=len(g),
            best_prototype=best["tag"],
            best_natoms=int(best["natoms"]),
            relaxed_spacegroup=best["spacegroup"],
            relaxed_spacegroup_number=int(best["spacegroup_number"]),
            energy_per_atom=round(float(best["energy_per_atom"]), 5),
            volume_per_atom=round(float(best["volume_per_atom"]), 4),
            converged=bool(best["converged"]),
            spread_eV_per_atom=round(float(g["energy_per_atom"].max() - g["energy_per_atom"].min()), 4),
            decomposition_energy_eV_per_atom=round(dE_at, 5) if np.isfinite(dE_at) else np.nan,
            decomposition_energy_eV_per_fu=round(dE_fu, 5) if np.isfinite(dE_fu) else np.nan,
            decomposition_recipe=json.dumps({k: round(v, 4) for k, v in (recipe or {}).items()}),
            stable_vs_binary_oxides=(bool(dE_at <= 0) if np.isfinite(dE_at) else None),
            note=note,
        ))
    return pd.DataFrame(rows).sort_values("decomposition_energy_eV_per_atom",
                                          na_position="last").reset_index(drop=True)


def cross_check(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    if a.empty or b.empty:
        return pd.DataFrame()
    m = a.merge(b, on="formula", suffixes=("_chgnet", "_mace"))
    m["dE_diff_eV_per_atom"] = (m["decomposition_energy_eV_per_atom_mace"]
                                - m["decomposition_energy_eV_per_atom_chgnet"])
    m["same_best_prototype"] = m["best_prototype_chgnet"] == m["best_prototype_mace"]
    m["same_spacegroup"] = m["relaxed_spacegroup_chgnet"] == m["relaxed_spacegroup_mace"]
    return m


if __name__ == "__main__":
    out = {}
    for calc in ("chgnet", "mace"):
        df = analyze(calc)
        if not df.empty:
            df.to_csv(os.path.join(RESULTS, f"a4_energetics_{calc}.csv"), index=False)
            n_ok = int(df["decomposition_energy_eV_per_atom"].notna().sum())
            n_stable = int((df["stable_vs_binary_oxides"] == True).sum())
            print(f"[{calc}] {len(df)} candidates with relaxed energies; "
                  f"{n_ok} with a complete binary-oxide reference set; "
                  f"{n_stable} have dE_decomp <= 0.")
            out[calc] = df
    if len(out) == 2:
        m = cross_check(out["chgnet"], out["mace"])
        if not m.empty:
            m.to_csv(os.path.join(RESULTS, "a4_cross_potential_check.csv"), index=False)
            d = m["dE_diff_eV_per_atom"].dropna()
            print(f"[cross] {len(m)} shared; MACE-CHGNet dE difference: "
                  f"mean {d.mean():+.4f}, MAD {d.abs().mean():.4f}, max {d.abs().max():.4f} eV/atom; "
                  f"same best prototype in {int(m['same_best_prototype'].sum())}/{len(m)}")
