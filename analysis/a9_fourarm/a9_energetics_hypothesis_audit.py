"""a9_energetics_hypothesis_audit.py -- check that every structure hypothesis a4's
builder produces actually has the candidate's composition.

Found while inspecting the two most negative DeltaE_dec values of the study (MgScGaO4 in
arm A, BaMgZrO4 in arm D): a4_hypotheses.prototype_hypotheses' spinel branch first writes
cations B and C onto the 16d ('Al') sites and THEN replaces every 'Mg' in the cell by
cation A.  When B or C is itself Mg (and A is not), the freshly placed Mg on 16d is also
overwritten, so the trial structure has the wrong stoichiometry (e.g. tag
'spinel_Zr[BaMg]O4' for BaMgZrO4 is actually Ba2Zr4O8).  Its energy per atom is then used
as if it were the candidate's.

This script regenerates the PROTOTYPE hypotheses (deterministic; pyxtal hypotheses are
built by pyxtal from the stoichiometry and are correct by construction) for every formula
in every arm's relaxation table, compares the reduced composition of each trial cell with
the formula, and writes results/a9_energetics_hypothesis_audit.csv listing every
(formula, tag) pair and whether it is valid.  Nothing in a4 is modified.
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys, warnings
import pandas as pd
from pymatgen.core import Composition
warnings.filterwarnings("ignore")
sys.path.insert(0, _CCID_ROOT + "/analysis/a4_umlip")
from a4_hypotheses import prototype_hypotheses  # noqa: E402

RESULTS = _CCID_ROOT + "/results"
TABLES = {"A": "a7_hypothesis_relaxations_chgnet.csv", "B": "a9_hypothesis_relaxations_B_chgnet.csv",
          "C": "a9_hypothesis_relaxations_C_chgnet.csv", "D": "a9_hypothesis_relaxations_D_chgnet.csv",
          "CVAE_dep": "a4_hypothesis_relaxations_chgnet.csv"}


def same(a: Composition, b: Composition) -> bool:
    a, b = a.reduced_composition, b.reduced_composition
    return set(a) == set(b) and all(abs(a[e] - b[e]) < 1e-6 for e in a)


def main():
    rows = []
    for arm, t in TABLES.items():
        h = pd.read_csv(f"{RESULTS}/{t}")
        for f in sorted(h["formula"].unique()):
            want = Composition(f)
            built = {tag: Composition(at.get_chemical_formula()) for tag, at in prototype_hypotheses(f)}
            for tag in h[h["formula"] == f]["tag"].unique():
                if str(tag).startswith("pyxtal"):
                    rows.append(dict(arm=arm, formula=f, tag=tag, built_formula="(pyxtal)", valid=True))
                elif tag in built:
                    rows.append(dict(arm=arm, formula=f, tag=tag,
                                     built_formula=built[tag].reduced_formula,
                                     valid=same(built[tag], want)))
                else:
                    rows.append(dict(arm=arm, formula=f, tag=tag, built_formula="(not regenerated)",
                                     valid=None))
    d = pd.DataFrame(rows)
    d.to_csv(f"{RESULTS}/a9_energetics_hypothesis_audit.csv", index=False)
    bad = d[d["valid"] == False]
    print(f"{len(d)} (arm, formula, tag) hypotheses checked; {len(bad)} have the WRONG composition "
          f"in {bad.groupby('arm')['formula'].nunique().to_dict()} formulas; "
          f"{int(d['valid'].isna().sum())} could not be regenerated")
    print(bad.to_string())


if __name__ == "__main__":
    main()
