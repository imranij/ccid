"""a9_energetics_sample_CD.py -- apply a7's selection rule (14-cation eligibility, then
default_rng(20260913).choice(min(50, n)) without replacement, original order kept) to the
arm-C and arm-D files written by the generation agent:
results/a9_umlip_sample_{C,D}.csv.  If such a file already holds <= 50 eligible formulas,
the rule keeps all of them.  Formulas are re-canonicalised with the project's own
canonical_formula() so names match a4/a7 relaxation tables.
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import os, sys
import pandas as pd
from pymatgen.core import Composition
sys.path.insert(0, _CCID_ROOT + "/analysis/a9_fourarm")
sys.path.insert(0, _CCID_ROOT + "/dgx/code")
from composition_cvae_common import canonical_formula   # noqa: E402
from a9_energetics_sample_B import draw                  # noqa: E402

RESULTS = _CCID_ROOT + "/results"


def canon(f):
    c = Composition(f).reduced_composition
    return canonical_formula({str(e): float(c[e]) for e in c})


if __name__ == "__main__":
    for arm in ("C", "D"):
        src = f"{RESULTS}/a9_umlip_sample_{arm}.csv"
        d = pd.read_csv(src)
        col = "canonical_formula" if "canonical_formula" in d else "formula"
        d["source_formula"] = d[col]
        d["canonical_formula"] = d[col].map(canon)
        changed = int((d["canonical_formula"] != d["source_formula"]).sum())
        print(f"[{arm}] {len(d)} rows in {src}; {changed} formulas re-canonicalised")
        tmp = f"{RESULTS}/a9_energetics_input_{arm}.csv"
        d.to_csv(tmp, index=False)
        # the eligible POOL for C/D is the generation agent's a9_umlip_eligible_<arm>.csv
        # (its input here is already the 50-formula sample), so no eligible file is written
        draw(tmp, f"{RESULTS}/a9_energetics_sample_{arm}.csv", None, arm)
        os.remove(tmp)
