"""a7_energetics_sample.py -- choose the random candidates to relax.

CONSTRAINT (stated plainly): the decomposition-energy analysis of a4 references every
candidate against BINARY OXIDES relaxed with the same potential, and a4 could only build
trustworthy binary-oxide seed structures for 14 cations
(Al Ga Sc Y La Zr Hf Ti Sn Si Ba Sr Ca Mg; Nb2O5/Ta2O5 were excluded as not
reconstructible offline, and every structure database is proxy-blocked).  Inventing seed
structures for Ce/Ni/Cu/Fe/Co/Bi/W/Mo/lanthanide oxides from memory would be exactly the
kind of fabricated input this analysis does not permit.

Therefore the energetics comparison is run on the subset of the random population whose
cations all lie in that same 14-cation set: 331 of 11,744 random candidates.  This is a
CHEMISTRY-MATCHED comparison -- random is held to the CVAE's own element set -- and it is
the conservative choice, because it removes any advantage the CVAE might get merely from
picking "nicer" elements.  The conclusion therefore applies within that chemical support,
not to the random population as a whole.  This is stated in the report.
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys, json
import numpy as np, pandas as pd
from pymatgen.core import Composition

RESULTS = _CCID_ROOT + "/results"
sys.path.insert(0, _CCID_ROOT + "/analysis/a4_umlip")
from a4_prototypes import OXIDE_OF, MISSING_BINARY_REFERENCES  # noqa: E402

REF_CATIONS = {c for c, ox in OXIDE_OF.items() if ox not in MISSING_BINARY_REFERENCES}
N_SAMPLE = 50
SEED = 20260913


def cations(f):
    return {str(e) for e in Composition(f).elements if str(e) != "O"}


def main():
    r = pd.read_csv(f"{RESULTS}/a7_population_random.csv")
    r["cations"] = r["canonical_formula"].map(cations)
    elig = r[[c <= REF_CATIONS for c in r["cations"]]].reset_index(drop=True)
    print(f"reference-able cations: {sorted(REF_CATIONS)}")
    print(f"{len(elig)} / {len(r)} random candidates lie entirely within that set "
          f"({100*len(elig)/len(r):.2f} %)")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(elig), size=min(N_SAMPLE, len(elig)), replace=False)
    samp = elig.iloc[np.sort(idx)].copy()
    samp["cations"] = samp["cations"].map(lambda s: " ".join(sorted(s)))
    samp.to_csv(f"{RESULTS}/a7_energetics_random_sample.csv", index=False)
    elig.assign(cations=elig["cations"].map(lambda s: " ".join(sorted(s)))).to_csv(
        f"{RESULTS}/a7_energetics_random_eligible.csv", index=False)
    print(f"selected {len(samp)} for relaxation (seed {SEED})")
    print(samp["canonical_formula"].tolist())


if __name__ == "__main__":
    main()
