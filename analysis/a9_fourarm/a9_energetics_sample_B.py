"""a9_energetics_sample_B.py -- choose the arm-B (element statistics, rand_freq41)
candidates to relax, with EXACTLY a7's selection rule for arm A
(a7_quality/a7_energetics_sample.py):

  * population: a2's rand_freq41 arm, seed 123, unique valid after the byte-identical
    decode_composition_vector filter = results/a7_population_random_freq.csv (built by
    a7_build_sets.py from a2_baseline_rand_freq41_seed123.csv);
  * eligibility: every cation lies in the 14-cation set for which a4 relaxed a binary-
    oxide reference (Al Ga Sc Y La Zr Hf Ti Sn Si Ba Sr Ca Mg);
  * draw: numpy default_rng(20260913).choice(len(eligible), min(50, n), replace=False),
    rows kept in their original order (np.sort of the drawn indices).
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys
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


def draw(pop_csv, out_sample, out_eligible, label):
    r = pd.read_csv(pop_csv)
    r = r.drop_duplicates("canonical_formula").reset_index(drop=True)
    r["cations"] = r["canonical_formula"].map(cations)
    elig = r[[c <= REF_CATIONS for c in r["cations"]]].reset_index(drop=True)
    print(f"[{label}] reference-able cations: {sorted(REF_CATIONS)}")
    print(f"[{label}] {len(elig)} / {len(r)} unique valid candidates lie entirely within that set "
          f"({100*len(elig)/len(r):.2f} %)")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(elig), size=min(N_SAMPLE, len(elig)), replace=False)
    samp = elig.iloc[np.sort(idx)].copy()
    samp["cations"] = samp["cations"].map(lambda s: " ".join(sorted(s)))
    samp.to_csv(out_sample, index=False)
    if out_eligible:
        elig.assign(cations=elig["cations"].map(lambda s: " ".join(sorted(s)))).to_csv(
            out_eligible, index=False)
    print(f"[{label}] selected {len(samp)} for relaxation (seed {SEED})")
    print(samp["canonical_formula"].tolist())
    return samp, elig


if __name__ == "__main__":
    draw(f"{RESULTS}/a7_population_random_freq.csv",
         f"{RESULTS}/a9_energetics_sample_B.csv",
         f"{RESULTS}/a9_energetics_eligible_B.csv", "B rand_freq41")
