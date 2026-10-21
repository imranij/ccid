"""a9_umlip_sample.py -- hand-off for the energetics agent: arms C and D.

a7's selection rule (a7_quality/a7_energetics_sample.py), applied to each arm's POOLED
unique valid set (seeds 0-4, a9_population_<arm>.csv, first-occurrence order):
  * eligible = every cation in the 14-cation set that has a relaxed binary-oxide reference
    (a4_prototypes.OXIDE_OF minus MISSING_BINARY_REFERENCES:
     Al Ga Sc Y La Zr Hf Ti Sn Si Ba Sr Ca Mg);
  * sample = default_rng(20260913).choice(len(eligible), min(50, len(eligible)),
    replace=False), original order kept.
Known (training-table) compounds are NOT removed -- a7 did not remove them either; the
column in_training_set flags them.
Writes results/a9_umlip_sample_{C,D}.csv, a9_umlip_eligible_{C,D}.csv,
a9_umlip_sample_summary.json and, when both samples exist, a9_umlip_sample_READY.flag.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys
sys.path.insert(0, _CCID_ROOT + "/analysis/a9_fourarm")
import a9_common as K  # noqa: E402

import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from a4_prototypes import OXIDE_OF, MISSING_BINARY_REFERENCES

REF_CATIONS = {c for c, ox in OXIDE_OF.items() if ox not in MISSING_BINARY_REFERENCES}
N_SAMPLE = 50


def cations(f):
    return {str(e) for e in Composition(f).elements if str(e) != "O"}


def main():
    R = K.RESULTS
    summ = {"reference_cations": sorted(REF_CATIONS), "seed": K.SAMPLE_SEED,
            "pool": "pooled unique valid, seeds 0-4"}
    for arm in ("C", "D"):
        p = pd.read_csv(R / f"a9_population_{arm}.csv")
        p["cations_set"] = p["canonical_formula"].map(cations)
        elig = p[[c <= REF_CATIONS for c in p["cations_set"]]].reset_index(drop=True)
        rng = np.random.default_rng(K.SAMPLE_SEED)
        idx = rng.choice(len(elig), size=min(N_SAMPLE, len(elig)), replace=False)
        samp = elig.iloc[np.sort(idx)].copy()
        for d in (samp, elig):
            d["cations"] = d["cations_set"].map(lambda s: " ".join(sorted(s)))
        cols = ["population", "canonical_formula", "reduced_formula", "formula", "elements",
                "n_draws", "sample_id", "cations", "seed", "n_seeds", "in_training_set"]
        samp[cols].to_csv(R / f"a9_umlip_sample_{arm}.csv", index=False)
        elig[cols].to_csv(R / f"a9_umlip_eligible_{arm}.csv", index=False)
        summ[arm] = dict(n_unique_valid=int(len(p)), n_eligible=int(len(elig)),
                         frac_eligible=float(len(elig) / len(p)), n_sampled=int(len(samp)),
                         n_sampled_in_training=int(samp["in_training_set"].sum()),
                         n_eligible_in_training=int(elig["in_training_set"].sum()))
        print(arm, summ[arm])
    (R / "a9_umlip_sample_summary.json").write_text(json.dumps(summ, indent=2))
    if all((R / f"a9_umlip_sample_{a}.csv").exists() for a in ("C", "D")):
        (R / "a9_umlip_sample_READY.flag").write_text(
            "a9_umlip_sample_C.csv and a9_umlip_sample_D.csv written by a9_umlip_sample.py\n")
        print("READY flag written")


if __name__ == "__main__":
    main()
