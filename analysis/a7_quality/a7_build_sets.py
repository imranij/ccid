"""a7_build_sets.py -- construct the matched CVAE / random candidate populations.

Both populations come from the SAME filter path: the a2 harness re-used the project's
own decode_composition_vector / charge-balance / whitelist code.  The CVAE arm is the
deposited checkpoint at the archived settings (jitter 0.30, seed 123); the random arm
keeps each sample's decoded fraction profile and randomises only WHICH whitelist
element carries each fraction (a2_exp2_random_baseline.py, arm rand_uniform41).

Outputs (to shared/results):
  a7_population_cvae.csv        the unique valid CVAE candidates
  a7_population_random.csv      the unique valid random candidates (full pool)
  a7_population_random_matched.csv   a size-matched random subsample (n = n_cvae)
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys, json
import numpy as np, pandas as pd
from pymatgen.core import Composition

RESULTS = _CCID_ROOT + "/results"
sys.path.insert(0, _CCID_ROOT + "/analysis/a2_generative")
sys.path.insert(0, _CCID_ROOT + "/analysis/a4_umlip")

SEED = 20260913


def unique_valid(csv):
    d = pd.read_csv(csv)
    v = d[d["valid"] == True].copy()
    v["canonical_formula"] = v["canonical_formula"].astype(str)
    # first occurrence, plus multiplicity in the 20,000-sample stream
    mult = v["canonical_formula"].value_counts()
    u = v.drop_duplicates("canonical_formula").copy()
    u["n_draws"] = u["canonical_formula"].map(mult).astype(int)
    u["reduced_formula"] = [Composition(f).reduced_formula for f in u["canonical_formula"]]
    return u.reset_index(drop=True)


def main():
    # CVAE population = the DEPOSITED archived candidate file (66 unique valid), which is
    # also exactly the set that the a4 uMLIP screen relaxed.  The a2 re-run at jitter 0.30
    # gives 68 unique valid with 58 in common; the deposited file is used so that every
    # downstream a4/a5 result attaches to the same formulas.
    cvae = unique_valid(_CCID_ROOT + "/dgx/outputs_dgx/composition_cvae_generated_candidates.csv")
    rand = unique_valid(f"{RESULTS}/a2_baseline_rand_uniform41_seed123.csv")
    rdir = unique_valid(f"{RESULTS}/a2_baseline_rand_dirichlet41_seed123.csv")
    # Frequency-weighted random: cations drawn from the whitelist with probability
    # proportional to their occurrence in the CVAE's own training table. This is the
    # "learned element marginals only" control -- the fairest hard test of whether the
    # CVAE contributes anything beyond first-order element frequencies.
    rfrq = unique_valid(f"{RESULTS}/a2_baseline_rand_freq41_seed123.csv")
    rfrq["population"] = "random_freq41"
    cvae["population"] = "cvae"
    rand["population"] = "random_uniform41"
    rdir["population"] = "random_dirichlet41"

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(rand), size=len(cvae), replace=False)
    matched = rand.iloc[np.sort(idx)].copy()
    matched["population"] = "random_matched"

    cols = ["population", "canonical_formula", "reduced_formula", "formula", "elements",
            "n_draws", "sample_id"]
    rfrq[cols].to_csv(f"{RESULTS}/a7_population_random_freq.csv", index=False)
    cvae[cols].to_csv(f"{RESULTS}/a7_population_cvae.csv", index=False)
    rand[cols].to_csv(f"{RESULTS}/a7_population_random.csv", index=False)
    rdir[cols].to_csv(f"{RESULTS}/a7_population_random_dirichlet.csv", index=False)
    matched[cols].to_csv(f"{RESULTS}/a7_population_random_matched.csv", index=False)

    summary = dict(n_cvae=len(cvae), n_random_uniform41=len(rand),
                   n_random_dirichlet41=len(rdir), n_random_freq41=len(rfrq), n_matched=len(matched),
                   overlap_cvae_random=len(set(cvae["canonical_formula"]) &
                                           set(rand["canonical_formula"])),
                   seed=SEED)
    print(json.dumps(summary, indent=2))
    json.dump(summary, open(f"{RESULTS}/a7_population_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
