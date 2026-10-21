"""a7_run_energetics.py -- run the a4 uMLIP workflow on the RANDOM-baseline sample.

Byte-identical physics to a4: same a4_hypotheses.all_hypotheses (prototype substitution
+ pyxtal random symmetric search, n_random=4, seed 0), same a4_relax.relax settings
(fmax 0.10, 150 steps), same binary-oxide references (a4_reference_oxides_*.csv, already
relaxed by a4 -- not recomputed), same a4_analyze.decomposition_recipe.
Only the input formulas differ.

Usage:
  python a7_run_energetics.py --calc chgnet
  python a7_run_energetics.py --calc mace       (best CHGNet prototype only)
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import argparse, os, sys, time, warnings
# 2-core sandbox shared with other agents' jobs: BLAS/OMP thread oversubscription made a
# single 18-atom CHGNet relaxation take 320 s instead of the ~24 s a4 measured. Pinning to
# one thread per process removes the thrashing. This changes NOTHING about the physics.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd

sys.path.insert(0, _CCID_ROOT + "/analysis/a4_umlip")
warnings.filterwarnings("ignore")

import torch  # noqa: E402
torch.set_num_threads(1)
from a4_run_screen import run_candidates              # noqa: E402
RESULTS = _CCID_ROOT + "/results"


def formulas():
    return pd.read_csv(f"{RESULTS}/a7_energetics_random_sample.csv")["canonical_formula"].tolist()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc", default="chgnet")
    args = ap.parse_args()
    fs = formulas()
    out = os.path.join(RESULTS, f"a7_hypothesis_relaxations_{args.calc}.csv")
    if args.calc == "mace":
        # refine only the lowest-energy CHGNet prototype per formula (a4_mace_refine policy)
        ch = pd.read_csv(os.path.join(RESULTS, "a7_hypothesis_relaxations_chgnet.csv"))
        ch = ch[np.isfinite(ch["energy_per_atom"])]
        best = ch.loc[ch.groupby("formula")["energy_per_atom"].idxmin()]
        keep = set(zip(best["formula"], best["tag"]))
        print(f"MACE refine: {len(keep)} best prototypes")
        for f in sorted({k[0] for k in keep}):
            tags = {t for (ff, t) in keep if ff == f}
            run_candidates([f], "mace", out, tag_filter=lambda t, _t=tags: t in _t)
    else:
        print(f"{len(fs)} random candidates, calc={args.calc}", flush=True)
        run_candidates(fs, args.calc, out)
