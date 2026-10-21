"""a9_chemistry.py -- chemical-sanity annotation of every arm's pooled unique valid set.

Uses a7_chem_sanity.annotate (oxidation-state assignment from the project's full
OXIDATION_STATES table preferring the highest states; open-shell d/f flag; forced
Ti3+/Sn2+) and a7_chem_sanity.nn_distance (Euclidean distance to the nearest of the
25,148 training compositions in the 77-dim fraction space), unchanged.
Output: results/a9_chemical_sanity.csv
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys
sys.path.insert(0, _CCID_ROOT + "/analysis/a9_fourarm")
import a9_common as K  # noqa: E402

import time
import numpy as np
import pandas as pd

import a2_common as C
import a7_chem_sanity as CS
from generate_composition_candidates import composition_vector


def main():
    t0 = time.time()
    vocab = C.element_vocab()
    table, known = C.get_known()
    train_formulas = table["formula"].astype(str).tolist()
    train_mat = np.vstack([composition_vector(f, vocab) for f in train_formulas])
    frames = []
    for arm in ("A", "B", "C", "D"):
        d = pd.read_csv(K.RESULTS / f"a9_population_{arm}.csv")
        d = CS.annotate(d)
        dist, arg = CS.nn_distance(d["canonical_formula"].tolist(), vocab, train_mat)
        d["nn_train_distance"] = dist
        d["nn_train_formula"] = [train_formulas[i] for i in arg]
        d["in_training_set"] = [f in known for f in d["canonical_formula"]]
        frames.append(d)
        print(f"{arm} n={len(d):6d} reduced-cation {d['requires_reduced_cation'].mean():.3f} "
              f"open-shell {d['has_open_shell'].mean():.3f} "
              f"nn_dist med {np.median(dist):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    pd.concat(frames, ignore_index=True).to_csv(K.RESULTS / "a9_chemical_sanity.csv", index=False)


if __name__ == "__main__":
    main()
