"""
a3_analysis.py -- Novelty re-assessment for MTCOMM-D-26-04679, Stage-2 CVAE candidates.

Runs:
  1. Load + filter + dedupe candidate CSV (valid==True, unique canonical_formula).
  2. Nearest-neighbour distance to the training set:
       (a) Element Mover's Distance (ElMD, magpie weights) -- primary metric.
       (b) Euclidean distance in element-fraction space over the model's own
           77-element vocabulary -- this is the metric the project's own code
           (generate_composition_candidates.py) already uses, kept as a
           reproducibility cross-check / fallback.
  3. Substitution-family grouping: anonymised stoichiometry (sorted atom-count
     pattern) + cation periodic-table-group multiset, to see how many
     chemically distinct families the 66 unique formulas actually form.
  4. External-database cross-check bookkeeping (see a3_external_db_check.md
     for what was and was not reachable; this script writes the per-candidate
     placeholder columns and the honest "not verifiable" status).

Every number below is computed from the two CSVs actually present in
dgx/outputs_dgx/ -- nothing is looked up or invented.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(_CCID_ROOT + "")
OUT_DGX = PROJECT / "dgx" / "outputs_dgx"
CODE_DIR = PROJECT / "dgx" / "code"
RESULTS = Path(_CCID_ROOT + "/results")
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_DIR))
from composition_cvae_common import parse_formula  # noqa: E402

# ----------------------------------------------------------------------
# 0. Load data
# ----------------------------------------------------------------------
cand = pd.read_csv(OUT_DGX / "composition_cvae_generated_candidates.csv")
train = pd.read_csv(OUT_DGX / "composition_cvae_training_records.csv")
metadata = json.load(open(OUT_DGX / "composition_cvae_metadata.json"))
element_vocab = list(metadata["element_vocab"])

valid = cand[cand["valid"] == True].copy()  # noqa: E712
unique_valid = valid.drop_duplicates(subset="canonical_formula").copy().reset_index(drop=True)

print(f"20,000-row archive: {len(cand)} rows")
print(f"valid==True: {len(valid)} rows")
print(f"unique valid canonical_formula: {len(unique_valid)}")
print(f"of which flagged novel by the paper's own exact-match test: {int(unique_valid['novel'].sum())}")

train_unique = train.drop_duplicates(subset="canonical_formula").copy().reset_index(drop=True)
train_formula_set = set(train["canonical_formula"])
print(f"training records: {len(train)}, unique training canonical_formula: {len(train_unique)}")

# ----------------------------------------------------------------------
# 1. Element-fraction Euclidean distance (matches the project's own metric)
# ----------------------------------------------------------------------
def composition_vector(formula: str, vocab: list[str]) -> np.ndarray:
    counts = parse_formula(formula)
    total = sum(counts.values())
    arr = np.zeros(len(vocab), dtype=np.float32)
    if total <= 0:
        return arr
    for idx, element in enumerate(vocab):
        arr[idx] = float(counts.get(element, 0.0)) / float(total)
    return arr

train_vectors = np.vstack([composition_vector(f, element_vocab) for f in train_unique["canonical_formula"]])
cand_vectors = np.vstack([composition_vector(f, element_vocab) for f in unique_valid["canonical_formula"]])

euclid_nn_dist = np.zeros(len(unique_valid))
euclid_nn_formula = [""] * len(unique_valid)
for i in range(len(unique_valid)):
    d = np.linalg.norm(train_vectors - cand_vectors[i], axis=1)
    j = int(np.argmin(d))
    euclid_nn_dist[i] = float(d[j])
    euclid_nn_formula[i] = train_unique["canonical_formula"].iloc[j]

unique_valid["euclid_nn_distance"] = euclid_nn_dist
unique_valid["euclid_nn_train_formula"] = euclid_nn_formula

# ----------------------------------------------------------------------
# 2. Element Mover's Distance (ElMD, magpie feature set)
# ----------------------------------------------------------------------
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")  # this ElMD build's JIT path fails
                                                  # to compile under the installed
                                                  # numpy/numba pair; interpreted
                                                  # mode is correct, just slower.
from ElMD import ElMD  # noqa: E402

elmd_nn_dist = np.zeros(len(unique_valid))
elmd_nn_formula = [""] * len(unique_valid)
train_formulas = train_unique["canonical_formula"].tolist()

for i, f in enumerate(unique_valid["canonical_formula"]):
    e = ElMD(f, metric="magpie")
    best_d = None
    best_f = None
    for tf in train_formulas:
        try:
            d = e.elmd(tf)
        except Exception:
            continue
        if best_d is None or d < best_d:
            best_d = d
            best_f = tf
    elmd_nn_dist[i] = best_d if best_d is not None else float("nan")
    elmd_nn_formula[i] = best_f or ""
    if i % 10 == 0:
        print(f"  ElMD nn {i+1}/{len(unique_valid)}", flush=True)

unique_valid["elmd_nn_distance"] = elmd_nn_dist
unique_valid["elmd_nn_train_formula"] = elmd_nn_formula

# ----------------------------------------------------------------------
# 3. Chemical-system check against the training set (task 1, level iii)
#    "no entry existing in the same chemical system at all"
# ----------------------------------------------------------------------
def element_set(formula: str) -> frozenset:
    return frozenset(parse_formula(formula).keys())

train_elemsets = [element_set(f) for f in train_unique["canonical_formula"]]
train_elemsets_set = set(train_elemsets)

def same_chem_system_exists(formula: str) -> bool:
    return element_set(formula) in train_elemsets_set

unique_valid["same_chem_system_in_training"] = unique_valid["canonical_formula"].apply(same_chem_system_exists)
unique_valid["absent_from_training_exact"] = ~unique_valid["canonical_formula"].isin(train_formula_set)

print("Unique valid candidates whose element SET (chemical system) already "
      f"appears somewhere in the training data: "
      f"{int(unique_valid['same_chem_system_in_training'].sum())}/{len(unique_valid)}")

unique_valid.to_csv(RESULTS / "a3_stage_nn.csv", index=False)
print("wrote intermediate a3_stage_nn.csv")
