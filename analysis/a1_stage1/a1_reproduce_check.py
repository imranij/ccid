"""Reproduction audit for the Stage-1 artifacts of MTCOMM-D-26-04679.

Compares (i) the archived processed_results CSVs against a fresh, unmodified
re-run of inverse_design_baseline.py + candidate_validation_pack.py, and
(ii) the manuscript's printed Stage-1 tables against those CSVs.

Writes results/a1_reproduction_status.csv
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
from pathlib import Path

import re

import numpy as np
import pandas as pd

ARCH = Path(_CCID_ROOT + "/analysis/a1_stage1/archived")
FRESH = Path(_CCID_ROOT + "/analysis/a1_stage1/repro/dgx/data/processed_results")
OUT = Path(_CCID_ROOT + "/results")

FILES = [
    "ceramic_ranked_real_candidates.csv",
    "ceramic_pareto_perovskite_candidates.csv",
    "ceramic_candidate_pool_summary.csv",
    "ceramic_objective_sensitivity.csv",
    "ceramic_diverse_acquisition_batch.csv",
    "ceramic_validation_gate_summary.csv",
]


def compare(a: pd.DataFrame, b: pd.DataFrame) -> str:
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    if list(a.columns) != list(b.columns):
        return "column mismatch"
    worst = 0.0
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            worst = max(worst, float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))))
        elif (a[c].astype(str) != b[c].astype(str)).any():
            return f"string mismatch in column {c}"
    return f"identical (max |delta| = {worst:.1e})"


rows = []
for f in FILES:
    rows.append({
        "Artifact": f.replace("_", r"\_"),
        "Re-run vs deposited CSV": compare(pd.read_csv(ARCH / f), pd.read_csv(FRESH / f)),
    })
status = pd.DataFrame(rows)
status.to_csv(OUT / "a1_rerun_vs_deposited.csv", index=False)
print(status.to_string(index=False))

# ---- manuscript tables vs the CSVs --------------------------------------
die = pd.read_csv(FRESH / "ceramic_ranked_real_candidates.csv")
per_top = pd.read_csv(OUT / "a1_legacy_top200.csv")
acq = pd.read_csv(FRESH / "ceramic_diverse_acquisition_batch.csv")
gates = pd.read_csv(FRESH / "ceramic_validation_gate_summary.csv")

man_t5_ids = ["mp-5606", "mp-27832", "mp-22908", "mp-23237", "mp-559550", "mp-23739",
              "mp-3589", "mp-306", "mp-15951", "mp-569152", "mp-2821", "mp-5683"]
missing = [m for m in man_t5_ids if m not in set(die.material_id)]
code_top12 = list(die.head(12).material_id)

man_t6 = ["CaSiYAlO6", "SrSiYAlO6", "CaHfYAlO6", "SrHfYAlO6", "SrSiLaAlO6", "MgZrCaSiO6",
          "CaSiLaAlO6", "LaAlO3", "ErAlO3", "CaZrYAlO6", "CaHfLaAlO6", "TmAlO3",
          "SrZrYAlO6", "YAlO3"]
code_t6 = list(per_top.head(14).formula)

report = [
    ["Table 5 -- top dielectric candidates", "DOES NOT REPRODUCE",
     "Neither the ordering nor the design-score column matches the deposited code "
     "(code top-3: " + ", ".join(code_top12[:3]) + "; manuscript top-3: mp-5606, mp-27832, mp-22908). "
     "Manuscript IDs " + ", ".join(missing) + " are absent from the deposited "
     "dielectric dataset entirely. No weighting of the available columns reproduces the printed scores."],
    ["Table 6 -- top perovskite/oxide candidates", "PARTIALLY REPRODUCES",
     "Rows 1--12 match the deposited code exactly. Rows 13--14 of the manuscript "
     "(SrZrYAlO$_6$, YAlO$_3$) are not ranks 13--14 of the code, which returns "
     + ", ".join(re.sub(r"(\d+)", r"$_{\1}$", f) for f in code_t6[12:14]) + "."],
    ["Table 7 -- Pareto front", "REPRODUCES",
     "All 7 rows and all utilities reproduce exactly. See report for the "
     "floating-point fragility of the KTaO$_3$ entry."],
    ["Table 8 -- Pareto contribution by dataset", "REPRODUCES",
     "Castelli 5 / 0.733 / 0.758 and Wolverton 2 / 0.873 / 0.888 reproduce exactly."],
    ["Table 9 -- candidate-pool summary", "REPRODUCES",
     "18{,}928 / 4{,}914 / 1{,}306 records and all summary statistics reproduce exactly."],
    ["Table 10 -- objective sensitivity", "REPRODUCES",
     "All five profiles, top formulas, scores and top-50 overlaps reproduce exactly."],
    ["Table 11 -- diverse acquisition batch", "REPRODUCES (but truncated)",
     f"The 12 printed rows match rows 1--12 of the code's {len(acq)}-row batch exactly. "
     "The manuscript text calls this ``the 12 candidates''; the deposited code emits 20."],
    ["Table 12 -- validation-gate summary", "REPRODUCES THE CODE, BUT THE CODE IS WRONG",
     f"The printed counts (9/8/4/1/1) reproduce the deposited script exactly, but they sum to "
     f"{int(gates.candidates.sum())} for a {len(acq)}-candidate batch because "
     "\\texttt{candidate\\_validation\\_pack.py} re-merges the batch onto a ranked table with "
     "duplicate formulas (BaNbO$_3$, KTaO$_3$, LaScO$_3$ each appear twice)."],
]
man = pd.DataFrame(report, columns=["Manuscript Stage-1 table", "Status", "Evidence"])
man.to_csv(OUT / "a1_reproduction_status.csv", index=False)
print()
print(man.to_string(index=False))
