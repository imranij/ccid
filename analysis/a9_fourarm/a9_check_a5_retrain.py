"""a9_check_a5_retrain.py -- compare the retrained a5 surrogates' test metrics (written to
the shadow root by a9_retrain_a5.sh) with the committed a5 metric files.
Output: results/a9_a5_retrain_check.csv
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import json
from pathlib import Path

import pandas as pd

ROOT = Path(_CCID_ROOT)
SHADOW = Path(_os.environ.get("A9_A5_SHADOW",
                               Path(_os.environ.get("TMPDIR", "/tmp")) / "a9_a5_shadow_root"))


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[pre + k] = float(v)
    return out


rows = []
for name in ("a5_stability_gap_metrics.json", "a5_dielectric_metrics.json"):
    old = flat(json.load(open(ROOT / "results" / name)))
    new = flat(json.load(open(SHADOW / "results" / name)))
    for k in sorted(set(old) & set(new)):
        rows.append(dict(file=name, metric=k, committed=old[k], retrained=new[k],
                         abs_diff=abs(old[k] - new[k])))
d = pd.DataFrame(rows)
d.to_csv(ROOT / "results" / "a9_a5_retrain_check.csv", index=False)
print(d.to_string())
print("max |diff| on mae/rmse/r2:",
      d[d["metric"].str.contains(r"\.(?:mae|rmse|r2)$")]["abs_diff"].max())

# Second check: retrained models vs a7's stored predictions on a7's 66 CVAE candidates.
import sys
import numpy as np
sys.path.insert(0, str(ROOT / "analysis/a7_quality"))
import a7_surrogates as S  # noqa: E402
a7 = pd.read_csv(ROOT / "results/a7_surrogate_predictions.csv")
a7 = a7[a7["population"] == "cvae"].reset_index(drop=True)
new = S.predict(a7["formula"].tolist())
rows2 = []
for col in ("pred_band_gap_eV_mean", "pred_poly_total_mean", "pred_poly_electronic_mean",
            "pred_stability_wolverton_oxides_mean", "pred_stability_castelli_perovskites_mean"):
    diff = np.abs(new[col].to_numpy() - a7[col].to_numpy())
    rows2.append(dict(column=col, n=len(a7), max_abs_diff=float(diff.max()),
                      median_abs_diff=float(np.median(diff)),
                      spearman=float(pd.Series(new[col]).corr(a7[col], method="spearman")),
                      median_a7=float(a7[col].median()), median_retrained=float(new[col].median())))
d2 = pd.DataFrame(rows2)
d2.to_csv(ROOT / "results" / "a9_a5_retrain_check_a7_cvae_predictions.csv", index=False)
print(d2.to_string())
