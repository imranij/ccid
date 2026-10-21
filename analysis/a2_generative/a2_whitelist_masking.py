"""How often does the CVAE propose a non-whitelist element, and how often is that
visible in the reported status? The charge-balance step runs first and absorbs most
of them, so the reported "risk elements present" count understates the problem."""
from __future__ import annotations

import json

import pandas as pd

import a2_common as C

def main() -> None:
    b = pd.read_csv(C.RESULTS_DIR / "a2_reproduce_seed123_jitter0.3_candidates.csv")
    els = b["elements"].astype(str).str.split()
    has_unusual = els.map(lambda L: any(e not in C.COMMON_CERAMIC_ELEMENTS for e in L))
    out = {
        "n_samples": int(len(b)),
        "frac_samples_with_nonwhitelist_element": float(has_unusual.mean()),
        "n_flagged_risk_elements_present": int((b["status"] == "risk elements present").sum()),
        "status_breakdown_of_nonwhitelist_samples":
            b[has_unusual]["status"].value_counts().to_dict(),
        "nonwhitelist_element_counts": pd.Series(
            [e for L in els for e in L if e not in C.COMMON_CERAMIC_ELEMENTS]
        ).value_counts().to_dict(),
    }
    (C.RESULTS_DIR / "a2_whitelist_masking.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
