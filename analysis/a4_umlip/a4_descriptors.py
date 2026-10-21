"""
a4_descriptors.py -- independent recomputation of the manuscript's Table 12 descriptor
screen, plus extension to the full unique-candidate pool.

Three jobs:
  1. FORENSICS. Determine whether the published Goldschmidt tolerance factors
     (t = 0.277-0.428), electronegativity differences (1.83-2.09) and composite
     "stability scores" (0.744-0.818) can be reproduced at all. No deposited code
     computes them: candidate_validation_pack.py contains no tolerance-factor,
     electronegativity or composite-score logic whatsoever. So we test a panel of
     plausible definitions against the six published values.
  2. CORRECT RECOMPUTATION with pymatgen Shannon radii and charge-balanced oxidation
     states, for all unique valid candidates.
  3. APPLICABILITY. Report, per candidate, whether a Goldschmidt tolerance factor is
     even a defined quantity for its stoichiometry.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import itertools
import json
import os
import warnings
from math import sqrt

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Species

warnings.filterwarnings("ignore")

RESULTS = _CCID_ROOT + "/results"
PROJ = _CCID_ROOT + "/dgx/outputs_dgx"

# Published Table 12 values, read from candidate_stability_assessment.json (NOT recomputed).
PUBLISHED = {
    "Al2BaHfO6": dict(t=0.428, dchi=2.087, score=0.818),
    "Al2ZrHfO7": dict(t=0.316, dchi=1.977, score=0.770),
    "Al2ScLaO6": dict(t=0.359, dchi=2.020, score=0.788),
    "Al2ZrSiO7": dict(t=0.277, dchi=1.827, score=0.744),
    "ScAlHfO5":  dict(t=0.335, dchi=2.017, score=0.780),
    "AlZrScO5":  dict(t=0.337, dchi=2.007, score=0.779),
}

# Shannon effective ionic radii, CN VI unless noted (Shannon, Acta Cryst A32, 751 (1976)).
# Used as a fallback where pymatgen's table lacks the exact (species, CN) entry.
R_VI = {("Al", 3): 0.535, ("Ga", 3): 0.620, ("Sc", 3): 0.745, ("Y", 3): 0.900,
        ("La", 3): 1.032, ("Ti", 4): 0.605, ("Zr", 4): 0.720, ("Hf", 4): 0.710,
        ("Si", 4): 0.400, ("Sn", 4): 0.690, ("Nb", 5): 0.640, ("Ta", 5): 0.640,
        ("Ba", 2): 1.350, ("Sr", 2): 1.180, ("Ca", 2): 1.000, ("Mg", 2): 0.720,
        ("Zn", 2): 0.740, ("O", -2): 1.400}
R_XII = {("Ba", 2): 1.610, ("Sr", 2): 1.440, ("Ca", 2): 1.340, ("La", 3): 1.360,
         ("Y", 3): 1.075, ("Sc", 3): 0.870, ("Na", 1): 1.390, ("K", 1): 1.640}


def shannon(el: str, ox: int, cn: str = "VI") -> float | None:
    """Shannon radius from pymatgen, with a curated fallback table."""
    try:
        return float(Species(el, ox).get_shannon_radius(cn))
    except Exception:
        pass
    if cn == "XII" and (el, ox) in R_XII:
        return R_XII[(el, ox)]
    return R_VI.get((el, ox))


# --------------------------------------------------------------------------
# Oxidation states
# --------------------------------------------------------------------------
COMMON_OX = {"Al": 3, "Ga": 3, "Sc": 3, "Y": 3, "La": 3, "Ti": 4, "Zr": 4, "Hf": 4,
             "Si": 4, "Sn": 4, "Nb": 5, "Ta": 5, "Ba": 2, "Sr": 2, "Ca": 2,
             "Mg": 2, "Zn": 2, "O": -2}


def oxidation_states(formula: str):
    """Charge-balanced oxidation states. Uses the standard single common state for each
    cation (all cations here are redox-inert in oxides), then verifies charge balance."""
    c = Composition(formula).reduced_composition
    d = {str(e): int(round(c[e])) for e in c}
    n_o = d.get("O", 0)
    cats = {k: v for k, v in d.items() if k != "O"}
    if not all(k in COMMON_OX for k in cats):
        return None, None
    q = sum(COMMON_OX[k] * v for k, v in cats.items()) - 2 * n_o
    return {**{k: COMMON_OX[k] for k in cats}, "O": -2}, q


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------
# Oxidation states the GENERATOR allows (composition_cvae_common.OXIDATION_STATES).
# Note Ti:[3,4] and Sn:[2,4] -- the generator may invoke the REDUCED state to declare a
# composition charge-balanced.
GEN_OX = {"Al": [3], "Ga": [3], "Sc": [3], "Y": [3], "La": [3], "Ti": [3, 4],
          "Zr": [4], "Hf": [4], "Si": [4], "Sn": [2, 4], "Nb": [5], "Ta": [5],
          "Ba": [2], "Sr": [2], "Ca": [2], "Mg": [2], "O": [-2]}

REDUCED_STATE = {("Ti", 3), ("Sn", 2)}


def reduction_requirement(formula: str):
    """Determine whether the composition can be charge-balanced using only the standard
    high oxidation state of each cation, or whether it REQUIRES a reduced cation
    (Ti3+ or Sn2+). Returns (requires_reduction, assignment_found, description).

    This matters because the manuscript's design target is a wide-gap dielectric
    INSULATOR: Ti(3+) is a d1 centre and Sn(2+) is a lone-pair centre, both of which
    introduce occupied mid-gap / shallow states and destroy the insulating character
    that the whole screen is supposed to select for.
    """
    c = Composition(formula).reduced_composition
    d = {str(e): int(round(c[e])) for e in c}
    n_o = d.pop("O", 0)
    els = sorted(d)
    if not all(e in GEN_OX for e in els):
        return None, None, "element outside generator oxidation table"
    sols = []
    for combo in itertools.product(*[GEN_OX[e] for e in els]):
        if sum(d[e] * s for e, s in zip(els, combo)) - 2 * n_o == 0:
            sols.append(dict(zip(els, combo)))
    if not sols:
        return None, None, "no charge-balanced assignment even with reduced states"
    all_high = [s for s in sols if not any((e, v) in REDUCED_STATE for e, v in s.items())]
    if all_high:
        return False, all_high[0], "balanced with standard (highest) oxidation states"
    red = sorted({f"{e}{v}+" for s in sols for e, v in s.items() if (e, v) in REDUCED_STATE})
    return True, sols[0], "requires reduced cation(s): " + ", ".join(red)


def electronegativity_diff(formula: str) -> float:
    """chi(O) minus the CATION-COUNT-WEIGHTED mean Pauling electronegativity.

    This exact definition reproduces all six published Delta-chi values to 3 dp, so it
    is the definition the manuscript used (see a4_descriptor_forensics.json)."""
    c = Composition(formula).reduced_composition
    d = {str(e): int(round(c[e])) for e in c}
    d.pop("O", None)
    num = sum(Element(k).X * v for k, v in d.items())
    den = sum(d.values())
    return float(Element("O").X - num / den)


def perovskite_site_assignment(formula: str):
    """Return (A_cations, B_cations) if the stoichiometry admits a defensible perovskite
    A/B assignment, else None. Only ABO3 and A2BB'O6 (n_cation:n_O = 1:3 and 4:6 with a
    doubled A) are treated as perovskite-derived; everything else returns None, because
    a Goldschmidt factor is NOT DEFINED for a non-perovskite framework."""
    c = Composition(formula).reduced_composition
    d = {str(e): int(round(c[e])) for e in c}
    n_o = d.pop("O", 0)
    cats = []
    for k, v in sorted(d.items()):
        cats += [k] * v
    n_c = len(cats)
    if n_c == 2 and n_o == 3:                       # ABO3
        big = max(cats, key=lambda e: shannon(e, COMMON_OX[e]) or 0)
        small = [e for e in cats if e != big] or [big]
        return [big], small
    if n_c == 4 and n_o == 6:                       # A2BB'O6 double perovskite
        cnt = {e: cats.count(e) for e in set(cats)}
        order = sorted(cnt, key=lambda e: -(shannon(e, COMMON_OX[e]) or 0))
        for e in order:
            if cnt[e] >= 2:
                rest = list(cats)
                rest.remove(e); rest.remove(e)
                return [e], rest
    return None


def goldschmidt(formula: str):
    """Goldschmidt tolerance factor, computed ONLY where a perovskite A/B assignment
    exists. A-site radius uses CN XII, B-site CN VI, oxide radius 1.40 A (CN VI)."""
    asg = perovskite_site_assignment(formula)
    if asg is None:
        return None, "not defined (non-perovskite stoichiometry)"
    A, B = asg
    rO = 1.40
    rA = np.mean([shannon(e, COMMON_OX[e], "XII") or shannon(e, COMMON_OX[e]) for e in A])
    rB = np.mean([shannon(e, COMMON_OX[e], "VI") for e in B])
    if rA is None or rB is None or not np.isfinite(rA) or not np.isfinite(rB):
        return None, "radius unavailable"
    return float((rA + rO) / (sqrt(2) * (rB + rO))), "perovskite-derived"


# --------------------------------------------------------------------------
# Forensics: can the published t values be reproduced by ANY plausible definition?
# --------------------------------------------------------------------------
def t_definition_panel(formula: str) -> dict:
    """Evaluate a panel of candidate tolerance-factor definitions on one formula."""
    c = Composition(formula).reduced_composition
    d = {str(e): int(round(c[e])) for e in c}
    d.pop("O", None)
    cats = []
    for k, v in sorted(d.items()):
        cats += [k] * v
    rO = 1.40
    r = {e: shannon(e, COMMON_OX[e]) for e in set(cats)}
    rXII = {e: (shannon(e, COMMON_OX[e], "XII") or r[e]) for e in set(cats)}
    if any(v is None for v in r.values()):
        return {}
    big = max(set(cats), key=lambda e: r[e])
    small = min(set(cats), key=lambda e: r[e])
    rmean = float(np.mean([r[e] for e in cats]))
    out = {}
    out["std_XII_over_VI"] = (rXII[big] + rO) / (sqrt(2) * (r[small] + rO))
    out["std_VI_over_VI"] = (r[big] + rO) / (sqrt(2) * (r[small] + rO))
    out["big_over_mean"] = (r[big] + rO) / (sqrt(2) * (rmean + rO))
    out["mean_over_mean"] = (rmean + rO) / (sqrt(2) * (rmean + rO))
    out["no_oxygen_in_numer"] = r[big] / (sqrt(2) * (r[small] + rO))
    out["no_oxygen_at_all"] = r[big] / (sqrt(2) * r[small])
    out["nO_scaled"] = (r[big] + rO) / (sqrt(2) * (r[small] + rO * int(round(Composition(formula).reduced_composition.get("O", 0)))))
    out["mean_over_big"] = (rmean + rO) / (sqrt(2) * (r[big] + rO))
    out["small_over_big"] = (r[small] + rO) / (sqrt(2) * (r[big] + rO))
    out["ratio_only"] = r[small] / r[big]
    out["mean_cation_over_O"] = rmean / rO
    out["sum_cat_over_sum_all"] = sum(r[e] for e in cats) / (sum(r[e] for e in cats) + rO * float(Composition(formula).reduced_composition.get("O", 0)))
    return out


def run_forensics() -> dict:
    rows = []
    for f, pub in PUBLISHED.items():
        panel = t_definition_panel(f)
        rows.append(dict(formula=f, published_t=pub["t"], **panel))
    df = pd.DataFrame(rows)
    # Which definition best matches the six published t values?
    best = {}
    for col in df.columns:
        if col in ("formula", "published_t"):
            continue
        err = float(np.nanmax(np.abs(df[col] - df["published_t"])))
        best[col] = err
    # Delta-chi check
    dchi = {f: dict(published=PUBLISHED[f]["dchi"], recomputed=round(electronegativity_diff(f), 3))
            for f in PUBLISHED}
    # Composite-score reconstruction: least squares on the six points
    X = np.array([[1.0, PUBLISHED[f]["t"], PUBLISHED[f]["dchi"]] for f in PUBLISHED])
    y = np.array([PUBLISHED[f]["score"] for f in PUBLISHED])
    coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - float(np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    # also try score as a pure function of dchi alone
    X1 = np.array([[1.0, PUBLISHED[f]["dchi"]] for f in PUBLISHED])
    c1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r2_1 = 1 - float(np.sum((y - X1 @ c1) ** 2) / np.sum((y - y.mean()) ** 2))
    return dict(
        t_panel=df.to_dict("records"),
        t_definition_max_abs_error=best,
        t_reproducible=bool(min(best.values()) < 0.005),
        t_best_definition=min(best, key=best.get),
        dchi_check=dchi,
        dchi_reproducible=bool(all(abs(v["published"] - v["recomputed"]) <= 0.002 for v in dchi.values())),
        composite_lstsq_on_t_and_dchi=dict(
            intercept=float(coef[0]), coef_t=float(coef[1]), coef_dchi=float(coef[2]),
            r2=r2, residual_max=float(np.max(np.abs(y - pred)))),
        composite_lstsq_on_dchi_only=dict(intercept=float(c1[0]), coef_dchi=float(c1[1]), r2=r2_1),
    )


# --------------------------------------------------------------------------
def screen(formulas) -> pd.DataFrame:
    rows = []
    for f in formulas:
        comp = Composition(f)
        rc = comp.reduced_composition
        d = {str(e): int(round(rc[e])) for e in rc}
        n_o = d.get("O", 0)
        n_c = sum(v for k, v in d.items() if k != "O")
        ox, q = oxidation_states(f)
        t, note = goldschmidt(f)
        req_red, assign, red_note = reduction_requirement(f)
        rows.append(dict(
            formula=comp.reduced_formula,
            n_cations=n_c, n_oxygen=n_o,
            anonymous_formula=rc.anonymized_formula,
            oxidation_states=json.dumps(ox) if ox else "",
            charge_sum=q,
            charge_balanced_standard_states=(q == 0) if q is not None else None,
            requires_reduced_cation=req_red,
            balanced_assignment=json.dumps(assign) if assign else "",
            redox_note=red_note,
            electronegativity_diff=round(electronegativity_diff(f), 4),
            goldschmidt_t=round(t, 4) if t is not None else np.nan,
            goldschmidt_applicable=(t is not None),
            goldschmidt_note=note,
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    cand = pd.read_csv(os.path.join(PROJ, "composition_cvae_generated_candidates.csv"))
    formulas = sorted(cand[cand["valid"] == True]["canonical_formula"].unique().tolist())

    f = run_forensics()
    with open(os.path.join(RESULTS, "a4_descriptor_forensics.json"), "w") as fh:
        json.dump(f, fh, indent=2)
    print("Delta-chi reproducible:", f["dchi_reproducible"])
    print("t reproducible by any tested definition:", f["t_reproducible"],
          "(best:", f["t_best_definition"], "max abs err",
          round(min(f["t_definition_max_abs_error"].values()), 4), ")")
    print("composite score ~ a + b*t + c*dchi  R2 =",
          round(f["composite_lstsq_on_t_and_dchi"]["r2"], 5),
          "max resid", round(f["composite_lstsq_on_t_and_dchi"]["residual_max"], 5))

    df = screen(formulas)
    df.to_csv(os.path.join(RESULTS, "a4_descriptor_screen_all_candidates.csv"), index=False)
    print(f"\nScreened {len(df)} unique valid candidates.")
    print("Goldschmidt t defined for:", int(df["goldschmidt_applicable"].sum()), "of", len(df))
    print("Charge-balanced with STANDARD states:", int((df["charge_balanced_standard_states"]==True).sum()), "of", len(df))
    print("REQUIRE a reduced cation (Ti3+ or Sn2+):", int((df["requires_reduced_cation"]==True).sum()))
    print(df[df["requires_reduced_cation"]==True]["redox_note"].value_counts().to_string())
