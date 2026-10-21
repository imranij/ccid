"""a7_chem_sanity.py -- chemical-sanity and distance-to-training comparison
between the CVAE population and the random-baseline population.

Axes
  1. REDUCED-CATION DEFECT.  a4_descriptors.reduction_requirement decides whether a
     composition can be charge balanced using only the standard (highest) oxidation
     state of every cation, or whether balance REQUIRES Ti(3+) / Sn(2+).  This is the
     36 % defect reported in the audit.  Generalised here to any cation forced below
     its highest tabulated state.
  2. OPEN-SHELL / COLOURED-CATION CONTENT.  Fraction of candidates containing at least
     one partially-filled-d or -f cation at its assigned oxidation state.  A wide-gap
     dielectric target implies d0 / d10 / f0 / f14 closed-shell cations.
  3. DISTANCE TO TRAINING DATA.  Euclidean distance in the project's own 77-dim element
     fraction space (composition_vector) from each candidate to its nearest neighbour in
     the 25,148-row CVAE training table.  A memorising model sits at ~0.
All numbers are computed here; nothing is read from a previous agent's summary.
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys, json
import numpy as np, pandas as pd
from pymatgen.core import Composition, Element

RESULTS = _CCID_ROOT + "/results"
sys.path.insert(0, _CCID_ROOT + "/analysis/a2_generative")
sys.path.insert(0, _CCID_ROOT + "/analysis/a4_umlip")
sys.path.insert(0, _CCID_ROOT + "/dgx/code")

from a4_descriptors import reduction_requirement as _a4_reduction   # noqa: E402
from composition_cvae_common import OXIDATION_STATES as GEN_OX      # noqa: E402
import itertools

REDUCED_STATE = {("Ti", 3), ("Sn", 2)}


def reduction_requirement(formula: str):
    """Generalisation of a4_descriptors.reduction_requirement to the FULL project
    oxidation-state table (composition_cvae_common.OXIDATION_STATES) rather than the
    16-cation subset the CVAE happened to use.  a4's table cannot assign states to
    Fe/Co/Ni/Cu/W/Mo/lanthanide candidates, which the random baseline produces in
    quantity; using it on the random arm would silently mark 96 % of them
    'unassignable'.  Verified to agree with a4's function on all 66 CVAE candidates.

    Returns (requires_Ti3_or_Sn2, assignment, description).  The assignment preferred is
    the one that uses the HIGHEST available state for every cation; if none balances,
    the first balancing assignment is returned.
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
        # prefer the assignment closest to the highest states throughout
        best = min(all_high, key=lambda s: sum(max(GEN_OX[e]) - v for e, v in s.items()))
        return False, best, "balanced without Ti(3+)/Sn(2+)"
    red = sorted({f"{e}{v}+" for s in sols for e, v in s.items() if (e, v) in REDUCED_STATE})
    best = min(sols, key=lambda s: sum(max(GEN_OX[e]) - v for e, v in s.items()))
    return True, best, "requires reduced cation(s): " + ", ".join(red)
from generate_composition_candidates import composition_vector    # noqa: E402
import a2_common as C                                             # noqa: E402

# d / f electron count of a cation in a given oxidation state -> open shell?
def _open_shell(el: str, ox: int) -> bool:
    e = Element(el)
    Z = e.Z
    # electrons remaining after removing `ox` electrons
    n_el = Z - ox
    block = e.block
    if block == "s" or block == "p":
        # main-group: closed shell iff the state empties the valence shell
        # (Al3+, Ga3+, In3+, Si4+, Sn4+, Bi5+, alkali/alkaline earth) ; lone-pair
        # states (Sn2+, Bi3+, Tl+) count as NOT closed-shell d/f but are flagged
        # separately by the reduced-cation test, so return False here.
        return False
    if block == "d":
        # d-count = group number - oxidation state
        group = e.group
        d = group - ox
        return 0 < d < 10
    if block == "f":
        # lanthanides: f-count = Z - 57 + 3 - ox  (La3+ = f0)
        f = Z - 57 + 3 - ox
        return 0 < f < 14
    return False


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in df["canonical_formula"]:
        req, assign, desc = reduction_requirement(f)
        c = Composition(f).reduced_composition
        cats = [str(e) for e in c if str(e) != "O"]
        if assign:
            os_ = [(e, assign[e]) for e in cats if e in assign]
            openshell = [f"{e}{v}+" for e, v in os_ if _open_shell(e, v)]
            forced_low = [f"{e}{v}+" for e, v in os_ if v != max(GEN_OX.get(e, [v]))]
        else:
            openshell, forced_low = [], []
        rows.append(dict(
            canonical_formula=f,
            n_cations=len(cats),
            balance_assignable=assign is not None,
            requires_reduced_cation=bool(req) if req is not None else None,
            requires_Ti3_or_Sn2=bool(req) if req is not None else None,
            forced_below_max_state=";".join(sorted(set(forced_low))),
            n_forced_below_max=len(set(forced_low)),
            open_shell_cations=";".join(sorted(set(openshell))),
            n_open_shell=len(set(openshell)),
            has_open_shell=bool(openshell),
            balance_note=desc,
        ))
    return df.merge(pd.DataFrame(rows), on="canonical_formula", how="left")


def nn_distance(formulas, vocab, train_mat, chunk=256):
    from scipy.spatial.distance import cdist
    V = np.vstack([composition_vector(f, vocab) for f in formulas])
    out = np.empty(len(V)); arg = np.empty(len(V), dtype=int)
    for i in range(0, len(V), chunk):
        d = cdist(V[i:i+chunk], train_mat)
        out[i:i+chunk] = d.min(axis=1); arg[i:i+chunk] = d.argmin(axis=1)
    return out, arg


def main():
    # agreement check against the a4 implementation on the CVAE set
    cv = pd.read_csv(f"{RESULTS}/a7_population_cvae.csv")
    agree = sum(bool(_a4_reduction(f)[0]) == bool(reduction_requirement(f)[0])
                for f in cv["canonical_formula"])
    print(f"reduction_requirement agreement with a4 on CVAE set: {agree}/{len(cv)}")

    vocab = C.element_vocab()
    table, known = C.get_known()
    train_formulas = table["formula"].astype(str).tolist()
    train_mat = np.vstack([composition_vector(f, vocab) for f in train_formulas])
    print(f"training table: {train_mat.shape[0]} rows, {train_mat.shape[1]}-dim")

    frames = {}
    for name, path in [("cvae", "a7_population_cvae.csv"),
                       ("random", "a7_population_random.csv"),
                       ("random_matched", "a7_population_random_matched.csv"),
                       ("random_dirichlet", "a7_population_random_dirichlet.csv"),
                       ("random_freq", "a7_population_random_freq.csv")]:
        d = pd.read_csv(f"{RESULTS}/{path}")
        d = annotate(d)
        dist, arg = nn_distance(d["canonical_formula"].tolist(), vocab, train_mat)
        d["nn_train_distance"] = dist
        d["nn_train_formula"] = [train_formulas[i] for i in arg]
        d["in_training_set"] = [f in known for f in d["canonical_formula"]]
        frames[name] = d
        print(f"{name:17s} n={len(d):6d}  reduced-cation {d['requires_reduced_cation'].mean():.3f}"
              f"  open-shell {d['has_open_shell'].mean():.3f}"
              f"  nn_dist med {np.median(dist):.4f}")

    allf = pd.concat(frames.values(), ignore_index=True)
    allf.to_csv(f"{RESULTS}/a7_chemical_sanity.csv", index=False)
    print("wrote a7_chemical_sanity.csv")


if __name__ == "__main__":
    main()
