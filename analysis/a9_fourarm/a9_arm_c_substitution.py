"""a9_arm_c_substitution.py -- arm C: data-mined ion substitution into known compounds.

Stands in for the ion-exchange baseline of Szymanski & Bartel (Mater. Horiz. 2025).
Design (every choice is listed here and repeated in a9_REPORT.md):

  C1  Parent pool.  Every DISTINCT canonical formula in the project's own training table
      (load_material_table, the 25,148-row table the CVAE was trained on) that is a pure
      oxide: pymatgen's most probable oxidation-state guess
      (Composition.oxi_state_guesses(), ICSD-frequency ranked; first guess taken) assigns
      O2- and a positive state to every other element.  Parents containing N, F or S (the
      Castelli oxynitride / oxyfluoride / oxysulfide rows; pymatgen sometimes assigns N a
      positive state, so these are excluded by element), with no guess, or with a
      species absent from the Hautier lambda table are excluded.
  C2  Substitution model.  pymatgen SubstitutionProbability (Hautier et al., Inorg. Chem.
      2011, 50, 656), default lambda table, alpha = -5.  For a parent cation species s
      the probability of replacing it by t is cond_prob(t, s) = exp(lambda(s,t)) /
      sum_u exp(lambda(s,u)); t = s (no substitution) is allowed at each site.  The joint
      probability of a substitution tuple is the product over the parent's cation
      species, exactly as pymatgen's SubstitutionPredictor scores a species list.
  C3  Replacement ions.  Restricted to positive-valence species of the 40 non-O elements
      of COMMON_CERAMIC_ELEMENTS (94 species in the lambda table) -- the same element pool
      arms A and B draw from.  A parent ion is always allowed to stay unchanged, even if
      its element is not on the whitelist (the filter then rejects it, as it would for
      any arm).  Oxygen is never substituted.
  C4  Threshold.  Tuples whose joint probability is below 1e-3 are discarded (the
      SubstitutionPredictor default threshold), as are tuples mapping two sites onto the
      same species (pymatgen's rule) and the all-identity tuple (no substitution).
  C5  Charge.  Ion exchange keeps the anion framework, so the tuple must be charge
      neutral with the parent's own oxygen count:  sum_i n_i q(t_i) = 2 n_O.
      Aliovalent substitutions are allowed only in compensating combinations.
  C6  Number of substitutions.  Not fixed: it is whatever the Hautier joint model gives
      after C3-C5 (one, two, ... sites), sampled in proportion to the joint probability.
  C7  Sampling.  For each proposal: parent drawn uniformly (with replacement) from the
      parents that have at least one admissible tuple; tuple drawn with probability
      proportional to its joint probability among that parent's admissible tuples.
      numpy default_rng(seed), seeds 0-4, 20,000 proposals per seed.
  C8  Filter.  The substituted formula (parent counts, new elements, parent O count) is
      converted to the 77-element fraction vector with the project's own
      composition_vector() and passed to decode_composition_vector, exactly as the other
      arms.  A formula containing an element outside the 77-element vocabulary cannot be
      represented in that vector (composition_vector would silently drop it); such
      proposals are booked invalid with status "element outside 77-element vocabulary"
      instead of being silently truncated.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys
sys.path.insert(0, _CCID_ROOT + "/analysis/a9_fourarm")
import a9_common  # noqa: E402,F401  (sets sys.path)

import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Species
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionProbability

import a2_common as C
from composition_cvae_common import canonical_formula, formula_from_counts

THRESHOLD = 1e-3
PARENT_CACHE = a9_common.A9_DIR / "a9_arm_c_parents.json"


NON_OXIDE_ANIONS = {"N", "F", "S"}   # the anions of the Castelli oxynitride/-fluoride/-sulfide rows


def _parent_species(formula: str):
    comp = Composition(formula)
    if {str(e) for e in comp.elements} & NON_OXIDE_ANIONS:
        return None, "contains N/F/S (not a pure oxide)"
    guesses = comp.oxi_state_guesses()
    if not guesses:
        return None, "no oxidation-state guess"
    g = guesses[0]
    if abs(g.get("O", 0) + 2) > 1e-9:
        return None, "O not assigned 2-"
    sites = []
    for el, q in g.items():
        if el == "O":
            continue
        if q <= 0 or abs(q - round(q)) > 1e-9:
            return None, "non-oxide anion or fractional state"
        sites.append((el, int(round(q)), float(comp[el])))
    n_o = float(comp["O"])
    return dict(sites=sites, n_O=n_o), "ok"


def _admissible_tuples(sites, n_o, p, repl):
    """Enumerate every substitution tuple with joint prob >= THRESHOLD (branch-and-bound,
    the same pruning SubstitutionPredictor.list_prediction uses), then apply C4/C5."""
    old = [Species(el, q) for el, q, _ in sites]
    counts = [n for _, _, n in sites]
    options = []
    for s in old:
        opts = {str(s): p.cond_prob(s, s)}
        for t in repl:
            opts[str(t)] = p.cond_prob(t, s)
        options.append(sorted(opts.items(), key=lambda kv: -kv[1]))
    best = [o[0][1] for o in options]
    out = []

    def rec(i, prob, chosen):
        if i == len(old):
            if prob < THRESHOLD:
                return
            if len(set(chosen)) != len(chosen):
                return
            if all(c == str(s) for c, s in zip(chosen, old)):
                return
            charge = sum(n * Species.from_str(c).oxi_state for n, c in zip(counts, chosen))
            if abs(charge - 2 * n_o) > 1e-6:
                return
            out.append((tuple(chosen), prob))
            return
        rest = float(np.prod(best[i + 1:])) if i + 1 < len(old) else 1.0
        for name, pr in options[i]:
            if prob * pr * rest < THRESHOLD:
                break                      # options sorted descending: prune the rest
            rec(i + 1, prob * pr, chosen + [name])

    rec(0, 1.0, [])
    return out


def build_parents(force: bool = False) -> dict:
    if PARENT_CACHE.exists() and not force:
        return json.loads(PARENT_CACHE.read_text())
    table, _ = C.get_known()
    distinct = table.drop_duplicates("canonical_formula")
    p = SubstitutionProbability()
    wl = sorted(C.COMMON_CERAMIC_ELEMENTS - {"O"})
    repl = sorted([s for s in p.species if s.oxi_state > 0 and str(s.element) in wl], key=str)
    reasons: dict[str, int] = {}
    parents = []
    for f, src in zip(distinct["canonical_formula"], distinct["source_dataset"]):
        info, why = _parent_species(f)
        if info is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        if any(Species(el, q) not in p.species for el, q, _ in info["sites"]):
            reasons["species not in Hautier lambda table"] = reasons.get(
                "species not in Hautier lambda table", 0) + 1
            continue
        tup = _admissible_tuples(info["sites"], info["n_O"], p, repl)
        if not tup:
            reasons["no admissible substitution (threshold/charge)"] = reasons.get(
                "no admissible substitution (threshold/charge)", 0) + 1
            continue
        parents.append(dict(parent=f, source=src, sites=info["sites"], n_O=info["n_O"],
                            tuples=[list(t) for t, _ in tup], probs=[pr for _, pr in tup]))
    out = dict(threshold=THRESHOLD, n_distinct_training_formulas=int(len(distinct)),
               n_parents=len(parents), excluded=reasons,
               n_replacement_species=len(repl), replacement_species=[str(s) for s in repl],
               parents=parents)
    PARENT_CACHE.write_text(json.dumps(out))
    return out


def substituted_formula(parent: dict, tup) -> str:
    counts: dict[str, float] = {}
    for (el, q, n), new in zip(parent["sites"], tup):
        e = Species.from_str(new).element.symbol
        counts[e] = counts.get(e, 0.0) + n
    counts["O"] = parent["n_O"]
    # integer counts in pymatgen order is not needed; the vector is fraction-based
    ints = {k: (int(v) if abs(v - round(v)) < 1e-9 else v) for k, v in counts.items()}
    return formula_from_counts(ints)


def generate(seed: int, n: int, parents: dict) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    P = parents["parents"]
    pidx = rng.integers(0, len(P), size=n)
    rows = []
    for k, i in enumerate(pidx):
        par = P[i]
        pr = np.asarray(par["probs"], float)
        j = rng.choice(len(pr), p=pr / pr.sum())
        tup = par["tuples"][j]
        f = substituted_formula(par, tup)
        olds = [str(Species(el, q)) for el, q, _ in par["sites"]]
        n_sub = sum(1 for o, t in zip(olds, tup) if o != t)
        rows.append(dict(sample_id=k, parent=par["parent"], parent_source=par["source"],
                         substitution=";".join(f"{o}>{t}" for o, t in zip(olds, tup) if o != t),
                         n_substituted_sites=n_sub, joint_prob=float(pr[j]),
                         pre_filter_formula=f,
                         pre_filter_canonical=canonical_formula(
                             {k2: float(v) for k2, v in Composition(f).get_el_amt_dict().items()})))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time
    t0 = time.time()
    par = build_parents(force=True)
    print(f"built parents in {time.time()-t0:.1f}s")
    print({k: v for k, v in par.items() if k != "parents" and k != "replacement_species"})
    nt = [len(p["tuples"]) for p in par["parents"]]
    print("admissible tuples per parent: median", np.median(nt), "max", max(nt))
    g = generate(0, 2000, par)
    print(g.head(10).to_string())
    print(g["n_substituted_sites"].value_counts().to_dict())
