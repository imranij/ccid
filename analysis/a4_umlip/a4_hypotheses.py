"""
a4_hypotheses.py -- map a composition-only candidate onto a bounded set of trial structures.

Two families, as described in a4_prototypes:
  (A) prototype substitution (stoichiometry-matched textbook prototypes)
  (B) pyxtal random symmetric structures (covers stoichiometries with no clean prototype)

Every hypothesis is tagged with its provenance so the report can state exactly which
structural assumption produced each energy.
"""
from __future__ import annotations

import itertools
import warnings
import numpy as np
from math import gcd
from functools import reduce

from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor

from a4_prototypes import (
    proto_pyrochlore, proto_defect_fluorite, proto_double_perovskite,
    proto_spinel, proto_k2nif4, proto_perovskite,
)
from a4_relax import primitive

warnings.filterwarnings("ignore")

MAX_ATOMS = 30


def cation_list(comp: Composition):
    """Return (list_of_cation_symbols_with_multiplicity, n_oxygen) for the reduced formula."""
    rc = comp.reduced_composition
    d = {str(e): int(round(rc[e])) for e in rc}
    n_o = d.pop("O", 0)
    cats = []
    for el, n in sorted(d.items()):
        cats += [el] * n
    return cats, n_o


def _sub(atoms, mapping):
    """Replace placeholder species in an ase.Atoms according to mapping {old: new}."""
    a = atoms.copy()
    syms = [mapping.get(s, s) for s in a.get_chemical_symbols()]
    a.set_chemical_symbols(syms)
    return a


def _orderings(items, cap):
    """Unique permutations of a multiset, capped."""
    seen, out = set(), []
    for p in itertools.permutations(items):
        if p in seen:
            continue
        seen.add(p)
        out.append(list(p))
        if len(out) >= cap:
            break
    return out


def _pyrochlore_hypotheses(cats, cap):
    """General A2B2O7 pyrochlore, including a MIXED B-site (e.g. Al4[Zr2Hf2]O14).

    The 22-atom primitive cell holds 4 A-sites and 4 B-sites = 2 formula units, so the
    composition is doubled and its 8 cations are partitioned 4/4 between the sites.
    The chemically sensible partition puts the four LARGEST cations on the 8-coordinate
    A-site; that partition is generated first, followed by alternatives up to `cap`.
    """
    from a4_prototypes import proto_pyrochlore_template, _R6
    out = []
    eight = list(cats) * 2                       # A2B2O7 -> A4B4O14
    try:
        tmpl = primitive(proto_pyrochlore_template())
    except Exception:
        return out
    syms = tmpl.get_chemical_symbols()
    a_idx = [i for i, s in enumerate(syms) if s == "Cs"]
    b_idx = [i for i, s in enumerate(syms) if s == "Xe"]
    if len(a_idx) != 4 or len(b_idx) != 4:
        return out

    order = sorted(range(8), key=lambda i: -_R6.get(eight[i], 0.75))
    partitions, seen = [], set()
    partitions.append((tuple(order[:4]), tuple(order[4:])))
    for combo in itertools.combinations(range(8), 4):
        A = tuple(sorted(eight[i] for i in combo))
        B = tuple(sorted(eight[i] for i in range(8) if i not in combo))
        if (A, B) in seen:
            continue
        seen.add((A, B))
        partitions.append((combo, tuple(i for i in range(8) if i not in combo)))
        if len(partitions) >= cap + 1:
            break

    used = set()
    for ai, bi in partitions:
        Agrp = tuple(sorted(eight[i] for i in ai))
        Bgrp = tuple(sorted(eight[i] for i in bi))
        key = (Agrp, Bgrp)
        if key in used:
            continue
        used.add(key)
        a = tmpl.copy()
        new = list(a.get_chemical_symbols())
        for slot, i in zip(a_idx, ai):
            new[slot] = eight[i]
        for slot, i in zip(b_idx, bi):
            new[slot] = eight[i]
        a.set_chemical_symbols(new)
        lab = "".join(Agrp) + "_" + "".join(Bgrp)
        out.append((f"pyrochlore_{lab}", a))
        if len(out) >= cap:
            break
    return out


def prototype_hypotheses(formula: str, cap_per_proto: int = 6):
    """Yield (tag, ase.Atoms) trial structures from stoichiometry-matched prototypes."""
    comp = Composition(formula)
    cats, n_o = cation_list(comp)
    n_c = len(cats)
    out = []

    # ---- M4O7 family: defect fluorite (always) + pyrochlore (when 2+2 orderable) ----
    if (n_c, n_o) == (4, 7):
        for i, order in enumerate(_orderings(cats, cap_per_proto)):
            out.append((f"defect_fluorite_{i}", proto_defect_fluorite(order)))
        out += _pyrochlore_hypotheses(cats, cap_per_proto)

    # ---- M4O6 family: ordered double perovskite A2BB'O6 ----
    if (n_c, n_o) == (4, 6):
        # choose which element sits on the 8c A-site (needs multiplicity 2)
        cnt = {e: cats.count(e) for e in set(cats)}
        combos = []
        for Ael, k in cnt.items():
            if k >= 2:
                rest = list(cats)
                rest.remove(Ael); rest.remove(Ael)
                if len(set(rest)) <= 2:
                    for B, Bp in _orderings(rest, 2):
                        combos.append((Ael, B, Bp))
        for i, (A, B, Bp) in enumerate(combos[:cap_per_proto]):
            try:
                out.append((f"dperovskite_{A}2{B}{Bp}O6", primitive(proto_double_perovskite(A, B, Bp))))
            except Exception:
                pass

    # ---- M3O4 family: spinel-derived (2 cations on 8a, 4 on 16d per primitive cell) ----
    if (n_c, n_o) == (3, 4):
        for i, order in enumerate(_orderings(cats, cap_per_proto)):
            A, B, C = order
            try:
                sp = primitive(proto_spinel(None))
                # spinel primitive: 2 Mg (8a) + 4 Al (16d) + 8 O  -> A2 B2 C2 O8 = 2x ABCO4
                syms = sp.get_chemical_symbols()
                al_idx = [j for j, s in enumerate(syms) if s == "Al"]
                new = list(syms)
                for j in al_idx[:2]:
                    new[j] = B
                for j in al_idx[2:]:
                    new[j] = C
                new = [A if s == "Mg" else s for s in new]
                sp.set_chemical_symbols(new)
                out.append((f"spinel_{A}[{B}{C}]O4", sp))
            except Exception:
                pass
        # K2NiF4 needs A2BO4; only applicable if a cation appears twice
        cnt = {e: cats.count(e) for e in set(cats)}
        for Ael, k in cnt.items():
            if k == 2:
                B = [e for e in cats if e != Ael]
                if len(B) == 1:
                    try:
                        out.append((f"k2nif4_{Ael}2{B[0]}O4", primitive(proto_k2nif4(Ael, B[0]))))
                    except Exception:
                        pass

    # ---- MO3 family: perovskite ----
    if n_c == 2 and n_o == 3:
        for i, (A, B) in enumerate(_orderings(cats, 2)):
            out.append((f"perovskite_{A}{B}O3", proto_perovskite(A, B)))

    return [(t, a) for t, a in out if len(a) <= MAX_ATOMS]


PYXTAL_SPACEGROUPS = [1, 2, 4, 9, 12, 14, 15, 19, 29, 33, 62, 63]


def pyxtal_hypotheses(formula: str, n: int = 4, seed: int = 0):
    """Random symmetric structures at the candidate stoichiometry (coarse structure search)."""
    from pyxtal import pyxtal as PyXtal
    comp = Composition(formula)
    rc = comp.reduced_composition
    species = [str(e) for e in rc]
    counts = [int(round(rc[e])) for e in rc]
    out = []
    rng = np.random.default_rng(seed)
    sgs = list(PYXTAL_SPACEGROUPS)
    rng.shuffle(sgs)
    for sg in sgs:
        if len(out) >= n:
            break
        for mult in (1, 2):
            ni = [c * mult for c in counts]
            if sum(ni) > MAX_ATOMS:
                continue
            try:
                x = PyXtal()
                x.from_random(3, sg, species, ni)
                if not x.valid:
                    continue
                a = AseAtomsAdaptor.get_atoms(x.to_pymatgen())
                if len(a) <= MAX_ATOMS:
                    out.append((f"pyxtal_sg{sg}_z{mult}", a))
                    break
            except Exception:
                continue
    return out


def all_hypotheses(formula: str, n_random: int = 4, seed: int = 0):
    h = prototype_hypotheses(formula)
    h += pyxtal_hypotheses(formula, n=n_random, seed=seed)
    return h
