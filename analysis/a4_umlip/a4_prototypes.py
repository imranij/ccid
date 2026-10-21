"""
a4_prototypes.py -- structure hypothesis generation for composition-only candidates.

The CVAE candidates of MTCOMM-D-26-04679 are compositions with no structure. To get
any physics-based energy we must first invent plausible structures. Two independent
families of hypothesis are built here:

  (A) PROTOTYPE SUBSTITUTION. Cations are substituted into textbook crystallographic
      prototypes whose stoichiometry matches the candidate. Prototype Wyckoff settings
      below are standard literature values; they are only *seeds* -- every structure is
      subsequently relaxed (cell + ions) with a uMLIP, so the starting internal
      coordinates and lattice constants need only be plausible, not accurate.

  (B) RANDOM SYMMETRIC SEARCH (pyxtal). Random structures are generated in a list of
      common oxide space groups at the candidate's stoichiometry, then relaxed.
      This covers stoichiometries with no clean prototype.

IMPORTANT INTERPRETATION NOTE: the minimum energy over any finite set of hypotheses is
an UPPER BOUND on the true ground-state energy. It is never a proof of stability. A
candidate that looks unstable under this bound is (modulo uMLIP error) genuinely
unstable; a candidate that looks stable may simply not have had its true ground state
sampled.
"""
from __future__ import annotations

import itertools
import numpy as np
from ase.spacegroup import crystal
from ase import Atoms
from pymatgen.core import Composition, Structure
from pymatgen.io.ase import AseAtomsAdaptor

# ---------------------------------------------------------------------------
# Ionic-radius-based lattice scaling helper
# ---------------------------------------------------------------------------
# Shannon-style effective ionic radii (CN 6, oxidation state as used in these oxides),
# in Angstrom. Standard tabulated values (Shannon 1976). Used ONLY to set a sensible
# starting cell volume before relaxation.
_R6 = {
    "Al": 0.535, "Ga": 0.620, "Sc": 0.745, "Y": 0.900, "La": 1.032,
    "Ti": 0.605, "Zr": 0.720, "Hf": 0.710, "Si": 0.400, "Sn": 0.690,
    "Nb": 0.640, "Ta": 0.640, "Ba": 1.350, "Sr": 1.180, "Ca": 1.000,
    "Mg": 0.720, "Zn": 0.740, "O": 1.400,
}


def _mean_cation_radius(symbols) -> float:
    r = [_R6.get(s, 0.75) for s in symbols if s != "O"]
    return float(np.mean(r)) if r else 0.75


def _rescale(atoms: Atoms, ref_mean_r: float, target_mean_r: float) -> Atoms:
    """Isotropically rescale a prototype cell to suit the substituted cation sizes."""
    f = (target_mean_r + 1.40) / (ref_mean_r + 1.40)
    atoms = atoms.copy()
    atoms.set_cell(atoms.get_cell() * f, scale_atoms=True)
    return atoms


# ---------------------------------------------------------------------------
# Prototype builders. Each returns an ase.Atoms with placeholder species that are
# substituted by the caller.  Reference lattice constants are the prototype's own.
# ---------------------------------------------------------------------------

def proto_pyrochlore(A: str, B: str, O: str = "O") -> Atoms:
    """A2B2O7 pyrochlore, Fd-3m (227), origin choice 2.
    A on 16d (1/2,1/2,1/2), B on 16c (0,0,0), O on 48f (x,1/8,1/8) x=0.33, O' on 8b (3/8,3/8,3/8).
    Reference: Y2Ti2O7, a = 10.10 A."""
    return crystal(
        symbols=[A, B, O, O],
        basis=[(0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (0.33, 0.125, 0.125), (0.375, 0.375, 0.375)],
        spacegroup=227,
        setting=2,
        cellpar=[10.10, 10.10, 10.10, 90, 90, 90],
    )


def proto_pyrochlore_template(O: str = "O") -> Atoms:
    """Pyrochlore Fd-3m with *placeholder* species Cs (16d, A-site) and Xe (16c, B-site).

    Neither Cs nor Xe occurs in any candidate here, so the placeholders can be safely
    substituted afterwards. The caller reduces to the 22-atom primitive cell
    (A4 B4 O14 = 2 formula units of A2B2O7) and then assigns real cations, which is
    what allows a MIXED B-site such as Zr2Hf2 -- the physically expected prototype for
    Al2ZrHfO7 -- to be represented at all.
    """
    return crystal(
        symbols=["Cs", "Xe", O, O],
        basis=[(0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (0.33, 0.125, 0.125), (0.375, 0.375, 0.375)],
        spacegroup=227,
        setting=2,
        cellpar=[10.10, 10.10, 10.10, 90, 90, 90],
    )


def proto_defect_fluorite(cations, O: str = "O") -> Atoms:
    """A2B2O7 as an ordered-vacancy defect fluorite.
    Conventional fluorite cell: cations on fcc 4a, anions on 8c. Removing one of the
    eight anions gives exactly M4O7. Reference a = 5.15 A (ZrO2 fluorite)."""
    assert len(cations) == 4
    cat_pos = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    o_pos = [(x, y, z) for x in (0.25, 0.75) for y in (0.25, 0.75) for z in (0.25, 0.75)]
    o_pos = o_pos[:-1]  # ordered oxygen vacancy -> M4O7
    syms = list(cations) + [O] * len(o_pos)
    pos = cat_pos + o_pos
    return Atoms(symbols=syms, scaled_positions=pos, cell=[5.15] * 3, pbc=True)


def proto_double_perovskite(A: str, B: str, Bp: str, O: str = "O") -> Atoms:
    """A2BB'O6 rock-salt-ordered double perovskite, Fm-3m (225).
    A on 8c (1/4,1/4,1/4), B on 4a (0,0,0), B' on 4b (1/2,1/2,1/2), O on 24e (x,0,0) x=0.25.
    Reference: Ba2MgWO6-type, a = 8.10 A."""
    return crystal(
        symbols=[A, B, Bp, O],
        basis=[(0.25, 0.25, 0.25), (0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.25, 0.0, 0.0)],
        spacegroup=225,
        cellpar=[8.10, 8.10, 8.10, 90, 90, 90],
    )


def proto_corundum_derived(cations, O: str = "O") -> Atoms:
    """A2BB'O6 corundum-derived (ilmenite-like ordering on the corundum cation sublattice).
    Corundum R-3c (167): M on 12c (0,0,0.3523), O on 18e (0.3064,0,1/4); a=4.76, c=12.99.
    Built in the hexagonal setting then cation-ordered by the caller."""
    return crystal(
        symbols=["Al", O],
        basis=[(0.0, 0.0, 0.3523), (0.3064, 0.0, 0.25)],
        spacegroup=167,
        cellpar=[4.7602, 4.7602, 12.9933, 90, 90, 120],
    )


def proto_spinel(cations, O: str = "O") -> Atoms:
    """AB2O4 spinel, Fd-3m (227) origin 2: A on 8a (1/8,1/8,1/8), B on 16d (1/2,1/2,1/2),
    O on 32e (u,u,u) u=0.2625. Reference MgAl2O4, a = 8.083 A."""
    return crystal(
        symbols=["Mg", "Al", O],
        basis=[(0.125, 0.125, 0.125), (0.5, 0.5, 0.5), (0.2625, 0.2625, 0.2625)],
        spacegroup=227,
        setting=2,
        cellpar=[8.083, 8.083, 8.083, 90, 90, 90],
    )


def proto_k2nif4(A: str, B: str, O: str = "O") -> Atoms:
    """A2BO4 K2NiF4 / Ruddlesden-Popper n=1, I4/mmm (139).
    A on 4e (0,0,z) z=0.355, B on 2a (0,0,0), O1 on 4c (0,1/2,0), O2 on 4e (0,0,z) z=0.16.
    Reference Sr2TiO4, a = 3.884, c = 12.60 A."""
    return crystal(
        symbols=[A, B, O, O],
        basis=[(0.0, 0.0, 0.355), (0.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.16)],
        spacegroup=139,
        cellpar=[3.884, 3.884, 12.60, 90, 90, 90],
    )


def proto_perovskite(A: str, B: str, O: str = "O") -> Atoms:
    """ABO3 cubic perovskite Pm-3m (221). Reference SrTiO3 a = 3.905 A."""
    return crystal(
        symbols=[A, B, O],
        basis=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.0)],
        spacegroup=221,
        cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
    )


# ---------------------------------------------------------------------------
# Binary-oxide reference structures (for decomposition energetics).
# All are standard experimental prototypes; each is relaxed with the same uMLIP
# before use, so these are seeds only.
# ---------------------------------------------------------------------------

def binary_oxide_seed(formula: str) -> Atoms | None:
    f = formula
    if f == "Al2O3":
        return crystal(["Al", "O"], [(0, 0, 0.3523), (0.3064, 0, 0.25)], spacegroup=167,
                       cellpar=[4.7602, 4.7602, 12.9933, 90, 90, 120])
    if f == "Ga2O3":  # alpha-Ga2O3, corundum type (metastable polymorph -- flagged)
        return crystal(["Ga", "O"], [(0, 0, 0.3554), (0.3061, 0, 0.25)], spacegroup=167,
                       cellpar=[4.9825, 4.9825, 13.433, 90, 90, 120])
    if f == "Sc2O3":
        return crystal(["Sc", "Sc", "O"], [(0.25, 0.25, 0.25), (-0.0345, 0, 0.25),
                                           (0.3907, 0.1520, 0.3800)], spacegroup=206,
                       cellpar=[9.845, 9.845, 9.845, 90, 90, 90])
    if f == "Y2O3":
        return crystal(["Y", "Y", "O"], [(0.25, 0.25, 0.25), (-0.0322, 0, 0.25),
                                         (0.3907, 0.1520, 0.3800)], spacegroup=206,
                       cellpar=[10.604, 10.604, 10.604, 90, 90, 90])
    if f == "La2O3":  # A-type hexagonal
        return crystal(["La", "O", "O"], [(1/3, 2/3, 0.2467), (1/3, 2/3, 0.6470), (0, 0, 0)],
                       spacegroup=164, cellpar=[3.937, 3.937, 6.130, 90, 90, 120])
    if f in ("ZrO2", "HfO2"):  # baddeleyite P2_1/c
        cp = [5.1505, 5.2116, 5.3173, 90, 99.23, 90] if f == "ZrO2" else \
             [5.1170, 5.1750, 5.2915, 90, 99.22, 90]
        m = "Zr" if f == "ZrO2" else "Hf"
        return crystal([m, "O", "O"], [(0.2758, 0.0411, 0.2082), (0.0703, 0.3359, 0.3406),
                                       (0.4423, 0.7549, 0.4789)], spacegroup=14, cellpar=cp)
    if f == "TiO2":  # rutile
        return crystal(["Ti", "O"], [(0, 0, 0), (0.3053, 0.3053, 0)], spacegroup=136,
                       cellpar=[4.5937, 4.5937, 2.9587, 90, 90, 90])
    if f == "SnO2":  # rutile (cassiterite)
        return crystal(["Sn", "O"], [(0, 0, 0), (0.3056, 0.3056, 0)], spacegroup=136,
                       cellpar=[4.7374, 4.7374, 3.1864, 90, 90, 90])
    if f == "SiO2":  # alpha-quartz
        return crystal(["Si", "O"], [(0.4697, 0, 1/3), (0.4135, 0.2669, 0.1191)],
                       spacegroup=152, cellpar=[4.9134, 4.9134, 5.4052, 90, 90, 120])
    if f == "Ti2O3":  # corundum type, the Ti(3+) binary oxide
        return crystal(["Ti", "O"], [(0, 0, 0.3459), (0.3106, 0, 0.25)], spacegroup=167,
                       cellpar=[5.158, 5.158, 13.611, 90, 90, 120])
    if f == "SnO":
        # Litharge-type SnO, P4/nmm. Built from EXPLICIT atomic positions rather than by
        # symmetry expansion: ASE's default origin choice for space group 129 placed the
        # atoms wrongly and the relaxation diverged to a positive energy. Here O forms a
        # square net at z = 0 and Sn sits alternately above and below it, which reproduces
        # the experimental Sn-O distance of 2.22 A.
        a_, c_ = 3.802, 4.836
        z = 0.2382
        return Atoms(symbols=["O", "O", "Sn", "Sn"],
                     scaled_positions=[(0, 0, 0), (0.5, 0.5, 0),
                                       (0, 0.5, z), (0.5, 0, -z)],
                     cell=[a_, a_, c_, 90, 90, 90], pbc=True)
    if f in ("BaO", "SrO", "CaO", "MgO"):  # rocksalt
        a = {"BaO": 5.539, "SrO": 5.160, "CaO": 4.8105, "MgO": 4.212}[f]
        m = f[:-1]
        return crystal([m, "O"], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
                       cellpar=[a, a, a, 90, 90, 90])
    return None


# Binary oxides for which no reliable seed structure could be built here.
# Nb2O5 and Ta2O5 have large, low-symmetry ground-state cells (e.g. H-Nb2O5, B-Ta2O5)
# that cannot be responsibly reconstructed from memory, and no structure database was
# reachable from this environment. Candidates requiring them are excluded from the
# decomposition-energy analysis and reported as such.
MISSING_BINARY_REFERENCES = {"Nb2O5", "Ta2O5"}

OXIDE_OF = {
    "Al": "Al2O3", "Ga": "Ga2O3", "Sc": "Sc2O3", "Y": "Y2O3", "La": "La2O3",
    "Zr": "ZrO2", "Hf": "HfO2", "Ti": "TiO2", "Sn": "SnO2", "Si": "SiO2",
    "Ba": "BaO", "Sr": "SrO", "Ca": "CaO", "Mg": "MgO",
    "Nb": "Nb2O5", "Ta": "Ta2O5",
}

# Binary oxide corresponding to a SPECIFIC cation oxidation state. Needed because 24 of
# the 66 candidates can only be charge-balanced with Ti(3+) or Sn(2+); referencing those
# against TiO2/SnO2 leaves the oxygen unbalanced and would require an O2 reference, which
# universal MLIPs reproduce poorly. Referencing them against Ti2O3 / SnO instead balances
# oxygen exactly and keeps the comparison free of any O2 energy.
OXIDE_OF_STATE = {
    ("Ti", 4): "TiO2", ("Ti", 3): "Ti2O3",
    ("Sn", 4): "SnO2", ("Sn", 2): "SnO",
}
