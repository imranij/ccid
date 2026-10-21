"""
a4_relax.py -- uMLIP relaxation and energetics utilities.

Potentials used (both verified working offline after a one-time weight download):
  * MACE-MP-0 (mace-torch), 'medium' model  -- primary
  * CHGNet v0.3.0                            -- independent cross-check

Both are trained on Materials Project PBE(+U) total energies, so their absolute
energies are on (approximately) the MP PBE scale and energy DIFFERENCES between
compounds built from the same elements are meaningful.
"""
from __future__ import annotations

import warnings
import numpy as np
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from ase.constraints import FixSymmetry
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings("ignore")

_CALCS = {}


def get_calc(name: str):
    if name in _CALCS:
        return _CALCS[name]
    if name == "mace":
        from mace.calculators import mace_mp
        # float32: this sandbox has 2 CPU cores and float64 MACE is ~2x slower. MACE is
        # used here to cross-check ENERGY DIFFERENCES at the 0.01 eV/atom level, and
        # float32 round-off is ~1e-4 eV/atom, far below the physical error of the model.
        c = mace_mp(model="medium", default_dtype="float32", device="cpu")
    elif name == "chgnet":
        from chgnet.model import CHGNetCalculator
        c = CHGNetCalculator()
    else:
        raise ValueError(name)
    _CALCS[name] = c
    return c


def primitive(atoms):
    """Reduce an ase.Atoms to its primitive cell via pymatgen."""
    try:
        s = AseAtomsAdaptor.get_structure(atoms)
        sga = SpacegroupAnalyzer(s, symprec=0.1)
        p = sga.get_primitive_standard_structure()
        return AseAtomsAdaptor.get_atoms(p)
    except Exception:
        return atoms


def relax(atoms, calc_name="mace", fmax=0.05, steps=250, fix_symmetry=False):
    """Variable-cell relaxation. Returns (relaxed_atoms, energy_eV, converged, nsteps)."""
    a = atoms.copy()
    a.calc = get_calc(calc_name)
    if fix_symmetry:
        a.set_constraint(FixSymmetry(a))
    filt = FrechetCellFilter(a)
    opt = FIRE(filt, logfile=None)
    try:
        opt.run(fmax=fmax, steps=steps)
        conv = bool(opt.converged())
    except Exception as e:
        return a, float("nan"), False, 0, f"{type(e).__name__}: {e}"
    e = float(a.get_potential_energy())
    if not np.isfinite(e):
        return a, float("nan"), False, opt.get_number_of_steps(), "non-finite energy"
    a.set_constraint()
    return a, e, conv, opt.get_number_of_steps(), ""


def spacegroup_of(atoms, symprec=0.1):
    try:
        s = AseAtomsAdaptor.get_structure(atoms)
        sga = SpacegroupAnalyzer(s, symprec=symprec)
        return sga.get_space_group_symbol(), sga.get_space_group_number()
    except Exception:
        return "unknown", 0
