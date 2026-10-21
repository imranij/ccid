"""a9_energetics_spinel_fix.py -- rebuild, correctly, the spinel hypotheses that a4's
builder produced with the wrong composition (see a9_energetics_hypothesis_audit.py), and
relax them with the unchanged protocol.

The a4 spinel branch does: B,C -> the four 16d 'Al' sites, then every 'Mg' -> A.  If B or
C is Mg, the new Mg on 16d is overwritten too.  Here the identical template
(primitive(proto_spinel(None))), the identical site assignment and the identical tag are
used, but substitution is done BY SITE INDEX, so it cannot collide.  The composition of
every rebuilt cell is asserted before relaxation.

CHGNet: a4_relax.relax(fmax=0.10, steps=150) on each rebuilt hypothesis.
MACE  : refinement of the lowest-energy VALID CHGNet hypothesis per affected formula
        (pyxtal or rebuilt spinel), from the CHGNet geometry, relax(fmax=0.10, steps=120).
Outputs: results/a9_hypothesis_relaxations_fix_{chgnet,mace}.csv,
         CIFs in results/a9_structures/fix/.  a4 and a7 files are not touched.
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import os, re, sys, time, warnings
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, _CCID_ROOT + "/analysis/a4_umlip")
import torch; torch.set_num_threads(1)
from pymatgen.core import Composition, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from a4_prototypes import proto_spinel
from a4_relax import primitive, relax, spacegroup_of

RESULTS = _CCID_ROOT + "/results"
CIFDIR = os.path.join(RESULTS, "a9_structures", "fix")
SRC = {"A": "a7_hypothesis_relaxations_chgnet.csv", "B": "a9_hypothesis_relaxations_B_chgnet.csv",
       "C": "a9_hypothesis_relaxations_C_chgnet.csv", "D": "a9_hypothesis_relaxations_D_chgnet.csv",
       "CVAE_dep": "a4_hypothesis_relaxations_chgnet.csv"}


def spinel_fixed(A, B, C):
    sp = primitive(proto_spinel(None))
    syms = sp.get_chemical_symbols()
    mg = [j for j, s in enumerate(syms) if s == "Mg"]
    al = [j for j, s in enumerate(syms) if s == "Al"]
    new = list(syms)
    for j in al[:2]:
        new[j] = B
    for j in al[2:]:
        new[j] = C
    for j in mg:
        new[j] = A
    sp.set_chemical_symbols(new)
    return sp


def row(f, tag, calc, atoms, r, e, conv, n, err, t0):
    nat = len(atoms)
    sg, sgn = spacegroup_of(r) if np.isfinite(e) else ("unknown", 0)
    return dict(formula=f, tag=tag, calc=calc, natoms=nat, energy_eV=e,
                energy_per_atom=e / nat if np.isfinite(e) else np.nan, converged=conv, nsteps=n,
                spacegroup=sg, spacegroup_number=sgn,
                volume_per_atom=r.get_volume() / nat if np.isfinite(e) else np.nan,
                walltime_s=round(time.time() - t0, 1), error=err)


def main():
    os.makedirs(CIFDIR, exist_ok=True)
    audit = pd.read_csv(f"{RESULTS}/a9_energetics_hypothesis_audit.csv")
    bad = audit[audit["valid"] == False]
    rows_c, rows_m = [], []
    for (arm, f), g in bad.groupby(["arm", "formula"]):
        want = Composition(f).reduced_composition
        for tag in g["tag"]:
            m = re.fullmatch(r"spinel_([A-Z][a-z]?)\[([A-Z][a-z]?)([A-Z][a-z]?)\]O4", tag)
            assert m, tag
            atoms = spinel_fixed(*m.groups())
            got = Composition(atoms.get_chemical_formula()).reduced_composition
            assert got == want, (f, tag, got)
            t0 = time.time()
            r, e, conv, n, err = relax(atoms, "chgnet", fmax=0.10, steps=150)
            rows_c.append(dict(arm=arm, **row(f, tag, "chgnet", atoms, r, e, conv, n, err, t0)))
            AseAtomsAdaptor.get_structure(r).to(filename=os.path.join(CIFDIR, f"a4_{f}_{tag}_chgnet.cif"))
            print(f"[fix chgnet] {arm} {f} {tag} E/at={e/len(atoms):.4f} conv={conv}", flush=True)
        pd.DataFrame(rows_c).to_csv(f"{RESULTS}/a9_hypothesis_relaxations_fix_chgnet.csv", index=False)
        # best VALID CHGNet hypothesis for this formula -> MACE refinement
        h = pd.read_csv(f"{RESULTS}/{SRC[arm]}")
        h = h[(h["formula"] == f) & (~h["tag"].isin(g["tag"]))]
        h["energy_per_atom"] = pd.to_numeric(h["energy_per_atom"], errors="coerce")
        cand = pd.concat([h, pd.DataFrame([r_ for r_ in rows_c if r_["formula"] == f])], ignore_index=True)
        cand = cand[np.isfinite(cand["energy_per_atom"])]
        b = cand.loc[cand["energy_per_atom"].idxmin()]
        tag = b["tag"]
        if tag in set(g["tag"]):
            cif = os.path.join(CIFDIR, f"a4_{f}_{tag}_chgnet.cif")
        elif arm in ("A", "CVAE_dep"):
            cif = os.path.join(RESULTS, "a4_structures", f"a4_{f}_{tag}_chgnet.cif")
        else:
            cif = os.path.join(RESULTS, "a9_structures", arm, f"a4_{f}_{tag}_chgnet.cif")
        atoms = AseAtomsAdaptor.get_atoms(Structure.from_file(cif))
        assert Composition(atoms.get_chemical_formula()).reduced_composition == want
        t0 = time.time()
        r, e, conv, n, err = relax(atoms, "mace", fmax=0.10, steps=120)
        rows_m.append(dict(arm=arm, **row(f, tag, "mace", atoms, r, e, conv, n, err, t0)))
        AseAtomsAdaptor.get_structure(r).to(filename=os.path.join(CIFDIR, f"a4_{f}_{tag}_mace.cif"))
        print(f"[fix mace] {arm} {f} best valid CHGNet = {tag}; MACE E/at={e/len(atoms):.4f} conv={conv}", flush=True)
        pd.DataFrame(rows_m).to_csv(f"{RESULTS}/a9_hypothesis_relaxations_fix_mace.csv", index=False)


if __name__ == "__main__":
    main()
