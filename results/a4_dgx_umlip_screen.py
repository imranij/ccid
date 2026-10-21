#!/usr/bin/env python
"""
a4_dgx_umlip_screen.py
======================
Full-scale universal-MLIP stability screen for the Composition-CVAE candidates of
MTCOMM-D-26-04679, written to be run by the authors on their own DGX hardware.

WHY THIS SCRIPT EXISTS
----------------------
The analysis delivered with the revision was produced on a 2-CPU sandbox with no access
to any structure or thermodynamics database (every OPTIMADE endpoint was blocked). Two
things therefore could NOT be done there and CAN be done on the DGX:

  1. A proper structure search. The sandbox run used a small, bounded set of prototype
     substitutions plus a handful of pyxtal random-symmetry trials. On a GPU you can
     afford hundreds of random trials per composition, which materially tightens the
     upper bound on each candidate's ground-state energy.
  2. A true energy above the convex hull. With a Materials Project API key, this script
     pulls the competing phases for each chemical system and builds the hull with the
     SAME potential used for the candidates, which is what makes E_hull internally
     consistent.

USAGE
-----
    pip install mace-torch chgnet pyxtal pymatgen mp-api ase
    export MP_API_KEY=...            # optional; without it, binary-oxide references only
    python a4_dgx_umlip_screen.py --candidates composition_cvae_generated_candidates.csv \
                                  --out results/ --device cuda --n-random 200 --calc mace

WHAT IT WRITES
--------------
    <out>/umlip_hypothesis_relaxations_<calc>.csv   every relaxed trial structure
    <out>/umlip_energetics_<calc>.csv               best structure + energetics per candidate
    <out>/umlip_hull_<calc>.csv                     E_above_hull, if MP_API_KEY is set
    <out>/structures/*.cif                          every relaxed structure

HONESTY NOTE, TO BE CARRIED INTO THE MANUSCRIPT
-----------------------------------------------
The minimum energy over any finite set of structure hypotheses is an UPPER BOUND on the
true ground-state energy, never a proof of stability. A uMLIP screen ranks and triages;
it does not replace DFT. Use a4_dft_inputs.py to generate DFT inputs for whatever
survives this screen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Potentials
# ---------------------------------------------------------------------------
def build_calculator(name: str, device: str):
    if name == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model="medium", default_dtype="float64", device=device)
    if name == "chgnet":
        from chgnet.model import CHGNetCalculator
        return CHGNetCalculator(use_device=device)
    if name == "matgl":
        import matgl
        from matgl.ext.ase import PESCalculator
        return PESCalculator(matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES"))
    if name == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator("7net-0", device=device)
    raise ValueError(f"unknown calculator {name}")


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------
def relax(atoms, calc, fmax=0.02, steps=500):
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    a = atoms.copy()
    a.calc = calc
    opt = FIRE(FrechetCellFilter(a), logfile=None)
    try:
        opt.run(fmax=fmax, steps=steps)
    except Exception as e:
        return a, float("nan"), False, 0, f"{type(e).__name__}: {e}"
    e = float(a.get_potential_energy())
    return a, e, bool(opt.converged()), opt.get_number_of_steps(), ""


# ---------------------------------------------------------------------------
# Structure hypotheses
# ---------------------------------------------------------------------------
PYXTAL_SPACEGROUPS = list(range(1, 231))


def random_hypotheses(formula: str, n: int, max_atoms: int, seed: int = 0):
    """Random symmetric structures at the candidate stoichiometry (pyxtal)."""
    from pyxtal import pyxtal as PyXtal
    from pymatgen.core import Composition
    from pymatgen.io.ase import AseAtomsAdaptor
    rc = Composition(formula).reduced_composition
    species = [str(e) for e in rc]
    counts = [int(round(rc[e])) for e in rc]
    rng = np.random.default_rng(seed)
    out, tries = [], 0
    while len(out) < n and tries < n * 20:
        tries += 1
        sg = int(rng.integers(1, 231))
        mult = int(rng.integers(1, 4))
        ni = [c * mult for c in counts]
        if sum(ni) > max_atoms:
            continue
        try:
            x = PyXtal()
            x.from_random(3, sg, species, ni)
            if not x.valid:
                continue
            out.append((f"pyxtal_sg{sg}_z{mult}_{len(out)}",
                        AseAtomsAdaptor.get_atoms(x.to_pymatgen())))
        except Exception:
            continue
    return out


def prototype_hypotheses(formula: str):
    """Stoichiometry-matched prototype substitutions.

    Uses the same prototype library as the sandbox analysis (a4_prototypes.py /
    a4_hypotheses.py, shipped alongside this script). If those modules are not on the
    path the script falls back to random search only, and says so.
    """
    try:
        from a4_hypotheses import prototype_hypotheses as ph
        return ph(formula)
    except Exception as e:
        print(f"  [warn] prototype library unavailable ({e}); random search only", flush=True)
        return []


# ---------------------------------------------------------------------------
# Hull construction (requires MP_API_KEY)
# ---------------------------------------------------------------------------
def hull_for_systems(systems, calc, out_csv, max_atoms=80):
    """Pull competing phases from Materials Project, RE-RELAX them with the same
    potential, and build a convex hull that is internally consistent with the candidate
    energies. Returns {chemsys: PhaseDiagram}."""
    from mp_api.client import MPRester
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
    from pymatgen.io.ase import AseAtomsAdaptor

    key = os.environ.get("MP_API_KEY")
    if not key:
        print("MP_API_KEY not set -- skipping hull construction.", flush=True)
        return {}

    pds, rows = {}, []
    with MPRester(key) as mpr:
        for sysname in systems:
            els = sysname.split("-")
            try:
                docs = mpr.materials.summary.search(
                    chemsys="-".join(sorted(els)), fields=["material_id", "formula_pretty", "structure"])
            except Exception as e:
                print(f"  [hull] {sysname}: MP query failed: {e}", flush=True)
                continue
            entries = []
            for d in docs:
                st = d.structure
                if len(st) > max_atoms:
                    continue
                a = AseAtomsAdaptor.get_atoms(st)
                _, e, conv, n, err = relax(a, calc)
                if not np.isfinite(e):
                    continue
                entries.append(PDEntry(st.composition, e))
                rows.append(dict(chemsys=sysname, material_id=d.material_id,
                                 formula=d.formula_pretty, natoms=len(st),
                                 energy_eV=e, energy_per_atom=e / len(st), converged=conv))
            if len(entries) >= len(els) + 1:
                try:
                    pds[sysname] = PhaseDiagram(entries)
                except Exception as e:
                    print(f"  [hull] {sysname}: PhaseDiagram failed: {e}", flush=True)
            print(f"  [hull] {sysname}: {len(entries)} competing phases relaxed", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pds


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True,
                    help="composition_cvae_generated_candidates.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--calc", default="mace", choices=["mace", "chgnet", "matgl", "sevennet"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-random", type=int, default=200,
                    help="pyxtal random trials per candidate (200+ recommended on GPU)")
    ap.add_argument("--max-atoms", type=int, default=60)
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--hull", action="store_true", help="build MP hulls (needs MP_API_KEY)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N candidates")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cifdir = os.path.join(args.out, "structures")
    os.makedirs(cifdir, exist_ok=True)

    from pymatgen.core import Composition
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    df = pd.read_csv(args.candidates)
    formulas = sorted(df[df["valid"] == True]["canonical_formula"].unique().tolist())
    if args.limit:
        formulas = formulas[:args.limit]
    print(f"{len(formulas)} unique valid candidates; calculator={args.calc} device={args.device}")

    calc = build_calculator(args.calc, args.device)

    hyp_csv = os.path.join(args.out, f"umlip_hypothesis_relaxations_{args.calc}.csv")
    rows = pd.read_csv(hyp_csv).to_dict("records") if os.path.exists(hyp_csv) else []
    done = {(r["formula"], r["tag"]) for r in rows}

    for fi, f in enumerate(formulas, 1):
        hyps = prototype_hypotheses(f) + random_hypotheses(f, args.n_random, args.max_atoms, seed=fi)
        print(f"[{fi}/{len(formulas)}] {f}: {len(hyps)} hypotheses", flush=True)
        for tag, atoms in hyps:
            if (f, tag) in done:
                continue
            t0 = time.time()
            r, e, conv, n, err = relax(atoms, calc, fmax=args.fmax, steps=args.steps)
            nat = len(atoms)
            sg, sgn = "unknown", 0
            if np.isfinite(e):
                try:
                    st = AseAtomsAdaptor.get_structure(r)
                    sga = SpacegroupAnalyzer(st, symprec=0.1)
                    sg, sgn = sga.get_space_group_symbol(), sga.get_space_group_number()
                    st.to(filename=os.path.join(cifdir, f"{f}_{tag}_{args.calc}.cif"))
                except Exception:
                    pass
            rows.append(dict(formula=f, tag=tag, calc=args.calc, natoms=nat, energy_eV=e,
                             energy_per_atom=e / nat if np.isfinite(e) else np.nan,
                             converged=conv, nsteps=n, spacegroup=sg, spacegroup_number=sgn,
                             volume_per_atom=r.get_volume() / nat if np.isfinite(e) else np.nan,
                             walltime_s=round(time.time() - t0, 1), error=err))
            pd.DataFrame(rows).to_csv(hyp_csv, index=False)

    # ---- best structure per candidate ----
    h = pd.DataFrame(rows)
    h = h[np.isfinite(h["energy_per_atom"])]
    best = h.loc[h.groupby("formula")["energy_per_atom"].idxmin()].copy()
    best.to_csv(os.path.join(args.out, f"umlip_energetics_{args.calc}.csv"), index=False)
    print(f"Best structure per candidate written for {len(best)} candidates.")

    # ---- hull ----
    if args.hull:
        systems = sorted({"-".join(sorted({str(e) for e in Composition(f).elements}))
                          for f in formulas})
        pds = hull_for_systems(systems, calc,
                               os.path.join(args.out, f"umlip_competing_phases_{args.calc}.csv"))
        from pymatgen.analysis.phase_diagram import PDEntry
        hull_rows = []
        for _, r in best.iterrows():
            comp = Composition(r["formula"]).reduced_composition
            sysname = "-".join(sorted({str(e) for e in comp.elements}))
            pdg = pds.get(sysname)
            if pdg is None:
                hull_rows.append(dict(formula=r["formula"], e_above_hull=np.nan,
                                      note="no phase diagram for this chemical system"))
                continue
            entry = PDEntry(comp, r["energy_per_atom"] * comp.num_atoms)
            try:
                eah = pdg.get_e_above_hull(entry)
                hull_rows.append(dict(formula=r["formula"], e_above_hull=float(eah),
                                      note="same-potential hull"))
            except Exception as e:
                hull_rows.append(dict(formula=r["formula"], e_above_hull=np.nan, note=str(e)))
        pd.DataFrame(hull_rows).to_csv(
            os.path.join(args.out, f"umlip_hull_{args.calc}.csv"), index=False)
        print("Hull distances written.")

    print("Done. Remember: lowest energy over a finite hypothesis set is an UPPER BOUND, "
          "not a proof of stability.")


if __name__ == "__main__":
    main()
