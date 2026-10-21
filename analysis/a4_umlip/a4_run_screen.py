"""
a4_run_screen.py -- uMLIP structure-hypothesis screen for the CVAE candidates.

Stage 1: build trial structures for every candidate (a4_hypotheses).
Stage 2: relax every trial with CHGNet (cheap), record energies.
Stage 3: re-relax the best trial per candidate with MACE-MP-0 (independent potential).
Stage 4: relax binary-oxide reference structures with BOTH potentials.
Stage 5: decomposition energy of each candidate into its binary oxides, same potential.

Results are checkpointed to CSV after every single relaxation, so a partial run is
always usable. Re-running resumes from the checkpoint.

Run:  python a4_run_screen.py [--tier A|ALL] [--calc chgnet|mace]
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import argparse
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor

from a4_relax import relax, spacegroup_of
from a4_hypotheses import all_hypotheses
from a4_prototypes import binary_oxide_seed, OXIDE_OF, OXIDE_OF_STATE, MISSING_BINARY_REFERENCES

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = _CCID_ROOT + "/results"
CIFDIR = os.path.join(RESULTS, "a4_structures")
CAND_CSV = _CCID_ROOT + "/dgx/outputs_dgx/composition_cvae_generated_candidates.csv"

# The 6 candidates of the manuscript's Table 12 plus the 10 top-ranked of Table 11,
# in canonical (pymatgen-reduced) form.
TIER_A = ["Al2BaHfO6", "Al2HfO7Zr", "Al2LaO6Sc", "Al2O7SiZr", "AlHfO5Sc", "AlO5ScZr",
          "CaHfO7Zr2", "AlBa2LaO5", "HfO7SrZr2", "AlBaLaO4", "CaO4SrZr", "AlBaO4Sc"]


def unique_candidates():
    d = pd.read_csv(CAND_CSV)
    v = d[d["valid"] == True]
    return sorted(v["canonical_formula"].unique().tolist())


# --------------------------------------------------------------------------
def hypothesis_table(formulas, seed=0):
    rows = []
    for f in formulas:
        try:
            hyps = all_hypotheses(f, n_random=4, seed=seed)
        except Exception as e:
            print(f"[hyp] {f}: FAILED {e}", flush=True)
            hyps = []
        if not hyps:
            print(f"[hyp] {f}: NO HYPOTHESES", flush=True)
        for tag, atoms in hyps:
            rows.append(dict(formula=f, tag=tag, natoms=len(atoms)))
    return pd.DataFrame(rows)


def run_candidates(formulas, calc_name, out_csv, seed=0, tag_filter=None):
    done = {}
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        for _, r in prev.iterrows():
            done[(r["formula"], r["tag"])] = True
        rows = prev.to_dict("records")
    else:
        rows = []

    os.makedirs(CIFDIR, exist_ok=True)
    for f in formulas:
        try:
            hyps = all_hypotheses(f, n_random=4, seed=seed)
        except Exception as e:
            print(f"[{calc_name}] {f}: hypothesis generation failed: {e}", flush=True)
            continue
        for tag, atoms in hyps:
            if tag_filter and not tag_filter(tag):
                continue
            if (f, tag) in done:
                continue
            t0 = time.time()
            r, e, conv, n, err = relax(atoms, calc_name, fmax=0.10, steps=150)
            nat = len(atoms)
            sg, sgn = spacegroup_of(r) if np.isfinite(e) else ("unknown", 0)
            rows.append(dict(formula=f, tag=tag, calc=calc_name, natoms=nat,
                             energy_eV=e, energy_per_atom=e / nat if np.isfinite(e) else np.nan,
                             converged=conv, nsteps=n, spacegroup=sg, spacegroup_number=sgn,
                             volume_per_atom=r.get_volume() / nat if np.isfinite(e) else np.nan,
                             walltime_s=round(time.time() - t0, 1), error=err))
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            if np.isfinite(e):
                try:
                    AseAtomsAdaptor.get_structure(r).to(
                        filename=os.path.join(CIFDIR, f"a4_{f}_{tag}_{calc_name}.cif"))
                except Exception:
                    pass
            print(f"[{calc_name}] {f:14s} {tag:26s} {nat:3d}at "
                  f"E/at={e/nat if np.isfinite(e) else float('nan'):8.4f} {sg:10s} "
                  f"{time.time()-t0:6.1f}s conv={conv} {err}", flush=True)
    return pd.DataFrame(rows)


def run_references(calc_name, out_csv):
    """Relax binary-oxide reference structures with the given potential."""
    done = set()
    rows = []
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(prev["oxide"])
        rows = prev.to_dict("records")
    os.makedirs(CIFDIR, exist_ok=True)
    for ox in sorted(set(OXIDE_OF.values()) | set(OXIDE_OF_STATE.values())):
        if ox in done or ox in MISSING_BINARY_REFERENCES:
            continue
        seed = binary_oxide_seed(ox)
        if seed is None:
            continue
        # Reduce to the primitive cell: the bixbyite (Ia-3) conventional cell is 80 atoms,
        # its primitive cell 40, which is several times cheaper and identical in energy/atom.
        # GUARD: accept the primitive cell only if it preserves the stoichiometry.
        # With symprec=0.1 the symmetry analyser once returned a 3-atom "SnO" cell whose
        # Sn:O ratio was not 1:1; relaxing that diverged to a nonsensical energy.
        from a4_relax import primitive as _prim
        p = _prim(seed)
        if len(p) < len(seed):
            want = Composition(ox).reduced_composition.as_dict()
            got = Composition(p.get_chemical_formula()).reduced_composition.as_dict()
            if set(want) == set(got) and all(abs(want[k] - got[k]) < 1e-6 for k in want):
                seed = p
            else:
                print(f"[ref {calc_name}] {ox}: primitive reduction changed stoichiometry "
                      f"({p.get_chemical_formula()}); using conventional cell", flush=True)
        t0 = time.time()
        r, e, conv, n, err = relax(seed, calc_name, fmax=0.05, steps=300)
        nat = len(seed)
        sg, sgn = spacegroup_of(r) if np.isfinite(e) else ("unknown", 0)
        c = Composition(ox)
        rows.append(dict(oxide=ox, calc=calc_name, natoms=nat, energy_eV=e,
                         energy_per_atom=e / nat, energy_per_fu=e / (nat / c.num_atoms),
                         converged=conv, nsteps=n, spacegroup=sg,
                         walltime_s=round(time.time() - t0, 1), error=err))
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        try:
            AseAtomsAdaptor.get_structure(r).to(filename=os.path.join(CIFDIR, f"a4_ref_{ox}_{calc_name}.cif"))
        except Exception:
            pass
        print(f"[ref {calc_name}] {ox:8s} {nat:3d}at E/fu={e/(nat/c.num_atoms):9.4f} {sg:10s} "
              f"{time.time()-t0:6.1f}s conv={conv} {err}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="A")
    ap.add_argument("--calc", default="chgnet")
    ap.add_argument("--refs", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    if args.refs:
        run_references(args.calc, os.path.join(RESULTS, f"a4_reference_oxides_{args.calc}.csv"))
        sys.exit(0)

    allf = unique_candidates()
    if args.tier == "A":
        formulas = [f for f in TIER_A if f in allf]
    else:
        formulas = [f for f in TIER_A if f in allf] + [f for f in allf if f not in TIER_A]
    print(f"{len(formulas)} candidates, calc={args.calc}", flush=True)
    run_candidates(formulas, args.calc, os.path.join(RESULTS, f"a4_hypothesis_relaxations_{args.calc}.csv"))
