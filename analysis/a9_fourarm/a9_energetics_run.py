"""a9_energetics_run.py -- run a7's corrected uMLIP energetics protocol on one arm.

Physics identical to a7 (a7_run_energetics.py + a7_mace_refine.py), which is itself a4's:
  CHGNet : a4_run_screen.run_candidates -> a4_hypotheses.all_hypotheses(n_random=4, seed=0),
           a4_relax.relax(fmax=0.10, steps=150) for every hypothesis.
  MACE   : MACE-MP-0 (medium, float32, as a4) refinement of the lowest-energy CHGNet
           hypothesis per formula, STARTED FROM THE CHGNet-RELAXED GEOMETRY,
           a4_relax.relax(fmax=0.10, steps=120).  (a7's corrected protocol; the seed-start
           variant a7 discarded is not used.)
The only deviation from a7 is WHERE relaxed CIFs are written: results/a9_structures/<arm>/
instead of results/a4_structures/, so that a formula shared with a4/a7 (or another arm) can
never overwrite an earlier deposited CIF.  File names inside are unchanged.

Usage:  python a9_energetics_run.py --arm B --calc chgnet
        python a9_energetics_run.py --arm B --calc mace
Inputs: results/a9_energetics_sample_<arm>.csv (column canonical_formula)
Output: results/a9_hypothesis_relaxations_<arm>_<calc>.csv (checkpointed; resumable)
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import argparse, os, sys, time, warnings
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
sys.path.insert(0, _CCID_ROOT + "/analysis/a4_umlip")
warnings.filterwarnings("ignore")
import torch  # noqa: E402
torch.set_num_threads(1)
from pymatgen.core import Structure                     # noqa: E402
from pymatgen.io.ase import AseAtomsAdaptor             # noqa: E402
import a4_run_screen                                    # noqa: E402
from a4_relax import relax, spacegroup_of               # noqa: E402

RESULTS = _CCID_ROOT + "/results"


def cifdir(arm):
    d = os.path.join(RESULTS, "a9_structures", arm)
    os.makedirs(d, exist_ok=True)
    return d


def run_chgnet(arm):
    fs = pd.read_csv(f"{RESULTS}/a9_energetics_sample_{arm}.csv")["canonical_formula"].tolist()
    a4_run_screen.CIFDIR = cifdir(arm)          # redirect CIF output only
    out = f"{RESULTS}/a9_hypothesis_relaxations_{arm}_chgnet.csv"
    print(f"[arm {arm}] {len(fs)} candidates, CHGNet", flush=True)
    a4_run_screen.run_candidates(fs, "chgnet", out)


def run_mace(arm):
    src = f"{RESULTS}/a9_hypothesis_relaxations_{arm}_chgnet.csv"
    out_csv = f"{RESULTS}/a9_hypothesis_relaxations_{arm}_mace.csv"
    cd = cifdir(arm)
    h = pd.read_csv(src)
    h["energy_per_atom"] = pd.to_numeric(h["energy_per_atom"], errors="coerce")
    h = h[np.isfinite(h["energy_per_atom"])]
    best = h.loc[h.groupby("formula")["energy_per_atom"].idxmin()]
    rows, done = [], set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv); rows = prev.to_dict("records")
        done = {(r["formula"], r["tag"]) for r in rows}
    print(f"[arm {arm}] MACE refine of {len(best)} best CHGNet prototypes", flush=True)
    for _, b in best.iterrows():
        f, tag = b["formula"], b["tag"]
        if (f, tag) in done:
            continue
        cif = os.path.join(cd, f"a4_{f}_{tag}_chgnet.cif")
        if not os.path.exists(cif):
            print(f"[mace] {f} {tag}: no CHGNet CIF", flush=True); continue
        atoms = AseAtomsAdaptor.get_atoms(Structure.from_file(cif))
        t0 = time.time()
        r, e, conv, n, err = relax(atoms, "mace", fmax=0.10, steps=120)
        nat = len(atoms)
        sg, sgn = spacegroup_of(r) if np.isfinite(e) else ("unknown", 0)
        rows.append(dict(formula=f, tag=tag, calc="mace", natoms=nat, energy_eV=e,
                         energy_per_atom=e / nat if np.isfinite(e) else np.nan,
                         converged=conv, nsteps=n, spacegroup=sg, spacegroup_number=sgn,
                         volume_per_atom=r.get_volume() / nat if np.isfinite(e) else np.nan,
                         walltime_s=round(time.time() - t0, 1), error=err))
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        if np.isfinite(e):
            try:
                AseAtomsAdaptor.get_structure(r).to(
                    filename=os.path.join(cd, f"a4_{f}_{tag}_mace.cif"))
            except Exception:
                pass
        print(f"[mace] {f:14s} {tag:26s} {nat:3d}at E/at="
              f"{e/nat if np.isfinite(e) else float('nan'):8.4f} {sg:10s} "
              f"{time.time()-t0:6.1f}s conv={conv} {err}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--calc", required=True, choices=["chgnet", "mace"])
    a = ap.parse_args()
    run_chgnet(a.arm) if a.calc == "chgnet" else run_mace(a.arm)
