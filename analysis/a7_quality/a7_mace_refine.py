"""a7_mace_refine.py -- MACE-MP-0 refinement of the RANDOM sample, using a4's EXACT
refinement protocol.

Why this file exists: a first attempt refined MACE from the raw structure seed. a4 does
not do that -- a4_mace_refine.py restarts MACE from the CHGNet-RELAXED geometry with
fmax=0.10, steps=120. Starting from the seed gave the random arm a 72 % MACE convergence
rate against a4's 99 % for the CVAE arm, which inflated the random arm's energies and
produced a spurious 'the CVAE wins' result. That first run is kept as
a7_hypothesis_relaxations_mace_SEEDSTART_discarded.csv and is NOT used.
"""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import os, sys, time, warnings
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
sys.path.insert(0, _CCID_ROOT + "/analysis/a4_umlip")
warnings.filterwarnings("ignore")
import torch; torch.set_num_threads(1)

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from a4_relax import relax, spacegroup_of
from a4_run_screen import RESULTS, CIFDIR


def main():
    src = os.path.join(RESULTS, "a7_hypothesis_relaxations_chgnet.csv")
    out_csv = os.path.join(RESULTS, "a7_hypothesis_relaxations_mace.csv")
    h = pd.read_csv(src)
    h["energy_per_atom"] = pd.to_numeric(h["energy_per_atom"], errors="coerce")
    h = h[np.isfinite(h["energy_per_atom"])]
    best = h.loc[h.groupby("formula")["energy_per_atom"].idxmin()]
    rows, done = [], set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv); rows = prev.to_dict("records")
        done = {(r["formula"], r["tag"]) for r in rows}
    for _, b in best.iterrows():
        f, tag = b["formula"], b["tag"]
        if (f, tag) in done:
            continue
        cif = os.path.join(CIFDIR, f"a4_{f}_{tag}_chgnet.cif")
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
                    filename=os.path.join(CIFDIR, f"a4_{f}_{tag}_mace.cif"))
            except Exception:
                pass
        print(f"[mace] {f:14s} {tag:26s} {nat:3d}at E/at="
              f"{e/nat if np.isfinite(e) else float('nan'):8.4f} {sg:10s} "
              f"{time.time()-t0:6.1f}s conv={conv} {err}", flush=True)


if __name__ == "__main__":
    main()
