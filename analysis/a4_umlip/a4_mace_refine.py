"""
a4_mace_refine.py -- independent MACE-MP-0 cross-check.

CHGNet does the broad hypothesis search (it is ~10x cheaper on CPU). This script then
re-relaxes, with MACE-MP-0, (a) every binary-oxide reference and (b) the best CHGNet
structure of each candidate. That gives a second, independently-trained potential for
the quantity that actually matters -- the decomposition energy -- so the report can
state cross-potential agreement rather than relying on one model.

MACE is started from the CHGNet-relaxed geometry, so this is a refinement, not an
independent structure search. Agreement between the two therefore tests the ENERGETICS,
not the structure hypothesis; the structure hypothesis remains the dominant uncertainty
and the report says so.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from a4_relax import relax, spacegroup_of
from a4_run_screen import run_references, RESULTS, CIFDIR, TIER_A


def refine(order_first=TIER_A):
    src = os.path.join(RESULTS, "a4_hypothesis_relaxations_chgnet.csv")
    out_csv = os.path.join(RESULTS, "a4_hypothesis_relaxations_mace.csv")
    if not os.path.exists(src):
        print("no CHGNet results yet")
        return
    h = pd.read_csv(src)
    h = h[np.isfinite(h["energy_per_atom"])]
    best = h.loc[h.groupby("formula")["energy_per_atom"].idxmin()]

    rows, done = [], set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        done = {(r["formula"], r["tag"]) for r in rows}

    # Tier-A candidates first so a partial run still covers the manuscript's headline set.
    best = best.copy()
    best["_pri"] = [order_first.index(f) if f in order_first else 999
                    for f in best["formula"]]
    best = best.sort_values("_pri")

    for _, b in best.iterrows():
        f, tag = b["formula"], b["tag"]
        if (f, tag) in done:
            continue
        cif = os.path.join(CIFDIR, f"a4_{f}_{tag}_chgnet.cif")
        if not os.path.exists(cif):
            continue
        try:
            atoms = AseAtomsAdaptor.get_atoms(Structure.from_file(cif))
        except Exception as e:
            print(f"[mace] {f} {tag}: unreadable CIF: {e}", flush=True)
            continue
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
    run_references("mace", os.path.join(RESULTS, "a4_reference_oxides_mace.csv"))
    refine()
