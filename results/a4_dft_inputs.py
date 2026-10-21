#!/usr/bin/env python
"""
a4_dft_inputs.py
================
Generate VASP and Quantum ESPRESSO inputs for the CVAE candidates of MTCOMM-D-26-04679,
starting from uMLIP-relaxed structures. This is the "one command away" hand-off: the
reviewers asked for DFT validation, and this produces ready-to-submit inputs.

USAGE
-----
    # from the CIFs produced by the sandbox analysis or by a4_dgx_umlip_screen.py
    python a4_dft_inputs.py --structures a4_structures/ --out dft/ --code vasp
    python a4_dft_inputs.py --structures a4_structures/ --out dft/ --code qe \
                            --pseudo-dir /path/to/SSSP_1.3.0_PBE_efficiency
    # restrict to the best structure per candidate, as selected by the energetics table
    python a4_dft_inputs.py --structures a4_structures/ --out dft/ --code both \
                            --energetics a4_energetics_chgnet.csv --top 12

VASP
----
Uses pymatgen's MPRelaxSet so the parameters match the Materials Project workflow, which
is what makes the resulting energies directly comparable to MP formation energies and
hull data. POTCARs are NOT written unless pymatgen is configured with a licensed
PBE_54 POTCAR directory (`pmg config -p <path> POTCAR_DIR`); this is a VASP licensing
requirement, not a limitation of this script.

A two-step protocol is written per candidate:
    relax/   ISIF=3 full cell+ion relaxation, 520 eV cutoff (MP standard)
    static/  single point on the relaxed geometry, denser k-mesh, LWAVE/LCHARG for
             a subsequent dielectric (LEPSILON) or band-structure run

Quantum ESPRESSO
----------------
Writes vc-relax then scf inputs. Pseudopotential filenames follow the SSSP efficiency
naming convention; adjust --pseudo-dir and the PSEUDO map below to match your library.
Cutoffs default to 60/480 Ry (wfc/rho), which is safe for these oxides but should be
converged explicitly before publication.

WHAT YOU STILL HAVE TO DO
-------------------------
1. Converge cutoffs and k-meshes. The defaults here are sensible, not converged.
2. For the formation energies the reviewers want, run the SAME settings on the
   competing phases (binary oxides at minimum; ideally the full MP hull).
3. These candidates include several that can only be charge-balanced with Ti(3+) or
   Sn(2+). Those are open-shell and need a spin-polarised calculation (ISPIN=2 /
   nspin=2) and probably a Hubbard U; the non-spin-polarised defaults here are wrong
   for them. Such structures are flagged in the generated MANIFEST.csv.
"""
from __future__ import annotations

import argparse
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

# Cations that can only charge-balance in the reduced, open-shell state.
OPEN_SHELL_RISK = {"Ti", "Sn"}

# SSSP efficiency (v1.3.0) pseudopotential filenames.
PSEUDO = {
    "O": "O.pbe-n-kjpaw_psl.0.1.UPF", "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF", "Ca": "Ca_pbe_v1.uspp.F.UPF",
    "Sr": "Sr_pbe_v1.uspp.F.UPF", "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Sc": "Sc_ONCV_PBE-1.0.oncvpsp.upf", "Y": "Y_pbe_v1.uspp.F.UPF",
    "La": "La.GGA-PBE-paw-v1.0.UPF", "Ti": "ti_pbe_v1.4.uspp.F.UPF",
    "Zr": "Zr_pbe_v1.uspp.F.UPF", "Hf": "Hf-sp.oncvpsp.upf",
    "Nb": "Nb.pbe-spn-kjpaw_psl.0.3.0.UPF", "Ta": "Ta_pbe_v1.uspp.F.UPF",
    "Sn": "Sn_pbe_v1.uspp.F.UPF", "Ga": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
}


def write_vasp(struct, outdir: str, spin: bool):
    from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
    os.makedirs(outdir, exist_ok=True)
    incar_extra = {"ISPIN": 2 if spin else 1}
    rel = os.path.join(outdir, "relax")
    stat = os.path.join(outdir, "static")
    try:
        MPRelaxSet(struct, user_incar_settings=incar_extra).write_input(rel, potcar_spec=True)
        MPStaticSet(struct, user_incar_settings={**incar_extra, "LWAVE": True, "LCHARG": True}
                    ).write_input(stat, potcar_spec=True)
        return True, "POTCAR.spec written (configure pmg POTCAR_DIR for real POTCARs)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


QE_TEMPLATE = """&CONTROL
  calculation   = '{calc}'
  prefix        = '{prefix}'
  outdir        = './out'
  pseudo_dir    = '{pseudo_dir}'
  tprnfor       = .true.
  tstress       = .true.
  etot_conv_thr = 1.0d-5
  forc_conv_thr = 1.0d-4
/
&SYSTEM
  ibrav       = 0
  nat         = {nat}
  ntyp        = {ntyp}
  ecutwfc     = {ecutwfc}
  ecutrho     = {ecutrho}
  occupations = 'smearing'
  smearing    = 'gaussian'
  degauss     = 0.005
{spin_block}/
&ELECTRONS
  conv_thr    = 1.0d-8
  mixing_beta = 0.3
/
&IONS
  ion_dynamics = 'bfgs'
/
&CELL
  cell_dynamics  = 'bfgs'
  press_conv_thr = 0.1
/
ATOMIC_SPECIES
{species}
CELL_PARAMETERS angstrom
{cell}
ATOMIC_POSITIONS crystal
{positions}
K_POINTS automatic
{kmesh} 0 0 0
"""


def write_qe(struct, outdir: str, pseudo_dir: str, spin: bool,
             ecutwfc=60, ecutrho=480, kspacing=0.03):
    from pymatgen.core import Element
    os.makedirs(outdir, exist_ok=True)
    els = sorted({str(s.specie) for s in struct})
    missing = [e for e in els if e not in PSEUDO]
    species = "\n".join(f"  {e} {Element(e).atomic_mass:.4f} {PSEUDO.get(e, 'MISSING.UPF')}"
                        for e in els)
    cell = "\n".join("  " + "  ".join(f"{x:14.10f}" for x in v) for v in struct.lattice.matrix)
    positions = "\n".join(f"  {str(s.specie):3s} " +
                          "  ".join(f"{x:14.10f}" for x in s.frac_coords) for s in struct)
    km = [max(1, int(round(1.0 / (kspacing * a)))) for a in struct.lattice.abc]
    spin_block = "  nspin = 2\n  starting_magnetization(1) = 0.2\n" if spin else ""
    for calc, fn in (("vc-relax", "vc-relax.in"), ("scf", "scf.in")):
        txt = QE_TEMPLATE.format(
            calc=calc, prefix=struct.composition.reduced_formula, pseudo_dir=pseudo_dir,
            nat=len(struct), ntyp=len(els), ecutwfc=ecutwfc, ecutrho=ecutrho,
            spin_block=spin_block, species=species, cell=cell, positions=positions,
            kmesh=" ".join(map(str, km)))
        with open(os.path.join(outdir, fn), "w") as fh:
            fh.write(txt)
    return True, ("missing pseudopotentials: " + ", ".join(missing)) if missing else "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structures", required=True, help="directory of relaxed CIFs")
    ap.add_argument("--out", default="dft")
    ap.add_argument("--code", default="both", choices=["vasp", "qe", "both"])
    ap.add_argument("--pseudo-dir", default="./pseudo")
    ap.add_argument("--energetics", default="",
                    help="a4_energetics_<calc>.csv; restricts to the best structure per candidate")
    ap.add_argument("--top", type=int, default=0, help="only the N lowest-energy candidates")
    args = ap.parse_args()

    from pymatgen.core import Structure

    wanted = None
    if args.energetics and os.path.exists(args.energetics):
        e = pd.read_csv(args.energetics)
        if args.top:
            e = e.nsmallest(args.top, "decomposition_energy_eV_per_atom")
        wanted = {(r["canonical_formula"], r["best_prototype"]) for _, r in e.iterrows()}
        print(f"Restricting to {len(wanted)} best-per-candidate structures.")

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for fn in sorted(os.listdir(args.structures)):
        if not fn.endswith(".cif") or fn.startswith("a4_ref_"):
            continue
        base = fn[:-4]
        if base.startswith("a4_"):
            base = base[3:]
        if wanted is not None:
            parts = base.rsplit("_", 1)          # strip trailing _<calc>
            stem = parts[0]
            if not any(stem == f"{f}_{t}" for f, t in wanted):
                continue
        path = os.path.join(args.structures, fn)
        try:
            st = Structure.from_file(path)
        except Exception as ex:
            manifest.append(dict(cif=fn, status="unreadable", detail=str(ex)))
            continue
        els = {str(s.specie) for s in st}
        spin = bool(els & OPEN_SHELL_RISK)
        row = dict(cif=fn, formula=st.composition.reduced_formula, natoms=len(st),
                   spin_polarised=spin,
                   spin_reason="contains Ti/Sn, which may be in a reduced open-shell state"
                               if spin else "closed-shell cations only")
        d = os.path.join(args.out, base)
        if args.code in ("vasp", "both"):
            ok, msg = write_vasp(st, os.path.join(d, "vasp"), spin)
            row["vasp"] = "ok" if ok else msg
        if args.code in ("qe", "both"):
            ok, msg = write_qe(st, os.path.join(d, "qe"), args.pseudo_dir, spin)
            row["qe"] = msg
        manifest.append(row)

    m = pd.DataFrame(manifest)
    m.to_csv(os.path.join(args.out, "MANIFEST.csv"), index=False)
    print(f"Wrote inputs for {len(m)} structures to {args.out}/")
    if "spin_polarised" in m.columns and m["spin_polarised"].any():
        print(f"{int(m['spin_polarised'].sum())} structures flagged spin-polarised "
              f"(Ti/Sn present; check oxidation state and consider +U).")


if __name__ == "__main__":
    main()
