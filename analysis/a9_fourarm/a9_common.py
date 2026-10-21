"""a9_common.py -- shared harness for the four-arm comparison (a9).

Every arm goes through the a2 harness (a2_common.decode_batch -> the project's own
decode_composition_vector; a2_common.score_frame for validity / uniqueness / novelty /
mean pairwise distance), so the filter path is byte-identical to a2 and a8.

Arms
  A  random            a2 arm rand_uniform41 (a2_exp2_random_baseline.relabel, uniform
                       over the 40 whitelist cations)
  B  element stats     a2 arm rand_freq41 (same relabel, cations ~ training frequency)
  C  ion substitution  a9_arm_c_substitution.py (Hautier 2011 probabilities, pymatgen)
  D  corrected CVAE    a8 checkpoint fb0.5_b0.01, archived settings (a2_common.cvae_decode)
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import ast
import sys
from pathlib import Path

for _p in ("/analysis/a2_generative", "/analysis/a8_retrain", "/analysis/a7_quality",
           "/analysis/a4_umlip", "/analysis/a9_fourarm", "/dgx/code"):
    if _CCID_ROOT + _p not in sys.path:
        sys.path.insert(0, _CCID_ROOT + _p)

RESULTS = Path(_CCID_ROOT + "/results")
A9_DIR = Path(_CCID_ROOT + "/analysis/a9_fourarm")
N = 20000
SEEDS = [0, 1, 2, 3, 4]
JITTER = 0.30                     # archived generation setting (a2 section 1.1)
ARM_D_CKPT = Path(_CCID_ROOT + "/analysis/a8_retrain/checkpoints/fb0.5_b0.01_checkpoint.pt")
SAMPLE_SEED = 20260913            # a7's sampling seed
ARM_LABEL = {"A": "Random (uniform, 41-element whitelist)",
             "B": "Element statistics (training frequency)",
             "C": "Ion substitution (Hautier 2011)",
             "D": "Learned (corrected CVAE)"}


def _load_a3_family_functions():
    """Import group_of / family_key from analysis/a3_novelty/a3_families.py WITHOUT
    executing its module-level code (which reads and rewrites a3 result files).  Only the
    import statements and the two function definitions are executed, verbatim."""
    src = Path(_CCID_ROOT + "/analysis/a3_novelty/a3_families.py").read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("group_of", "family_key")]
    ns: dict = {}
    exec(compile("from pymatgen.core import Element\n"
                 "from composition_cvae_common import parse_formula\n", "a3_imports", "exec"), ns)
    exec(compile(ast.Module(body=keep, type_ignores=[]), "a3_families.py", "exec"), ns)
    return ns["family_key"]


family_key = _load_a3_family_functions()


def family_id(formula: str) -> str:
    return repr(family_key(formula))
