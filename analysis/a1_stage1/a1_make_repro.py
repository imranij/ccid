"""Build analysis/a1_stage1/repro: a fresh, unmodified re-run of the original Stage-1 pipeline.

a1_reproduce_check.py compares the deposited Stage-1 CSVs against this re-run.
Requires the public datasets in their usual places at the repo root
(shared_datasets/matminer/*.json.gz and
 Paper_1_Uncertainty_Calibrated_Data_Repair_for_Ceramic_Materials/data/dielectric_constant.json.gz).
"""
import os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(os.environ.get("CCID_ROOT") or Path(__file__).resolve().parents[2])
REPRO = ROOT / "analysis" / "a1_stage1" / "repro"
if REPRO.exists():
    shutil.rmtree(REPRO)
(REPRO / "dgx").mkdir(parents=True)
shutil.copytree(ROOT / "dgx" / "code", REPRO / "dgx" / "code", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(ROOT / "shared_datasets", REPRO / "shared_datasets")
dielectric = "Paper_1_Uncertainty_Calibrated_Data_Repair_for_Ceramic_Materials"
shutil.copytree(ROOT / dielectric, REPRO / dielectric)
for script in ("inverse_design_baseline.py", "candidate_validation_pack.py"):
    subprocess.run([sys.executable, str(REPRO / "dgx" / "code" / script)], check=True)
print("repro written to", REPRO)
