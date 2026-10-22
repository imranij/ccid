# Comparison of Machine Learning, Random Selection and Ion Substitution for Ceramic Oxide Design and Generation

Code, trained models, generated candidates and analysis scripts for:

> Imran, M. Nouioua, T. S. Tamir. *Comparison of Machine Learning, Random Selection and Ion Substitution for Ceramic Oxide Design and Generation* (submitted).

The manuscript compares four ways of proposing ceramic oxide compositions through one chemical filter: (A) random element selection, (B) sampling by training-set element frequency, (C) data-mined ion substitution into known compounds (Hautier et al. 2011, via pymatgen), and (D) the Composition-CVAE after correction of its free-bits regulariser. Release v1 (July 2026) contained only the original pipeline in `dgx/code/`.

## What is here

| Path | Contents |
|---|---|
| `dgx/code/` | The original two-stage pipeline, **unchanged**: Stage-1 screening (`inverse_design_baseline.py`, `candidate_validation_pack.py`), the Composition-CVAE (`composition_cvae_common.py`, `train_composition_cvae_dgx.py`) and candidate generation (`generate_composition_candidates.py`). Kept as deposited because it produced the checkpoint analysed in the paper. |
| `dgx/code/free_bits_fix.patch` | Correction of the free-bits regulariser (per-dimension KL floor instead of a clamp on the summed KL). Apply with `git apply dgx/code/free_bits_fix.patch`. The analyses below use their own copy of both losses, so applying it is not required to reproduce the paper. |
| `dgx/outputs_dgx/` | The deposited checkpoint, its metadata and training history, the 25,148 training records, and the 20,000 archived generated samples. |
| `analysis/a1_stage1/` | Stage-1 forensics: the scoring rules actually applied, reproduction of the deposited tables, and the corrected screen (harmonised band-gap filter, element exclusion, imputation sensitivity). |
| `analysis/a2_generative/` | Reproduction of the generation run, the five random-selection controls, condition-destroyed controls, the condition-grid sweep and diversity metrics. |
| `analysis/a3_novelty/` | Element Mover's Distance nearest neighbours and substitution-family grouping. |
| `analysis/a4_umlip/` | Structure hypotheses (prototype substitution + PyXtal), CHGNet / MACE-MP-0 relaxations, decomposition energies into binary oxides, descriptor recomputation, and hand-off scripts for a GPU uMLIP screen and DFT inputs. |
| `analysis/a5_surrogate/` | Composition-only band-gap, stability and dielectric surrogates, candidate scoring and the rediscovery test. |
| `analysis/a6_figures/` | Every figure in the paper (`sh analysis/a6_figures/a6_make_all.sh`). |
| `analysis/a7_quality/` | Candidate-quality comparison, CVAE versus random, on energetics, surrogate properties and chemistry. |
| `analysis/a8_retrain/` | Diagnosis of the posterior collapse, the corrected loss, the free-bits sweep and the retrained checkpoints. The learned arm (D) uses `checkpoints/fb0.5_b0.01`. |
| `analysis/a9_fourarm/` | The four-arm comparison: the ion-substitution arm (C), counts over five seeds for all arms, chemistry, surrogate predictions, statistics, and the uMLIP energetics for arms B, C and D (arm A reuses the a7 random sample), including the corrected spinel builder. |
| `results/` | Every result table (CSV/JSON), LaTeX table bodies, per-analysis reports (`*_REPORT.md`), relaxed structures (`a4_structures/`, `a9_structures/`) and figures. `results/a9_REPORT.md` and `results/a9_energetics_REPORT.md` document the four-arm study. |

## Setup

```bash
pip install -r requirements-analysis.txt
```

Download the public datasets (not redistributed here) to these locations, relative to the repository root:

* `Paper_1_Uncertainty_Calibrated_Data_Repair_for_Ceramic_Materials/data/dielectric_constant.json.gz` — Materials Project dielectric dataset ([figshare](https://ndownloader.figshare.com/files/13213475); `dgx/code/inverse_design_baseline.py` downloads it automatically if absent).
* `shared_datasets/matminer/castelli_perovskites.json.gz`, `wolverton_oxides.json.gz`, `double_perovskites_gap.json.gz` and `phonon_dielectric_mp.json.gz` — via [Matminer](https://github.com/hackingmaterials/matminer) (`matminer.datasets.load_dataset`).

All scripts resolve paths from the repository root. To run them from elsewhere, set `CCID_ROOT` to the repository path.

## Reproducing the paper

Run from the repository root. Times are for a 2-core CPU.

```bash
# Stage 1: fresh re-run of the original pipeline, then audit and correct it  (~1 min)
python analysis/a1_stage1/a1_make_repro.py
python analysis/a1_stage1/a1_reproduce_check.py
python analysis/a1_stage1/a1_corrected_stage1.py
python analysis/a1_stage1/a1_make_tables.py

# Posterior-collapse diagnosis on the deposited checkpoint  (~1.5 min)
python analysis/a8_retrain/a8_exp0_baseline_pathology.py

# Generation reproduction, random controls, unconditional controls, condition sweep, diversity
python analysis/a2_generative/a2_exp1_reproduce.py
python analysis/a2_generative/a2_exp2_random_baseline.py
python analysis/a2_generative/a2_exp3_unconditional.py
python analysis/a2_generative/a2_exp4_condition_sweep.py
python analysis/a2_generative/a2_exp5_diversity.py
# Corrected-loss retraining and sweep (~5 min per 150-epoch run): analysis/a8_retrain/a8_train.py
# Surrogates: run analysis/a5_surrogate/ before analysis/a7_quality/ (a7 loads the a5 models)
# uMLIP screen: analysis/a4_umlip/run_chgnet.sh, then run_mace.sh

# Four-arm comparison (run in this order; the a5 retrain is ~1 h, the energetics several hours on 2 cores)
python analysis/a9_fourarm/a9_arm_c_substitution.py
python analysis/a9_fourarm/a9_counts.py
python analysis/a9_fourarm/a9_umlip_sample.py
python analysis/a9_fourarm/a9_chemistry.py
sh analysis/a9_fourarm/a9_retrain_a5.sh
python analysis/a9_fourarm/a9_check_a5_retrain.py
python analysis/a9_fourarm/a9_surrogates.py
python analysis/a9_fourarm/a9_compare.py
python analysis/a9_fourarm/a9_tables.py
python analysis/a9_fourarm/a9_energetics_sample_B.py
sh analysis/a9_fourarm/a9_energetics_driver.sh
python analysis/a9_fourarm/a9_energetics_spinel_fix.py
python analysis/a9_fourarm/a9_energetics_analyze.py

# All figures, from the result files  (~20 s)
sh analysis/a6_figures/a6_make_all.sh
```

Each `results/aN_REPORT.md` lists the scripts of that analysis, what they compute, and the provenance of every reported number.

Four steps were checked by re-running them from a fresh copy of this repository: the figure build reproduces the figure files byte for byte; the collapse diagnosis reproduces all 56 numeric fields of `results/a8_baseline_pathology.json`; the Stage-1 audit reproduces `results/a1_reproduction_status.csv` exactly; and the corrected Stage-1 screen reproduces its tables, including `results/a1_tables.tex`, exactly.

## Not included

* The source datasets (public; see Setup).
* The trained surrogate models (`analysis/a5_surrogate/*.pkl`, 69–313 MB each, above GitHub's file limit). Regenerate them with the `a5_train_*.py` scripts.
* External database cross-checks (Materials Project, OQMD, AFLOW) and DFT calculations, which were not part of this release.
* Run logs (`*.log`), which are excluded by `.gitignore`.

## Citation

Please cite the paper above. An archived copy of this release will be deposited on Zenodo; its DOI will be added here.
