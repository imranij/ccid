# Compositional CVAE for Ceramic Materials Design

This repository contains code for a two-stage candidate screening and generation framework for ceramic dielectric materials.

## Structure

* `dgx/code/`: Contains Python scripts for training the Conditional Variational Autoencoder (CVAE) and running baselines.
  * `inverse_design_baseline.py`: Screening baseline for existing materials records.
  * `candidate_validation_pack.py`: Script to check compositions against stability and process constraints.
  * `composition_cvae_common.py`: Shared CVAE architecture and helper utilities.
  * `train_composition_cvae_dgx.py`: Script to train the composition CVAE.
  * `generate_composition_candidates.py`: Candidate generator using the trained model.

## Datasets

The scripts require the following publicly available materials datasets:

1. **Dielectric Constant Dataset**:
   * Source: Materials Project dielectric constant dataset (via Figshare).
   * Download Link: [dielectric_constant.json.gz](https://ndownloader.figshare.com/files/13213475)
   * The baseline script automatically downloads this dataset to a `Paper_1_Uncertainty_Calibrated_Data_Repair_for_Ceramic_Materials/data/` folder if it is not present.

2. **Perovskite and Oxide Datasets**:
   * Source: Castelli Perovskites, Wolverton Oxides (OQMD), and Double Perovskite datasets retrieved via [Matminer](https://github.com/hackingmaterials/matminer).
   * Place the downloaded `.json.gz` files in a `shared_datasets/matminer/` directory in the repository root:
     * `castelli_perovskites.json.gz`
     * `wolverton_oxides.json.gz`
     * `double_perovskites_gap.json.gz`

## Installation

Ensure you have PyTorch, NumPy, Pandas, and Scikit-Learn installed:

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

## Running the Code

Run the baseline and generation scripts in order:

```bash
# Run baseline screening
python dgx/code/inverse_design_baseline.py

# Run candidate validation checks
python dgx/code/candidate_validation_pack.py

# Train CVAE model
python dgx/code/train_composition_cvae_dgx.py --epochs 150

# Generate candidates
python dgx/code/generate_composition_candidates.py --n-samples 20000 --condition-jitter 0.30
```
