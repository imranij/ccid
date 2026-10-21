"""Merge the three sweep stages into one CSV, ordered for the trade-off curve."""
from __future__ import annotations
import pandas as pd
import a8_common as A

R = A.RESULTS_DIR
parts = [pd.read_csv(R / f) for f in
         ("a8_sweep_results.csv", "a8_sweep_results_stage2.csv", "a8_sweep_results_stage3.csv")]
df = pd.concat(parts, ignore_index=True)
# the buggy-loss control has no per-dimension budget; the recorded value is the
# unused argparse default, so blank it out to avoid a misleading sweep row.
df.loc[df["loss"] == "buggy", "free_bits_per_dim"] = float("nan")
df = df.sort_values(["free_bits_per_dim", "beta_max"], na_position="first").reset_index(drop=True)
df.to_csv(R / "a8_sweep_all.csv", index=False)
cols = ["tag", "loss", "free_bits_per_dim", "beta_max", "test_composition_mse",
        "total_kl_nats", "active_units_kl_gt_0.01", "gen_validity_rate_mean",
        "gen_unique_valid_count_mean", "gen_novelty_rate_mean",
        "gen_mean_pairwise_composition_distance_mean",
        "gen_unique_formula_count_all_mean", "fixed_cond_unique_valid",
        "fixed_cond_distinct_formulas", "wall_clock_seconds"]
print(df[cols].to_string(index=False))
