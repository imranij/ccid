"""Experiment 1 - reproduce the archived Composition-CVAE generation run.

20,000 samples from the deposited checkpoint, 8 seeds, at two condition-jitter
settings:
  * jitter = 0.08 -- the default in the deposited generate_composition_candidates.py
  * jitter = 0.30 -- the setting recovered from the archived candidate CSV
    (the archived condition columns match np.random.default_rng(123).normal(mu, 0.30)
     to machine precision; see a2_seed_forensics.py)
"""
from __future__ import annotations

import json

import pandas as pd

import a2_common as C

SEEDS = [123, 0, 1, 2, 3, 4, 5, 42]
JITTERS = [0.08, 0.30]


def main() -> None:
    vocab = C.element_vocab()
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for jitter in JITTERS:
        for seed in SEEDS:
            decoded, cond, _ = C.cvae_decode(20000, seed, jitter=jitter)
            cand = C.decode_batch(decoded, vocab)
            m = C.score_frame(cand, vocab)
            m["seed"] = seed
            m["condition_jitter"] = jitter
            rows.append(m)
            print(jitter, seed, round(m["validity_rate"], 5), m["unique_valid_count"],
                  m["novel_unique_count"], m["unique_formula_count_all"], round(m["mean_pairwise_composition_distance"], 5))
            if seed == 123:
                cand["target_stability_utility"] = cond[:, 0]
                cand["target_gap_utility"] = cond[:, 1]
                cand["candidate_score"] = 0.5 * cond[:, 0] + 0.5 * cond[:, 1]
                cand.to_csv(
                    C.RESULTS_DIR / f"a2_reproduce_seed123_jitter{jitter:g}_candidates.csv",
                    index=False,
                )

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "a2_reproduce_seeds.csv", index=False)

    cols = ["validity_rate", "uniqueness_rate", "novelty_rate", "valid_count",
            "unique_valid_count", "novel_unique_count", "mean_pairwise_composition_distance",
            "unique_formula_count_all", "modal_formula_share_all"]
    summary = {
        "archived_metrics_json": json.loads(
            (C.PROJECT_ROOT / "outputs_dgx" / "composition_cvae_generated_metrics.json").read_text()
        ),
        "manuscript_reported": {
            "valid_count": 6577, "validity_rate": 0.329, "unique_valid_count": 71,
            "novel_unique_count": 69, "novelty_rate": 0.972,
            "mean_pairwise_composition_distance": 0.278,
            "note": "read from the manuscript by the coordinating agent, not computed here",
        },
        "by_jitter": {
            f"{j:g}": {
                c: {
                    "mean": float(g[c].mean()), "std": float(g[c].std(ddof=1)),
                    "min": float(g[c].min()), "max": float(g[c].max()), "n_seeds": int(len(g)),
                }
                for c in cols
            }
            for j, g in df.groupby("condition_jitter")
        },
    }
    (C.RESULTS_DIR / "a2_reproduce_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["by_jitter"], indent=2))


if __name__ == "__main__":
    main()
