"""Experiment 3 - unconditional / condition-destroyed controls.

Same checkpoint, same decoder, same latent draw; only the condition vector is
changed. Isolates what the condition signal contributes on top of the decoder.

Arms (all 20,000 samples, shared latent draw per seed):
  conditioned       archived path: c ~ (N(0.9,0.30), N(0.8,0.30)) clipped to [0,1]
  shuffled          the same condition vectors, randomly permuted across samples
                    (marginal distribution identical, pairing with z destroyed)
  const_mid         c = (0.5, 0.5) for every sample
  const_target      c = (0.9, 0.8) for every sample (jitter removed)
  const_train_mean  c = training-set mean (stability_utility, gap_utility)
  uniform_random    c ~ U(0,1)^2
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

import a2_common as C

N = 20000
SEEDS = [123, 0, 1, 2, 3]


def main() -> None:
    vocab = C.element_vocab()
    model, meta = C.get_model()
    table, _ = C.get_known()
    train_mean = (
        float(table["stability_utility"].mean()),
        float(table["gap_utility"].mean()),
    )
    print("training-set mean conditions:", train_mean)

    rows = []
    for seed in SEEDS:
        _, cond, z = C.cvae_decode(N, seed, jitter=0.30)
        rng = np.random.default_rng(50_000 + seed)
        arms = {
            "conditioned": cond,
            "shuffled": rng.permutation(cond, axis=0),
            "const_mid": np.tile([0.5, 0.5], (N, 1)).astype(np.float32),
            "const_target": np.tile([0.9, 0.8], (N, 1)).astype(np.float32),
            "const_train_mean": np.tile(train_mean, (N, 1)).astype(np.float32),
            "uniform_random": rng.random((N, 2)).astype(np.float32),
        }
        for name, c in arms.items():
            with torch.no_grad():
                dec = model.decode(z, torch.tensor(np.asarray(c), dtype=torch.float32)).numpy()
            dec = dec[:, : int(meta["composition_dim"])]
            cand = C.decode_batch(dec, vocab)
            m = C.score_frame(cand, vocab)
            m["arm"] = name
            m["seed"] = seed
            rows.append(m)
            print(seed, name, round(m["validity_rate"], 4), m["unique_valid_count"],
                  m["unique_formula_count_all"], round(m["mean_pairwise_composition_distance"], 4))

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "a2_unconditional_runs.csv", index=False)
    cols = ["validity_rate", "uniqueness_rate", "novelty_rate", "unique_valid_count",
            "novel_unique_count", "mean_pairwise_composition_distance",
            "unique_formula_count_all", "modal_formula_share_all"]
    summ = {arm: {c: {"mean": float(g[c].mean()), "std": float(g[c].std(ddof=1))} for c in cols}
            for arm, g in df.groupby("arm")}
    summ["_training_mean_conditions"] = {"stability_utility": train_mean[0],
                                         "gap_utility": train_mean[1]}
    (C.RESULTS_DIR / "a2_unconditional_summary.json").write_text(json.dumps(summ, indent=2))
    print(df.groupby("arm")[cols].agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
