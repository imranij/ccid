"""Experiment 2 - random element-selection controls through the identical filter path.

Design. Every arm reuses the *same* decoded fraction profile that the CVAE produced
(so the number of cations, the sorted cation-fraction profile and the decoded oxygen
fraction are held fixed) and randomises only WHICH element carries each fraction.
The vectors are then passed to the project's own decode_composition_vector, so the
integer-stoichiometry rounding, the charge-balance search and the whitelist/risk test
are byte-identical to the manuscript's.

Arms
  cvae            reference (deposited checkpoint, archived settings)
  rand_uniform41  the 3 selected cations drawn uniformly from the 41-element
                  COMMON_CERAMIC_ELEMENTS whitelist (minus O)
  rand_freq77     drawn from the 76 non-O vocabulary elements with probability
                  proportional to their occurrence count in the training table
  rand_freq41     training-frequency weighted but restricted to the 41-element whitelist
  rand_uniform77  drawn uniformly from the 76 non-O vocabulary elements
  rand_dirichlet41 no CVAE information at all: Dirichlet(1,1,1) cation fractions,
                  Uniform(0,1) oxygen fraction, identities uniform from the whitelist
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import a2_common as C

N = 20000
SEEDS = [123, 0, 1, 2, 3]
JITTER = 0.30  # the setting that reproduces the archived run


def training_element_weights(vocab: list[str]) -> np.ndarray:
    tr = pd.read_csv(C.PROJECT_ROOT / "outputs_dgx" / "composition_cvae_training_records.csv")
    counts = {e: 0 for e in vocab}
    for f in tr["formula"].astype(str):
        for e in C.parse_formula(f):
            if e in counts:
                counts[e] += 1
    return np.array([counts[e] for e in vocab], dtype=float)


def relabel(decoded: np.ndarray, vocab: list[str], rng: np.random.Generator,
            pool_idx: np.ndarray, probs: np.ndarray | None) -> np.ndarray:
    """Keep each sample's fraction profile; move the 3 largest non-O entries onto
    randomly drawn elements."""
    o_idx = vocab.index("O")
    non_o = np.array([i for i in range(len(vocab)) if i != o_idx])
    out = np.zeros_like(decoded)
    out[:, o_idx] = decoded[:, o_idx]
    sub = decoded[:, non_o]
    order = np.argsort(-sub, axis=1)
    for s in range(len(decoded)):
        vals = sub[s, order[s]]
        top = rng.choice(pool_idx, size=3, replace=False, p=probs)
        rest = rng.permutation(np.array([i for i in non_o if i not in set(top.tolist())]))
        targets = np.concatenate([top, rest])
        out[s, targets] = vals[: len(targets)]
    return out


def dirichlet_arm(n: int, vocab: list[str], rng: np.random.Generator,
                  pool_idx: np.ndarray) -> np.ndarray:
    o_idx = vocab.index("O")
    out = np.zeros((n, len(vocab)), dtype=np.float32)
    for s in range(n):
        top = rng.choice(pool_idx, size=3, replace=False)
        f = rng.dirichlet(np.ones(3))
        o_frac = rng.uniform(0.0, 1.0)
        out[s, top] = f * (1.0 - o_frac)
        out[s, o_idx] = o_frac
    return out


def main() -> None:
    vocab = C.element_vocab()
    o_idx = vocab.index("O")
    whitelist_idx = np.array(
        [i for i, e in enumerate(vocab) if e in C.COMMON_CERAMIC_ELEMENTS and e != "O"]
    )
    non_o_idx = np.array([i for i in range(len(vocab)) if i != o_idx])
    w = training_element_weights(vocab)
    w[o_idx] = 0.0
    freq_probs = w[non_o_idx] / w[non_o_idx].sum()
    freq41_probs = w[whitelist_idx] / w[whitelist_idx].sum()

    print(f"whitelist cation pool = {len(whitelist_idx)}, full non-O pool = {len(non_o_idx)}")

    rows = []
    for seed in SEEDS:
        decoded, _, _ = C.cvae_decode(N, seed, jitter=JITTER)
        rng = np.random.default_rng(10_000 + seed)
        arms = {
            "cvae": decoded,
            "rand_uniform41": relabel(decoded, vocab, rng, whitelist_idx, None),
            "rand_freq77": relabel(decoded, vocab, rng, non_o_idx, freq_probs),
            "rand_freq41": relabel(decoded, vocab, rng, whitelist_idx, freq41_probs),
            "rand_uniform77": relabel(decoded, vocab, rng, non_o_idx, None),
            "rand_dirichlet41": dirichlet_arm(N, vocab, rng, whitelist_idx),
        }
        for name, vec in arms.items():
            cand = C.decode_batch(vec, vocab)
            m = C.score_frame(cand, vocab)
            m["arm"] = name
            m["seed"] = seed
            m["whitelist_reject_rate"] = float(
                (cand["status"] == "risk elements present").mean()
            )
            m["no_charge_balance_rate"] = float(
                (cand["status"] == "no charge-balanced oxygen count found").mean()
            )
            rows.append(m)
            print(seed, name, round(m["validity_rate"], 4), m["unique_valid_count"],
                  m["unique_formula_count_all"], round(m["mean_pairwise_composition_distance"], 4))
            if seed == SEEDS[0]:
                cand.to_csv(C.RESULTS_DIR / f"a2_baseline_{name}_seed{seed}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "a2_random_baseline_runs.csv", index=False)

    cols = ["validity_rate", "uniqueness_rate", "novelty_rate", "unique_valid_count",
            "novel_unique_count", "mean_pairwise_composition_distance",
            "unique_formula_count_all", "modal_formula_share_all",
            "whitelist_reject_rate", "no_charge_balance_rate"]
    summ = df.groupby("arm")[cols].agg(["mean", "std"])
    summ.to_csv(C.RESULTS_DIR / "a2_random_baseline_summary.csv")
    print(summ.to_string())
    (C.RESULTS_DIR / "a2_random_baseline_summary.json").write_text(
        json.dumps(
            {arm: {c: {"mean": float(g[c].mean()), "std": float(g[c].std(ddof=1))} for c in cols}
             for arm, g in df.groupby("arm")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
