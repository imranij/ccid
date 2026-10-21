"""Experiment 5 - diversity, mode concentration, and the true cause of low uniqueness.

Part A  latent-sigma sweep (z ~ N(0, sigma^2 I)) at fixed condition jitter, and
        condition-jitter sweep at fixed sigma. Metrics: unique valid formulas per
        1,000 valid samples, element-frequency Shannon entropy, number of distinct
        stoichiometry classes, quantiles of the pairwise composition distance.
Part B  decoder reachability. Enumerates every formula decode_composition_vector can
        emit, to separate "the decoder is coarse" from "the model mode-collapsed".
"""
from __future__ import annotations

import itertools
import json
import math
from collections import Counter

import numpy as np
import pandas as pd

import a2_common as C
from composition_cvae_common import (
    COMMON_CERAMIC_ELEMENTS,
    HIGH_RISK_ELEMENTS,
    charge_balance_possible,
    canonical_formula,
    formula_from_counts,
    parse_formula,
)

N = 20000
SEEDS = [123, 0, 1]
SIGMAS = [0.08, 0.3, 0.5, 1.0, 2.0]
JITTERS = [0.08, 0.3, 0.5, 1.0]


def stoich_class(formula: str) -> str:
    """Stoichiometry class = sorted integer count pattern, cations then O."""
    counts = parse_formula(formula)
    cat = sorted((int(v) for k, v in counts.items() if k != "O"), reverse=True)
    return "-".join(map(str, cat)) + f":O{int(counts.get('O', 0))}"


def element_entropy(cand: pd.DataFrame) -> float:
    c = Counter()
    for elems in cand["elements"].astype(str):
        for e in elems.split():
            if e != "O":
                c[e] += 1
    tot = sum(c.values())
    if not tot:
        return 0.0
    p = np.array(list(c.values()), dtype=float) / tot
    return float(-(p * np.log2(p)).sum())


def diversity_block(cand: pd.DataFrame, vocab: list[str]) -> dict:
    m = C.score_frame(cand, vocab)
    valid = cand[cand["valid"]]
    uv = valid.drop_duplicates("canonical_formula")
    vecs = (np.vstack([C.composition_vector(f, vocab) for f in uv["formula"]])
            if len(uv) else np.zeros((0, len(vocab))))
    if len(vecs) >= 2:
        n = len(vecs)
        rng = np.random.default_rng(11)
        if n * (n - 1) // 2 <= 20000:
            idx = list(itertools.combinations(range(n), 2))
            d = np.array([np.linalg.norm(vecs[i] - vecs[j]) for i, j in idx])
        else:
            i = rng.integers(0, n, 20000); j = rng.integers(0, n, 20000)
            k = i != j
            d = np.linalg.norm(vecs[i[k]] - vecs[j[k]], axis=1)
        q = np.percentile(d, [5, 25, 50, 75, 95])
    else:
        q = [0.0] * 5
    m.update({
        "element_entropy_bits_all": element_entropy(cand),
        "element_entropy_bits_valid": element_entropy(valid),
        "n_elements_used_valid": len({e for s in valid["elements"].astype(str)
                                      for e in s.split() if e != "O"}),
        "n_stoich_classes_valid": int(pd.Series(
            [stoich_class(f) for f in valid["formula"]]).nunique()) if len(valid) else 0,
        "unique_valid_per_1000_valid": 1000.0 * len(uv) / max(1, len(valid)),
        "pdist_p5": float(q[0]), "pdist_p25": float(q[1]), "pdist_median": float(q[2]),
        "pdist_p75": float(q[3]), "pdist_p95": float(q[4]),
    })
    return m


def achievable_count_patterns(n_cat: int, max_cation_count: int = 4,
                              n_draw: int = 400000, seed: int = 3) -> set[tuple[int, ...]]:
    """Which gcd-reduced integer count patterns can the rounding step actually emit?"""
    rng = np.random.default_rng(seed)
    f = rng.dirichlet(np.ones(n_cat), size=n_draw)
    raw = np.maximum(1, np.rint(f * max_cation_count).astype(int))
    pats = set()
    for row in raw:
        g = row[0]
        for v in row[1:]:
            g = math.gcd(g, int(v))
        pats.add(tuple(sorted((row // max(g, 1)).tolist(), reverse=True)))
    return pats


def reachability() -> dict:
    """Enumerate the full set of formulas decode_composition_vector can emit."""
    cations = sorted(e for e in COMMON_CERAMIC_ELEMENTS if e != "O")
    out = {"whitelist_cation_pool": len(cations)}
    total_valid, total_attempted = set(), 0
    per_k = {}
    for k in (1, 2, 3):
        pats = achievable_count_patterns(k)
        valid_k = set()
        for combo in itertools.combinations(cations, k):
            for pat in pats:
                for perm in set(itertools.permutations(pat)):
                    counts = {e: int(c) for e, c in zip(combo, perm)}
                    total_attempted += 1
                    ok = [o for o in range(1, 13)
                          if charge_balance_possible({**counts, "O": o})]
                    for o in ok:
                        cc = {**counts, "O": o}
                        if not (set(cc) & HIGH_RISK_ELEMENTS) and not (
                                set(cc) - COMMON_CERAMIC_ELEMENTS):
                            valid_k.add(canonical_formula({k2: float(v2) for k2, v2 in cc.items()}))
        per_k[k] = {"count_patterns": sorted(map(list, pats)),
                    "reachable_valid_formulas": len(valid_k)}
        total_valid |= valid_k
        print(f"k={k}: patterns={sorted(map(list,pats))} reachable={len(valid_k)}")
    out["per_k"] = per_k
    out["total_reachable_valid_formulas"] = len(total_valid)
    out["cation_set_combinations_k3"] = math.comb(len(cations), 3)
    return out, total_valid


def main() -> None:
    vocab = C.element_vocab()
    rows = []
    for seed in SEEDS:
        for sigma in SIGMAS:
            dec, _, _ = C.cvae_decode(N, seed, jitter=0.30, latent_sigma=sigma)
            m = diversity_block(C.decode_batch(dec, vocab), vocab)
            m.update({"seed": seed, "latent_sigma": sigma, "condition_jitter": 0.30,
                      "sweep": "latent_sigma"})
            rows.append(m)
            print("sigma", sigma, seed, round(m["validity_rate"], 4),
                  m["unique_valid_count"], m["n_stoich_classes_valid"],
                  round(m["element_entropy_bits_valid"], 3))
        for jit in JITTERS:
            dec, _, _ = C.cvae_decode(N, seed, jitter=jit, latent_sigma=1.0)
            m = diversity_block(C.decode_batch(dec, vocab), vocab)
            m.update({"seed": seed, "latent_sigma": 1.0, "condition_jitter": jit,
                      "sweep": "condition_jitter"})
            rows.append(m)
            print("jitter", jit, seed, round(m["validity_rate"], 4),
                  m["unique_valid_count"], m["n_stoich_classes_valid"],
                  round(m["element_entropy_bits_valid"], 3))

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "a2_diversity_sweeps.csv", index=False)

    reach, reach_set = reachability()

    # how much of the reachable space does the CVAE actually cover?
    dec, _, _ = C.cvae_decode(N, 123, jitter=0.30)
    cand = C.decode_batch(dec, vocab)
    cvae_valid_formulas = set(cand.loc[cand["valid"], "canonical_formula"].astype(str))
    reach["cvae_valid_formulas_20k"] = len(cvae_valid_formulas)
    reach["coverage_fraction_of_reachable_space"] = (
        len(cvae_valid_formulas) / max(1, reach["total_reachable_valid_formulas"]))
    reach["cvae_formulas_outside_enumerated_reachable_set"] = sorted(
        cvae_valid_formulas - reach_set)[:20]
    # random-baseline coverage, for comparison
    b = pd.read_csv(C.RESULTS_DIR / "a2_baseline_rand_uniform41_seed123.csv")
    rand_valid = set(b.loc[b["valid"], "canonical_formula"].astype(str))
    reach["rand_uniform41_valid_formulas_20k"] = len(rand_valid)
    reach["rand_uniform41_coverage_fraction"] = (
        len(rand_valid) / max(1, reach["total_reachable_valid_formulas"]))

    (C.RESULTS_DIR / "a2_decoder_reachability.json").write_text(json.dumps(reach, indent=2))
    print(json.dumps({k: v for k, v in reach.items() if k != "per_k"}, indent=2))


if __name__ == "__main__":
    main()
