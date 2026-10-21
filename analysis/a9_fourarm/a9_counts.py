"""a9_counts.py -- generation and count metrics for all four arms, one code path.

For every arm and seed (0-4) 20,000 proposals are turned into 77-element fraction vectors
and decoded with a2_common.decode_batch (-> decode_composition_vector, byte-identical);
metrics come from a2_common.score_frame (validity, unique valid, uniqueness, novelty
against the training table, mean pairwise composition distance among novel unique valid
formulas -- a2's definition, exact below 5,000 pairs, 5,000-pair estimate above).
Substitution families use a3's family_key (same anonymised stoichiometry + same multiset
of periodic-table groups on the cation sites).

Arms A and B are recomputed with a2's own code (a2_exp2_random_baseline.relabel) and
a2's RNG stream (default_rng(10000 + seed), arms drawn in a2's order) so seeds 0-3 must
reproduce a2_random_baseline_runs.csv exactly; arm D uses a8's checkpoint and
a2_common.cvae_decode (a8 block A) so seeds 0-3 must reproduce
a8_downstream_fb0.5_b0.01_generation_runs.csv exactly.  Both checks are written out.

NOTE (design inherited from a2, stated in the report): arms A and B keep each sample's
decoded fraction profile from the DEPOSITED CVAE (number of cations, cation-fraction
pattern, oxygen fraction) and randomise only which element carries each fraction.  The
model-free Dirichlet arm (a2 rand_dirichlet41) is computed alongside as a sensitivity row.

Outputs: a9_counts_runs.csv, a9_counts_summary.{csv,json}, a9_population_{A,B,C,D}.csv,
a9_arm_c_proposals_seed0.csv, a9_reproduction_check.csv
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys
sys.path.insert(0, _CCID_ROOT + "/analysis/a9_fourarm")
import a9_common as K  # noqa: E402

import json
import time

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition

import a2_common as C
import a2_exp2_random_baseline as E2
import a8_common as A8
import a9_arm_c_substitution as SUB
from composition_cvae_common import canonical_formula
from generate_composition_candidates import composition_vector

torch.set_num_threads(1)
R = K.RESULTS


def reduced_canon(f: str) -> str:
    c = Composition(f).reduced_composition
    return canonical_formula({str(e): float(c[e]) for e in c})


def run_metrics(cand: pd.DataFrame, vocab, arm: str, seed: int) -> dict:
    m = C.score_frame(cand, vocab)
    uv = cand[cand["valid"]].drop_duplicates("canonical_formula")
    m["n_families_unique_valid"] = int(uv["canonical_formula"].map(K.family_id).nunique())
    m["whitelist_reject_rate"] = float((cand["status"] == "risk elements present").mean())
    m["no_charge_balance_rate"] = float(
        (cand["status"] == "no charge-balanced oxygen count found").mean())
    m["non_vocab_rate"] = float(
        (cand["status"] == "element outside 77-element vocabulary").mean())
    m["arm"] = arm
    m["seed"] = seed
    return m


def arms_ab(vocab, seeds):
    """a2_exp2_random_baseline.main(), restricted to the arms we need but with the RNG
    consumed in exactly a2's order so the streams are identical."""
    o_idx = vocab.index("O")
    whitelist_idx = np.array([i for i, e in enumerate(vocab)
                              if e in C.COMMON_CERAMIC_ELEMENTS and e != "O"])
    non_o_idx = np.array([i for i in range(len(vocab)) if i != o_idx])
    w = E2.training_element_weights(vocab)
    w[o_idx] = 0.0
    freq_probs = w[non_o_idx] / w[non_o_idx].sum()
    freq41_probs = w[whitelist_idx] / w[whitelist_idx].sum()
    out = {}
    for seed in seeds:
        decoded, _, _ = C.cvae_decode(K.N, seed, jitter=K.JITTER)   # deposited model profile
        rng = np.random.default_rng(10_000 + seed)
        u41 = E2.relabel(decoded, vocab, rng, whitelist_idx, None)
        _f77 = E2.relabel(decoded, vocab, rng, non_o_idx, freq_probs)       # a2 order
        f41 = E2.relabel(decoded, vocab, rng, whitelist_idx, freq41_probs)
        _u77 = E2.relabel(decoded, vocab, rng, non_o_idx, None)             # a2 order
        dr41 = E2.dirichlet_arm(K.N, vocab, rng, whitelist_idx)
        out[seed] = {"A": C.decode_batch(u41, vocab), "B": C.decode_batch(f41, vocab),
                     "A_dirichlet": C.decode_batch(dr41, vocab)}
    return out


def arm_c(vocab, seeds):
    parents = SUB.build_parents()
    vset = set(vocab)
    out = {}
    for seed in seeds:
        g = SUB.generate(seed, K.N, parents)
        vec = np.vstack([composition_vector(f, vocab) for f in g["pre_filter_formula"]])
        cand = C.decode_batch(vec, vocab)
        nonv = np.array([not set(Composition(f).get_el_amt_dict()) <= vset
                         for f in g["pre_filter_formula"]])
        cand.loc[nonv, ["valid", "charge_balanced"]] = False
        cand.loc[nonv, "status"] = "element outside 77-element vocabulary"
        cand.loc[nonv, "canonical_formula"] = g.loc[nonv, "pre_filter_canonical"].values
        cand.loc[nonv, "formula"] = g.loc[nonv, "pre_filter_formula"].values
        for col in ("parent", "parent_source", "substitution", "n_substituted_sites",
                    "joint_prob", "pre_filter_formula"):
            cand[col] = g[col].values
        cand["pre_filter_reduced"] = [reduced_canon(f) for f in g["pre_filter_formula"]]
        cand["parent_reduced"] = [reduced_canon(f) for f in g["parent"]]
        red_out = [reduced_canon(f) if v else "" for f, v in
                   zip(cand["canonical_formula"], cand["valid"])]
        cand["filter_kept_substituted_formula"] = [v and r == p for v, r, p in
                                                   zip(cand["valid"], red_out,
                                                       cand["pre_filter_reduced"])]
        cand["filter_output_is_parent"] = [v and r == p for v, r, p in
                                           zip(cand["valid"], red_out, cand["parent_reduced"])]
        cand["n_parent_cations"] = [len(Composition(p).elements) - 1 for p in cand["parent"]]
        out[seed] = cand
    return out, parents


def arm_d(vocab, seeds):
    A8.use_checkpoint(K.ARM_D_CKPT)
    out = {}
    for seed in seeds:
        dec, _, _ = C.cvae_decode(K.N, seed, jitter=K.JITTER)
        out[seed] = C.decode_batch(dec, vocab)
    return out


def pool(frames: dict, arm: str, known) -> pd.DataFrame:
    vs = []
    for seed, cand in frames.items():
        v = cand[cand["valid"]].copy()
        v["seed"] = seed
        vs.append(v)
    v = pd.concat(vs, ignore_index=True)
    v["canonical_formula"] = v["canonical_formula"].astype(str)
    g = v.groupby("canonical_formula", sort=False)
    u = v.drop_duplicates("canonical_formula").copy()
    u["n_draws"] = u["canonical_formula"].map(g.size()).astype(int)
    u["n_seeds"] = u["canonical_formula"].map(g["seed"].nunique()).astype(int)
    u["population"] = arm
    u["reduced_formula"] = [Composition(f).reduced_formula for f in u["canonical_formula"]]
    u["in_training_set"] = [f in known for f in u["canonical_formula"]]
    u["family_key"] = u["canonical_formula"].map(K.family_id)
    keep = ["population", "canonical_formula", "reduced_formula", "formula", "elements",
            "n_draws", "n_seeds", "seed", "sample_id", "in_training_set", "family_key"]
    extra = [c for c in ("parent", "substitution", "n_substituted_sites",
                         "pre_filter_formula", "filter_kept_substituted_formula",
                         "filter_output_is_parent") if c in u.columns]
    return u[keep + extra].reset_index(drop=True)


def main():
    t0 = time.time()
    _, meta = C.get_model()                      # deposited model (arms A, B profiles)
    vocab = list(meta["element_vocab"])
    _, known = C.get_known()
    rows, frames = [], {}

    ab = arms_ab(vocab, K.SEEDS)
    for arm in ("A", "B", "A_dirichlet"):
        frames[arm] = {s: ab[s][arm] for s in K.SEEDS}
    print(f"arms A/B done {time.time()-t0:.0f}s", flush=True)

    frames["C"], parents = arm_c(vocab, K.SEEDS)
    frames["C"][0].to_csv(R / "a9_arm_c_proposals_seed0.csv", index=False)
    print(f"arm C done {time.time()-t0:.0f}s", flush=True)

    vocab_d = None
    frames["D"] = arm_d(vocab, K.SEEDS)
    vocab_d = list(C._CACHE["metadata"]["element_vocab"])
    assert vocab_d == vocab, "arm D checkpoint vocabulary differs from the deposited one"
    print(f"arm D done {time.time()-t0:.0f}s", flush=True)

    for arm, fr in frames.items():
        for s, cand in fr.items():
            m = run_metrics(cand, vocab, arm, s)
            if arm == "C":
                v = cand[cand["valid"]]
                m["c_frac_valid_filter_kept_substituted_formula"] = float(
                    v["filter_kept_substituted_formula"].mean())
                m["c_frac_valid_filter_output_is_parent"] = float(
                    v["filter_output_is_parent"].mean())
                m["c_frac_valid_in_training"] = float(
                    v["canonical_formula"].isin(known).mean())
                m["c_mean_substituted_sites"] = float(cand["n_substituted_sites"].mean())
            rows.append(m)
            print(arm, s, round(m["validity_rate"], 4), m["unique_valid_count"],
                  round(m["novelty_rate"], 4), m["n_families_unique_valid"], flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(R / "a9_counts_runs.csv", index=False)

    # pooled unique valid populations
    pooled = {}
    for arm in ("A", "B", "C", "D"):
        p = pool(frames[arm], arm, known)
        p.to_csv(R / f"a9_population_{arm}.csv", index=False)
        pooled[arm] = p
    pdir = pool(frames["A_dirichlet"], "A_dirichlet", known)

    # summary
    cols = ["validity_rate", "valid_count", "unique_valid_count", "uniqueness_rate",
            "novel_unique_count", "novelty_rate", "mean_pairwise_composition_distance",
            "n_families_unique_valid", "unique_formula_count_all",
            "whitelist_reject_rate", "no_charge_balance_rate", "non_vocab_rate"]
    summ = {}
    for arm, g in runs.groupby("arm"):
        d = {c: {"mean": float(g[c].mean()), "std": float(g[c].std(ddof=1))} for c in cols}
        p = pooled.get(arm, pdir if arm == "A_dirichlet" else None)
        d["pooled_unique_valid"] = int(len(p))
        d["pooled_novel_unique"] = int((~p["in_training_set"]).sum())
        d["pooled_novelty_rate"] = float((~p["in_training_set"]).mean())
        d["pooled_n_families"] = int(p["family_key"].nunique())
        d["seeds"] = K.SEEDS
        if arm == "C":
            for c in [c for c in g.columns if c.startswith("c_")]:
                d[c] = {"mean": float(g[c].mean()), "std": float(g[c].std(ddof=1))}
        summ[arm] = d
    summ["_arm_c_parents"] = {k: v for k, v in parents.items()
                              if k not in ("parents",)}
    (R / "a9_counts_summary.json").write_text(json.dumps(summ, indent=2))
    flat = []
    for arm in ("A", "B", "C", "D", "A_dirichlet"):
        d = summ[arm]
        row = {"arm": arm}
        for c in cols:
            row[c + "_mean"] = d[c]["mean"]; row[c + "_std"] = d[c]["std"]
        for c in ("pooled_unique_valid", "pooled_novel_unique", "pooled_novelty_rate",
                  "pooled_n_families"):
            row[c] = d[c]
        flat.append(row)
    pd.DataFrame(flat).to_csv(R / "a9_counts_summary.csv", index=False)

    # reproduction checks against a2 / a8
    chk = []
    a2 = pd.read_csv(R / "a2_random_baseline_runs.csv")
    a8 = pd.read_csv(R / "a8_downstream_fb0.5_b0.01_generation_runs.csv")
    for arm, ref, refarm in (("A", a2, "rand_uniform41"), ("B", a2, "rand_freq41"),
                             ("A_dirichlet", a2, "rand_dirichlet41"), ("D", a8, None)):
        for s in K.SEEDS:
            r = ref[(ref["seed"] == s) & ((ref["arm"] == refarm) if refarm else True)]
            mine = runs[(runs["arm"] == arm) & (runs["seed"] == s)].iloc[0]
            if len(r) == 0:
                chk.append(dict(arm=arm, seed=s, reference="none (new seed)"))
                continue
            r = r.iloc[0]
            chk.append(dict(arm=arm, seed=s, reference=refarm or "a8 fb0.5_b0.01",
                            valid_ref=int(r["valid_count"]), valid_a9=int(mine["valid_count"]),
                            unique_valid_ref=int(r["unique_valid_count"]),
                            unique_valid_a9=int(mine["unique_valid_count"]),
                            exact_match=bool(int(r["valid_count"]) == int(mine["valid_count"]) and
                                             int(r["unique_valid_count"]) ==
                                             int(mine["unique_valid_count"]))))
    pd.DataFrame(chk).to_csv(R / "a9_reproduction_check.csv", index=False)
    print(pd.DataFrame(chk).to_string())
    print(pd.DataFrame(flat)[["arm", "validity_rate_mean", "unique_valid_count_mean",
                              "novelty_rate_mean", "pooled_unique_valid",
                              "pooled_n_families"]].to_string())
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
