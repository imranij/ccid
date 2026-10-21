"""a8 — full downstream evaluation of a retrained checkpoint, using the a2 harness.

Reproduces, for the retrained model, every control the a2 agent ran on the deposited
checkpoint, so the numbers are directly comparable:

  block A  generation at the archived settings (target 0.9/0.8, jitter 0.30,
           z ~ N(0,I), 20,000 samples x 5 seeds): validity, uniqueness, novelty,
           mean pairwise composition distance, distinct decoded formulas.
  block B  random element-selection controls through the identical filter path,
           recomputed from THIS model's decoded fraction profiles
           (a2_exp2_random_baseline.relabel / dirichlet_arm, verbatim).
  block C  unconditional / condition-destroyed controls (a2 exp-3 arms).
  block D  conditional-generation sweep over a (stability, gap) grid with a shared
           latent draw, with mutual information and a label-permutation null
           (a2 exp-4 estimators, verbatim).
  block E  decoder-capacity experiment: top_cations in {2,3,4,5}.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch

import a8_common as A
import a2_common as C
import a2_exp2_random_baseline as E2
import a2_exp4_condition_sweep as E4

N = 20000
JITTER = 0.30


def block_a(meta, seeds, top_cations=3, pre="a8"):
    vocab = list(meta["element_vocab"])
    rows = []
    for s in seeds:
        dec, cond, z = C.cvae_decode(N, s, jitter=JITTER)
        cand = C.decode_batch(dec, vocab, top_cations=top_cations)
        m = C.score_frame(cand, vocab)
        m["seed"] = s
        rows.append(m)
        if s == seeds[0]:
            cand.to_csv(A.RESULTS_DIR / f"{pre}_candidates_seed{s}.csv", index=False)
    return pd.DataFrame(rows)


def block_b(meta, seeds):
    vocab = list(meta["element_vocab"])
    o_idx = vocab.index("O")
    whitelist_idx = np.array([i for i, e in enumerate(vocab)
                              if e in C.COMMON_CERAMIC_ELEMENTS and e != "O"])
    non_o_idx = np.array([i for i in range(len(vocab)) if i != o_idx])
    w = E2.training_element_weights(vocab)
    w[o_idx] = 0.0
    freq_probs = w[non_o_idx] / w[non_o_idx].sum()
    freq41 = w[whitelist_idx] / w[whitelist_idx].sum()
    rows = []
    for s in seeds:
        dec, _, _ = C.cvae_decode(N, s, jitter=JITTER)
        rng = np.random.default_rng(10_000 + s)
        arms = {
            "cvae_fixed": dec,
            "rand_uniform41": E2.relabel(dec, vocab, rng, whitelist_idx, None),
            "rand_freq41": E2.relabel(dec, vocab, rng, whitelist_idx, freq41),
            "rand_uniform77": E2.relabel(dec, vocab, rng, non_o_idx, None),
            "rand_freq77": E2.relabel(dec, vocab, rng, non_o_idx, freq_probs),
            "rand_dirichlet41": E2.dirichlet_arm(N, vocab, rng, whitelist_idx),
        }
        for name, vec in arms.items():
            cand = C.decode_batch(vec, vocab)
            m = C.score_frame(cand, vocab)
            m["arm"] = name
            m["seed"] = s
            m["whitelist_reject_rate"] = float((cand["status"] == "risk elements present").mean())
            m["no_charge_balance_rate"] = float(
                (cand["status"] == "no charge-balanced oxygen count found").mean())
            rows.append(m)
            print("  B", s, name, round(m["validity_rate"], 4), m["unique_valid_count"], flush=True)
    return pd.DataFrame(rows)


def block_c(model, meta, seeds):
    vocab = list(meta["element_vocab"])
    table, _ = C.get_known()
    train_mean = (float(table["stability_utility"].mean()), float(table["gap_utility"].mean()))
    rows = []
    for s in seeds:
        _, cond, z = C.cvae_decode(N, s, jitter=JITTER)
        rng = np.random.default_rng(50_000 + s)
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
            m["seed"] = s
            rows.append(m)
            print("  C", s, name, round(m["validity_rate"], 4), m["unique_valid_count"],
                  m["unique_formula_count_all"], flush=True)
    return pd.DataFrame(rows), train_mean


def block_d(model, meta, n_per_cell=4000, seed=123):
    vocab = list(meta["element_vocab"])
    cdim = int(meta["composition_dim"])
    STAB, GAP = [0.2, 0.5, 0.8, 1.0], [0.5, 0.8, 1.0]
    torch.manual_seed(seed)
    z = torch.randn(n_per_cell, int(meta["latent_dim"]))
    cells, frames = [], []
    for s, g in itertools.product(STAB, GAP):
        c = np.tile([s, g], (n_per_cell, 1)).astype(np.float32)
        with torch.no_grad():
            dec = model.decode(z, torch.tensor(c)).numpy()[:, :cdim]
        cand = C.decode_batch(dec, vocab)
        cand["cell"] = f"s{s}_g{g}"
        m = C.score_frame(cand, vocab)
        m.update({"cell": f"s{s}_g{g}", "stability_utility": s, "gap_utility": g})
        cells.append(m)
        frames.append(cand)
        print("  D", s, g, round(m["validity_rate"], 4), m["unique_formula_count_all"],
              round(m["modal_formula_share_all"], 4), flush=True)
    allc = pd.concat(frames, ignore_index=True)
    labels = allc["cell"].to_numpy()
    formulas = allc["canonical_formula"].astype(str).to_numpy()
    el_lab, el_el = [], []
    for lab, elems in zip(labels, allc["elements"].astype(str).to_numpy()):
        for e in elems.split():
            if e != "O":
                el_lab.append(lab); el_el.append(e)
    el_lab, el_el = np.array(el_lab), np.array(el_el)

    def entropy(v):
        p = pd.Series(v).value_counts(normalize=True).to_numpy()
        return float(-(p * np.log2(p)).sum())

    mi_f = E4.mutual_information(labels, formulas)
    mi_f0, mi_f0s = E4.permutation_null(labels, formulas)
    mi_e = E4.mutual_information(el_lab, el_el)
    mi_e0, mi_e0s = E4.permutation_null(el_lab, el_el)
    cell_names = [c["cell"] for c in cells]
    pivot = allc.pivot_table(index="sample_id", columns="cell",
                             values="canonical_formula", aggfunc="first")
    switch = [float((pivot[a] != pivot[b]).mean()) for a, b in itertools.combinations(cell_names, 2)]
    summary = {
        "n_per_cell": n_per_cell, "shared_latent_draw": True,
        "mutual_information_bits": {
            "cell_vs_formula": mi_f, "cell_vs_formula_permutation_null_mean": mi_f0,
            "cell_vs_formula_permutation_null_sd": mi_f0s,
            "cell_vs_element": mi_e, "cell_vs_element_permutation_null_mean": mi_e0,
            "cell_vs_element_permutation_null_sd": mi_e0s,
            "max_possible_bits_12_cells": float(np.log2(len(cell_names))),
            "H_formula_bits": entropy(formulas), "H_element_bits": entropy(el_el),
            "normalised_MI_formula": mi_f / max(1e-12, entropy(formulas)),
            "normalised_MI_element": mi_e / max(1e-12, entropy(el_el)),
        },
        "paired_formula_switch_rate": {"mean": float(np.mean(switch)),
                                       "min": float(np.min(switch)), "max": float(np.max(switch))},
        "modal_formula_per_cell": {c: allc.loc[allc["cell"] == c, "canonical_formula"]
                                   .value_counts().index[0] for c in cell_names},
    }
    return pd.DataFrame(cells), summary


def block_e(meta, seeds, tops=(2, 3, 4, 5)):
    vocab = list(meta["element_vocab"])
    rows = []
    for s in seeds:
        dec, _, _ = C.cvae_decode(N, s, jitter=JITTER)
        for t in tops:
            cand = C.decode_batch(dec, vocab, top_cations=t)
            m = C.score_frame(cand, vocab)
            m["top_cations"] = t
            m["seed"] = s
            rows.append(m)
            print("  E", s, "top_cations", t, round(m["validity_rate"], 4),
                  m["unique_valid_count"], m["unique_formula_count_all"], flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 0, 1, 2, 3])
    ap.add_argument("--deposited", action="store_true")
    ap.add_argument("--blocks", default="ABCDE")
    args = ap.parse_args()

    if args.deposited:
        model, meta, _ = A.use_deposited()
    else:
        model, meta, _ = A.use_checkpoint(A.CKPT_DIR / f"{args.tag}_checkpoint.pt")
    pre = args.prefix or f"a8_downstream_{args.tag}"
    R = A.RESULTS_DIR
    out = {"tag": args.tag, "seeds": args.seeds}

    if "A" in args.blocks:
        a = block_a(meta, args.seeds, pre=pre)
        a.to_csv(R / f"{pre}_generation_runs.csv", index=False)
        print(a[["seed", "validity_rate", "unique_valid_count", "novelty_rate",
                 "mean_pairwise_composition_distance", "unique_formula_count_all"]].to_string())
    if "B" in args.blocks:
        b = block_b(meta, args.seeds[:3])
        b.to_csv(R / f"{pre}_random_baseline_runs.csv", index=False)
    if "C" in args.blocks:
        c, tm = block_c(model, meta, args.seeds[:3])
        c.to_csv(R / f"{pre}_unconditional_runs.csv", index=False)
        out["training_mean_conditions"] = tm
    if "D" in args.blocks:
        d, dsum = block_d(model, meta)
        d.to_csv(R / f"{pre}_condition_sweep_cells.csv", index=False)
        out["condition_sweep"] = dsum
    if "E" in args.blocks:
        e = block_e(meta, args.seeds[:3])
        e.to_csv(R / f"{pre}_top_cations.csv", index=False)
    (R / f"{pre}_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print("done", pre)


if __name__ == "__main__":
    main()
