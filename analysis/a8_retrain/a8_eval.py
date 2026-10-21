"""a8 — evaluate one or more checkpoints through the a2 downstream harness.

For each checkpoint:
  * latent diagnostics   per-dimension KL over the full 25,148-record table, active units
  * reconstruction       test-split composition MSE (from the run's test_metrics.json)
  * generation           the archived generation path (target 0.9/0.8, jitter 0.30,
                         z ~ N(0,I), 20,000 samples) -> validity / uniqueness / novelty /
                         mean pairwise composition distance / distinct decoded formulas
  * fixed-condition      condition pinned at (0.9, 0.8), 20,000 latent draws: how many
                         distinct formulas does the decoder actually reach?
All filters are the project's own decode_composition_vector path via a2_common.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import a8_common as A
import a2_common as C

N = 20000
JITTER = 0.30


def fixed_condition(model, meta, cond, n=N, seed=123, top_cations=3) -> dict:
    torch.manual_seed(seed)
    z = torch.randn(n, int(meta["latent_dim"]))
    c = torch.tensor(np.tile(cond, (n, 1)).astype(np.float32))
    with torch.no_grad():
        dec = model.decode(z, c).numpy()[:, : int(meta["composition_dim"])]
    vocab = list(meta["element_vocab"])
    cand = C.decode_batch(dec, vocab, top_cations=top_cations)
    m = C.score_frame(cand, vocab)
    vc = cand["canonical_formula"].astype(str).value_counts()
    m.update({"modal_formula": str(vc.index[0]), "modal_share": float(vc.iloc[0] / len(cand)),
              "distinct_decoded_formulas": int(vc.size)})
    return m


def eval_one(tag: str, ckpt_path: Path, seeds=(123,), top_cations=3) -> dict:
    model, meta, ckpt = A.use_checkpoint(ckpt_path)
    lat = A.latent_diagnostics(model, meta)
    tm_path = ckpt_path.parent / f"{tag}_test_metrics.json"
    tm = json.loads(tm_path.read_text()) if tm_path.exists() else {}

    gen = []
    for s in seeds:
        dec, cond, z = C.cvae_decode(N, s, jitter=JITTER)
        cand = C.decode_batch(dec, list(meta["element_vocab"]), top_cations=top_cations)
        gen.append(C.score_frame(cand, list(meta["element_vocab"])))
    g = pd.DataFrame(gen)

    fc = fixed_condition(model, meta, (0.9, 0.8), top_cations=top_cations)
    row = {
        "tag": tag,
        "loss": meta.get("a8_loss", "deposited"),
        "free_bits_per_dim": meta.get("a8_free_bits_per_dim", float("nan")),
        "beta_max": meta.get("a8_beta_max", float("nan")),
        "top_cations": top_cations,
        "test_composition_mse": tm.get("composition_mse", float("nan")),
        "test_property_mse": tm.get("property_mse", float("nan")),
        "best_val_loss": tm.get("best_val_loss", float("nan")),
        "wall_clock_seconds": tm.get("wall_clock_seconds", float("nan")),
        "total_kl_nats": lat["total_kl_nats"],
        "active_units_kl_gt_0.01": lat["active_units_kl_gt_0.01"],
        "active_units_kl_gt_0.05": lat["active_units_kl_gt_0.05"],
        "active_units_kl_gt_0.1": lat["active_units_kl_gt_0.1"],
        "max_kl_dim_nats": lat["max_kl_dim_nats"],
        "kl_per_dim_sorted": json.dumps([round(v, 5) for v in lat["kl_per_dim_nats_sorted"]]),
        "n_seeds": len(seeds),
    }
    for c in ["validity_rate", "uniqueness_rate", "novelty_rate", "unique_valid_count",
              "novel_unique_count", "mean_pairwise_composition_distance",
              "unique_formula_count_all", "modal_formula_share_all"]:
        row[f"gen_{c}_mean"] = float(g[c].mean())
        row[f"gen_{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
    row.update({
        "fixed_cond_distinct_formulas": fc["distinct_decoded_formulas"],
        "fixed_cond_unique_valid": fc["unique_valid_count"],
        "fixed_cond_validity_rate": fc["validity_rate"],
        "fixed_cond_modal_formula": fc["modal_formula"],
        "fixed_cond_modal_share": fc["modal_share"],
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(A.RESULTS_DIR / "a8_sweep_results.csv"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[123])
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--include-deposited", action="store_true")
    ap.add_argument("--top-cations", type=int, default=3)
    args = ap.parse_args()

    jobs = []
    if args.include_deposited:
        jobs.append(("deposited", A.PROJECT_ROOT / "outputs_dgx" / "composition_cvae_checkpoint.pt"))
    tags = args.tags
    if tags is None:
        tags = sorted(p.name.replace("_checkpoint.pt", "") for p in A.CKPT_DIR.glob("*_checkpoint.pt"))
    for t in tags:
        jobs.append((t, A.CKPT_DIR / f"{t}_checkpoint.pt"))

    rows = []
    for tag, path in jobs:
        if not path.exists():
            print("MISSING", path)
            continue
        r = eval_one(tag, path, seeds=tuple(args.seeds), top_cations=args.top_cations)
        rows.append(r)
        print(f"{tag:16s} AU={r['active_units_kl_gt_0.01']:2d} KL={r['total_kl_nats']:8.3f} "
              f"recon={r['test_composition_mse']:.5f} valid={r['gen_validity_rate_mean']:.4f} "
              f"uniqvalid={r['gen_unique_valid_count_mean']:.0f} "
              f"allform={r['gen_unique_formula_count_all_mean']:.0f} "
              f"fixedcond={r['fixed_cond_distinct_formulas']}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
