"""a8 experiment 0 — reproduce the posterior-collapse pathology on the DEPOSITED checkpoint.

Three measurements, all on dgx/outputs_dgx/composition_cvae_checkpoint.pt:
  1. per-dimension KL of the aggregate posterior over the full 25,148-record table
  2. active-unit counts at several thresholds
  3. fixed-condition collapse: 20,000 independent z ~ N(0,I) draws decoded at the
     paper's own target condition (0.9, 0.8) held constant -> how many formulas?
"""
from __future__ import annotations

import json

import numpy as np
import torch

import a8_common as A
import a2_common as C

N = 20000


def fixed_condition_collapse(model, meta, cond=(0.9, 0.8), n=N, seed=123) -> dict:
    torch.manual_seed(seed)
    z = torch.randn(n, int(meta["latent_dim"]))
    c = torch.tensor(np.tile(cond, (n, 1)).astype(np.float32))
    with torch.no_grad():
        dec = model.decode(z, c).numpy()[:, : int(meta["composition_dim"])]
    vocab = list(meta["element_vocab"])
    cand = C.decode_batch(dec, vocab)
    m = C.score_frame(cand, vocab)
    vc = cand["canonical_formula"].astype(str).value_counts()
    m.update({
        "condition": list(cond),
        "n_latent_draws": n,
        "distinct_decoded_formulas": int(vc.size),
        "modal_formula": str(vc.index[0]),
        "modal_share": float(vc.iloc[0] / len(cand)),
        "top5": {str(k): int(v) for k, v in vc.head(5).items()},
        "decoder_output_std_over_z_mean": float(dec.std(axis=0).mean()),
        "decoder_output_std_over_z_max": float(dec.std(axis=0).max()),
    })
    return m


def main() -> None:
    model, meta, ckpt = A.use_deposited()
    out = {
        "checkpoint": str(A.PROJECT_ROOT / "outputs_dgx" / "composition_cvae_checkpoint.pt"),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "latent": A.latent_diagnostics(model, meta),
        "fixed_condition_0.9_0.8": fixed_condition_collapse(model, meta),
        "fixed_condition_0.5_0.5": fixed_condition_collapse(model, meta, cond=(0.5, 0.5)),
    }
    p = A.RESULTS_DIR / "a8_baseline_pathology.json"
    p.write_text(json.dumps(out, indent=2))
    L = out["latent"]
    print("epoch", out["checkpoint_epoch"])
    print("total KL nats:", round(L["total_kl_nats"], 4))
    print("active (KL>0.01):", L["active_units_kl_gt_0.01"], "of", L["latent_dim"])
    print("KL per dim sorted:", np.round(L["kl_per_dim_nats_sorted"], 5).tolist())
    for k in ("fixed_condition_0.9_0.8", "fixed_condition_0.5_0.5"):
        f = out[k]
        print(k, "distinct formulas from", f["n_latent_draws"], "draws:",
              f["distinct_decoded_formulas"], "modal", f["modal_formula"], round(f["modal_share"], 4),
              "validity", round(f["validity_rate"], 4))
    print("wrote", p)


if __name__ == "__main__":
    main()
