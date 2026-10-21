"""Experiment 6 - direct posterior-collapse diagnostics on the deposited checkpoint.

Encodes the full training table and reports, per latent dimension, the mean KL to the
prior and the variance of the posterior mean. A dimension with KL below ~0.01 nats is
conventionally called collapsed / inactive.
"""
from __future__ import annotations

import json

import numpy as np
import torch

import a2_common as C
from composition_cvae_common import featurize


def main() -> None:
    model, meta = C.get_model()
    table, _ = C.get_known()
    vocab = list(meta["element_vocab"])
    feats, conds = featurize(table, vocab)
    x = torch.tensor(feats, dtype=torch.float32)
    c = torch.tensor(conds, dtype=torch.float32)
    with torch.no_grad():
        mu, logvar = model.encode(x, c)
    kl_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0).numpy()
    out = {
        "n_records_encoded": int(len(x)),
        "latent_dim": int(meta["latent_dim"]),
        "kl_per_dim_nats": [float(v) for v in np.sort(kl_dim)[::-1]],
        "total_kl_nats": float(kl_dim.sum()),
        "active_units_kl_gt_0.01": int((kl_dim > 0.01).sum()),
        "active_units_kl_gt_0.1": int((kl_dim > 0.1).sum()),
        "var_of_posterior_mean_per_dim": [float(v) for v in np.sort(mu.var(dim=0).numpy())[::-1]],
        "free_bits_threshold_used_in_training_nats": 0.5 * int(meta["latent_dim"]),
        "note": (
            "cvae_loss clamps the *total* KL at free_bits = 0.5 * latent_dim = 16 nats, so "
            "the objective stops penalising KL below 16 nats in total; the KL term is also "
            "weighted by beta <= 0.1 against an unnormalised sum-reduced reconstruction loss."
        ),
    }
    (C.RESULTS_DIR / "a2_latent_diagnostics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("kl_per_dim_nats", "var_of_posterior_mean_per_dim")}, indent=2))
    print("KL per dim (sorted):", np.round(np.sort(kl_dim)[::-1], 4).tolist())


if __name__ == "__main__":
    main()
