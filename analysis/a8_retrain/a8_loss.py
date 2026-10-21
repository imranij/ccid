"""The corrected CVAE loss (per-dimension free bits).

This is the reference implementation of the patch delivered as
`a8_composition_cvae_common_free_bits.patch`. It is kept as a standalone module so
the retraining sweep can vary `free_bits_per_dim` and `beta_max` without editing the
project tree.

Original (buggy) form in composition_cvae_common.cvae_loss:

    free_bits = 0.5 * mu.shape[1]                      # ONE scalar = 16.0 nats
    kl = -0.5 * torch.sum(1 + logvar - mu**2 - exp(logvar))   # summed over ALL dims
    loss = comp + w*prop + beta * clamp(kl, min=free_bits)

Clamping the TOTAL KL at a floor makes the term constant (zero gradient) whenever the
total KL is below the floor, so every latent dimension may collapse at no cost.

Corrected form: the floor is applied PER DIMENSION before summing, which is the
definition in Kingma et al. 2016 ("free bits"). Scale is preserved exactly: the
original `kl` is a sum over batch AND dims, so the corrected term is multiplied by the
batch size to sit in the same units as comp_loss (reduction="sum").
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cvae_loss_fixed(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    composition_dim: int,
    epoch: int,
    max_epochs: int,
    property_weight: float = 2.0,
    beta_max: float = 0.1,
    free_bits_per_dim: float = 0.05,
):
    comp_loss = F.mse_loss(recon_x[:, :composition_dim], x[:, :composition_dim], reduction="sum")
    prop_loss = F.mse_loss(recon_x[:, composition_dim:], x[:, composition_dim:], reduction="sum")

    kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (batch, latent)
    kl = kl_elem.sum()                                          # unclamped total, for logging
    kl_per_dim = kl_elem.mean(dim=0)                            # (latent,) nats/sample/dim
    kl_term = x.shape[0] * torch.clamp(kl_per_dim, min=free_bits_per_dim).sum()

    anneal_epochs = max(1, min(50, max_epochs // 2))
    beta = min(beta_max, beta_max * (epoch / anneal_epochs))
    loss = comp_loss + property_weight * prop_loss + beta * kl_term
    return loss, comp_loss, prop_loss, kl
