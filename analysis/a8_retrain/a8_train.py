"""a8 — retrain the Composition-CVAE with corrected per-dimension free bits.

Single-process CPU re-implementation of dgx/code/train_composition_cvae_dgx.py.
Everything except the KL term is identical: same data table, same element vocab
(min_count=5), same seed-42 80/10/10 split, same architecture (79-dim input,
32 latent, 256 hidden), same AdamW(lr=1e-3), same CosineAnnealingLR(T_max=epochs),
same grad-norm clip 1.0, same batch size 128, same 150 epochs, same beta warm-up
(linear 0 -> beta_max over min(50, epochs//2) epochs), same property_weight=2.0.

--loss fixed   per-dimension free bits (the correction)
--loss buggy   the deposited total-KL clamp, reproduced here on CPU as a control
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import a8_common as A
from a8_loss import cvae_loss_fixed
from composition_cvae_common import (
    ArrayDataset,
    CompositionCVAE,
    build_element_vocab,
    cvae_loss,
    featurize,
    load_material_table,
    split_indices,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--loss", choices=["fixed", "buggy"], default="fixed")
    p.add_argument("--free-bits-per-dim", type=float, default=0.05)
    p.add_argument("--beta-max", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--min-element-count", type=int, default=5)
    p.add_argument("--property-weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def loss_fn(args):
    if args.loss == "fixed":
        def f(recon, x, mu, logvar, cdim, epoch, max_epochs):
            return cvae_loss_fixed(recon, x, mu, logvar, cdim, epoch, max_epochs,
                                   property_weight=args.property_weight,
                                   beta_max=args.beta_max,
                                   free_bits_per_dim=args.free_bits_per_dim)
    else:
        def f(recon, x, mu, logvar, cdim, epoch, max_epochs):
            return cvae_loss(recon, x, mu, logvar, cdim, epoch, max_epochs,
                             property_weight=args.property_weight,
                             beta_max=args.beta_max)
    return f


def evaluate(model, loader, cdim, epoch, max_epochs, f):
    model.eval()
    t = {"loss": 0.0, "comp": 0.0, "prop": 0.0, "kl": 0.0, "n": 0.0}
    with torch.no_grad():
        for x, c in loader:
            recon, mu, logvar = model(x, c)
            loss, comp, prop, kl = f(recon, x, mu, logvar, cdim, epoch, max_epochs)
            t["loss"] += float(loss); t["comp"] += float(comp)
            t["prop"] += float(prop); t["kl"] += float(kl); t["n"] += x.shape[0]
    d = max(1.0, t["n"])
    return {"loss": t["loss"] / d, "composition_mse": t["comp"] / d,
            "property_mse": t["prop"] / d, "kl": t["kl"] / d}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    A.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    workspace_root = A.PROJECT_ROOT.parent
    table = load_material_table(workspace_root, include_double=True)
    vocab = build_element_vocab(table, min_count=args.min_element_count)
    features, conditions = featurize(table, vocab)
    splits = split_indices(len(table), seed=args.seed)

    train_ds = ArrayDataset(features, conditions, splits["train"])
    val_ds = ArrayDataset(features, conditions, splits["val"])
    test_ds = ArrayDataset(features, conditions, splits["test"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = features.shape[1]
    cdim = len(vocab)
    model = CompositionCVAE(input_dim=input_dim, cond_dim=2,
                            latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    f = loss_fn(args)

    metadata = {
        "model": args.tag,
        "feature_type": "element_fraction_plus_condition",
        "input_dim": int(input_dim), "composition_dim": int(cdim),
        "condition_dim": 2, "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim,
        "element_vocab": vocab, "n_records": int(len(table)),
        "splits": {k: int(len(v)) for k, v in splits.items()},
        "include_double_perovskites": True, "seed": args.seed,
        "a8_loss": args.loss, "a8_free_bits_per_dim": args.free_bits_per_dim,
        "a8_beta_max": args.beta_max, "a8_epochs": args.epochs,
        "a8_property_weight": args.property_weight,
    }

    hist = []
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        t = {"loss": 0.0, "comp": 0.0, "prop": 0.0, "kl": 0.0, "n": 0.0}
        for x, c in train_loader:
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x, c)
            loss, comp, prop, kl = f(recon, x, mu, logvar, cdim, epoch, args.epochs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t["loss"] += float(loss); t["comp"] += float(comp)
            t["prop"] += float(prop); t["kl"] += float(kl); t["n"] += x.shape[0]
        sched.step()
        d = max(1.0, t["n"])
        tr = {"loss": t["loss"]/d, "composition_mse": t["comp"]/d,
              "property_mse": t["prop"]/d, "kl": t["kl"]/d}
        va = evaluate(model, val_loader, cdim, epoch, args.epochs, f)
        best_val = min(best_val, va["loss"])
        hist.append({"epoch": epoch, "seconds": round(time.time()-started, 2),
                     "train_loss": tr["loss"], "train_composition_mse": tr["composition_mse"],
                     "train_property_mse": tr["property_mse"], "train_kl": tr["kl"],
                     "val_loss": va["loss"], "val_composition_mse": va["composition_mse"],
                     "val_property_mse": va["property_mse"], "val_kl": va["kl"],
                     "best_val_loss": best_val})
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{args.tag}] epoch={epoch} train={tr['loss']:.5f} val={va['loss']:.5f} "
                  f"val_kl={va['kl']:.4f} val_comp_mse={va['composition_mse']:.5f} "
                  f"({hist[-1]['seconds']}s)", flush=True)

    wall = time.time() - t0
    test = evaluate(model, test_loader, cdim, args.epochs, args.epochs, f)
    test["best_val_loss"] = best_val
    test["wall_clock_seconds"] = wall

    ckpt_path = A.CKPT_DIR / f"{args.tag}_checkpoint.pt"
    torch.save({"epoch": args.epochs, "model": model.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "best_val_loss": best_val, "metadata": metadata}, ckpt_path)
    pd.DataFrame(hist).to_csv(A.CKPT_DIR / f"{args.tag}_history.csv", index=False)
    (A.CKPT_DIR / f"{args.tag}_test_metrics.json").write_text(json.dumps(test, indent=2))
    print(f"[{args.tag}] done in {wall:.1f}s  best_val={best_val:.6f}  "
          f"test_comp_mse={test['composition_mse']:.6f} test_kl={test['kl']:.4f}")


if __name__ == "__main__":
    main()
