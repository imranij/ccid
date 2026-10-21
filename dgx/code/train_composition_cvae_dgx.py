
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from composition_cvae_common import (
    ArrayDataset,
    CompositionCVAE,
    build_element_vocab,
    cvae_loss,
    featurize,
    load_material_table,
    split_indices,
)

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def is_master() -> bool:
    return (not is_dist()) or dist.get_rank() == 0

def setup_distributed(local_rank: int) -> torch.device:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="composition_cvae")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--min-element-count", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--no-resume", "--no_resume", dest="no_resume", action="store_true")
    p.add_argument("--no-double", dest="include_double", action="store_false")
    p.set_defaults(include_double=True)
    return p.parse_args()

def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    composition_dim: int,
    epoch: int,
    max_epochs: int,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "comp": 0.0, "prop": 0.0, "kl": 0.0, "count": 0.0}
    with torch.no_grad():
        for x, c in loader:
            x = x.to(device)
            c = c.to(device)
            recon, mu, logvar = model(x, c)
            loss, comp, prop, kl = cvae_loss(recon, x, mu, logvar, composition_dim, epoch, max_epochs)
            totals["loss"] += float(loss.item())
            totals["comp"] += float(comp.item())
            totals["prop"] += float(prop.item())
            totals["kl"] += float(kl.item())
            totals["count"] += float(x.shape[0])
    denom = max(1.0, totals["count"])
    return {
        "loss": totals["loss"] / denom,
        "composition_mse": totals["comp"] / denom,
        "property_mse": totals["prop"] / denom,
        "kl": totals["kl"] / denom,
    }

def append_history(path: Path, row: dict[str, object]) -> None:
    fields = [
        "epoch",
        "seconds",
        "train_loss",
        "train_composition_mse",
        "train_property_mse",
        "train_kl",
        "val_loss",
        "val_composition_mse",
        "val_property_mse",
        "val_kl",
        "best_val_loss",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row[field] for field in fields})

def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    device = setup_distributed(local_rank)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    out_dir = ensure_dir(project_root / "outputs_dgx")
    ckpt_path = out_dir / f"{args.model}_checkpoint.pt"
    history_path = out_dir / f"{args.model}_history.csv"
    metadata_path = out_dir / f"{args.model}_metadata.json"
    test_metrics_path = out_dir / f"{args.model}_test_metrics.json"

    table = load_material_table(workspace_root, include_double=args.include_double)
    element_vocab = build_element_vocab(table, min_count=args.min_element_count)
    features, conditions = featurize(table, element_vocab)
    splits = split_indices(len(table), seed=args.seed)

    train_ds = ArrayDataset(features, conditions, splits["train"])
    val_ds = ArrayDataset(features, conditions, splits["val"])
    test_ds = ArrayDataset(features, conditions, splits["test"])

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist() else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")

    input_dim = features.shape[1]
    composition_dim = len(element_vocab)
    model = CompositionCVAE(
        input_dim=input_dim,
        cond_dim=2,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    if is_dist():
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    core = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    start_epoch = 1
    best_val_loss = float("inf")
    if args.no_resume and is_master():
        for stale_path in (history_path, test_metrics_path):
            if stale_path.exists():
                stale_path.unlink()
    if not args.no_resume and ckpt_path.exists():
        loc = f"cuda:{local_rank}" if device.type == "cuda" else "cpu"
        ckpt = torch.load(ckpt_path, map_location=loc, weights_only=False)
        core.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt["best_val_loss"])
        if is_master():
            print(f"[composition-cvae] resumed from epoch {start_epoch}")

    if is_master():
        metadata = {
            "model": args.model,
            "feature_type": "element_fraction_plus_condition",
            "input_dim": int(input_dim),
            "composition_dim": int(composition_dim),
            "condition_dim": 2,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "element_vocab": element_vocab,
            "n_records": int(len(table)),
            "splits": {key: int(len(value)) for key, value in splits.items()},
            "include_double_perovskites": bool(args.include_double),
            "seed": args.seed,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        table[["formula", "canonical_formula", "source_dataset", "stability", "gap", "stability_known", "stability_utility", "gap_utility", "candidate_score"]].to_csv(
            out_dir / f"{args.model}_training_records.csv", index=False
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        started = time.time()
        totals = {"loss": 0.0, "comp": 0.0, "prop": 0.0, "kl": 0.0, "count": 0.0}

        for x, c in train_loader:
            x = x.to(device)
            c = c.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x, c)
            loss, comp, prop, kl = cvae_loss(recon, x, mu, logvar, composition_dim, epoch, args.epochs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            totals["loss"] += float(loss.item())
            totals["comp"] += float(comp.item())
            totals["prop"] += float(prop.item())
            totals["kl"] += float(kl.item())
            totals["count"] += float(x.shape[0])

        scheduler.step()

        if is_master():
            train_count = max(1.0, totals["count"])
            train_metrics = {
                "loss": totals["loss"] / train_count,
                "composition_mse": totals["comp"] / train_count,
                "property_mse": totals["prop"] / train_count,
                "kl": totals["kl"] / train_count,
            }
            val_metrics = evaluate(core, val_loader, device, composition_dim, epoch, args.epochs)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
            row = {
                "epoch": epoch,
                "seconds": round(time.time() - started, 2),
                "train_loss": train_metrics["loss"],
                "train_composition_mse": train_metrics["composition_mse"],
                "train_property_mse": train_metrics["property_mse"],
                "train_kl": train_metrics["kl"],
                "val_loss": val_metrics["loss"],
                "val_composition_mse": val_metrics["composition_mse"],
                "val_property_mse": val_metrics["property_mse"],
                "val_kl": val_metrics["kl"],
                "best_val_loss": best_val_loss,
            }
            append_history(history_path, row)
            torch.save(
                {
                    "epoch": epoch,
                    "model": core.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
                },
                ckpt_path,
            )
            print(
                "[composition-cvae] "
                f"epoch={epoch} train={train_metrics['loss']:.5f} "
                f"val={val_metrics['loss']:.5f} best_val={best_val_loss:.5f}"
            )

        if is_dist():
            dist.barrier()

    if is_master():
        test_metrics = evaluate(core, test_loader, device, composition_dim, args.epochs, args.epochs)
        test_metrics["best_val_loss"] = best_val_loss
        test_metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        print(f"[composition-cvae] complete. best_val_loss={best_val_loss:.6f}")
        print(f"[composition-cvae] test_metrics={json.dumps(test_metrics)}")

    if is_dist():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()