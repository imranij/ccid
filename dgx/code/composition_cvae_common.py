
from __future__ import annotations

import gzip
import json
import math
import re
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

FORMULA_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")
HIGH_RISK_ELEMENTS = {"Pb", "Cd", "Hg", "As", "Tl", "Be", "Po", "Ra", "Th", "U", "Pu"}
PROCESS_RISK_ELEMENTS = {"F", "Cl", "Br", "I", "Se", "Te", "S"}
NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}
COMMON_CERAMIC_ELEMENTS = {
    "O",
    "Ba",
    "Sr",
    "Ca",
    "Na",
    "K",
    "Li",
    "Mg",
    "Al",
    "Si",
    "Ti",
    "Zr",
    "Hf",
    "Nb",
    "Ta",
    "Sc",
    "Y",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Bi",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "In",
    "Sn",
    "W",
    "Mo",
}

OXIDATION_STATES = {
    "Li": [1],
    "Na": [1],
    "K": [1],
    "Rb": [1],
    "Cs": [1],
    "Be": [2],
    "Mg": [2],
    "Ca": [2],
    "Sr": [2],
    "Ba": [2],
    "Sc": [3],
    "Y": [3],
    "La": [3],
    "Ce": [3, 4],
    "Pr": [3, 4],
    "Nd": [3],
    "Sm": [3],
    "Gd": [3],
    "Tb": [3, 4],
    "Dy": [3],
    "Ho": [3],
    "Er": [3],
    "Tm": [3],
    "Yb": [2, 3],
    "Lu": [3],
    "Ti": [3, 4],
    "Zr": [4],
    "Hf": [4],
    "V": [3, 4, 5],
    "Nb": [5],
    "Ta": [5],
    "Cr": [3, 6],
    "Mo": [4, 6],
    "W": [6],
    "Mn": [2, 3, 4],
    "Fe": [2, 3],
    "Co": [2, 3],
    "Ni": [2, 3],
    "Cu": [1, 2],
    "Zn": [2],
    "Al": [3],
    "Ga": [3],
    "In": [3],
    "Si": [4],
    "Ge": [4],
    "Sn": [2, 4],
    "Bi": [3, 5],
    "O": [-2],
}

def parse_formula(formula: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for element, amount in FORMULA_RE.findall(str(formula)):
        value = float(amount) if amount else 1.0
        counts[element] = counts.get(element, 0.0) + value
    return counts

def canonical_formula(counts: dict[str, float]) -> str:
    parts = []
    for element in sorted(counts):
        value = counts[element]
        if abs(value - round(value)) < 1e-8:
            value = int(round(value))
        if value == 1:
            parts.append(element)
        else:
            parts.append(f"{element}{value:g}")
    return "".join(parts)

def load_json_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict) and {"data", "columns", "index"} <= set(raw):
        return pd.DataFrame(raw["data"], columns=raw["columns"], index=raw["index"])
    return pd.DataFrame(raw)

def minmax(values: pd.Series | np.ndarray, invert: bool = False) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if invert:
        arr = -arr
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.full(len(arr), 0.5, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)

def load_material_table(workspace_root: Path, include_double: bool = True) -> pd.DataFrame:
    shared = workspace_root / "shared_datasets" / "matminer"
    rows: list[pd.DataFrame] = []

    castelli = shared / "castelli_perovskites.json.gz"
    if castelli.exists():
        df = load_json_gz(castelli)
        rows.append(
            pd.DataFrame(
                {
                    "formula": df["formula"].astype(str),
                    "source_dataset": "castelli_perovskites",
                    "stability": pd.to_numeric(df.get("e_form", 0.0), errors="coerce").fillna(0.0),
                    "gap": pd.to_numeric(df.get("gap gllbsc", 0.0), errors="coerce").fillna(0.0),
                    "stability_known": 1,
                }
            )
        )

    wolverton = shared / "wolverton_oxides.json.gz"
    if wolverton.exists():
        df = load_json_gz(wolverton)
        rows.append(
            pd.DataFrame(
                {
                    "formula": df["formula"].astype(str),
                    "source_dataset": "wolverton_oxides",
                    "stability": pd.to_numeric(df.get("e_hull", 0.0), errors="coerce").fillna(0.0),
                    "gap": pd.to_numeric(df.get("gap pbe", 0.0), errors="coerce").fillna(0.0),
                    "stability_known": 1,
                }
            )
        )

    double = shared / "double_perovskites_gap.json.gz"
    if include_double and double.exists():
        df = load_json_gz(double)
        rows.append(
            pd.DataFrame(
                {
                    "formula": df["formula"].astype(str),
                    "source_dataset": "double_perovskites_gap",
                    "stability": 0.0,
                    "gap": pd.to_numeric(df.get("gap gllbsc", 0.0), errors="coerce").fillna(0.0),
                    "stability_known": 0,
                }
            )
        )

    if not rows:
        raise FileNotFoundError(f"No matminer datasets found under {shared}")

    table = pd.concat(rows, ignore_index=True)
    table["counts"] = table["formula"].map(parse_formula)
    table = table[table["counts"].map(bool)].reset_index(drop=True)

    known = table["stability_known"].astype(bool).to_numpy()
    stability_utility = np.full(len(table), 0.5, dtype=np.float32)
    if known.any():
        stability_utility[known] = minmax(table.loc[known, "stability"], invert=True)
    table["stability_utility"] = stability_utility
    table["gap_utility"] = minmax(table["gap"])
    table["candidate_score"] = 0.5 * table["stability_utility"] + 0.5 * table["gap_utility"]
    table["canonical_formula"] = table["counts"].map(canonical_formula)
    return table

def build_element_vocab(table: pd.DataFrame, min_count: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for comp in table["counts"]:
        for element in comp:
            counts[element] = counts.get(element, 0) + 1
    vocab = sorted([element for element, count in counts.items() if count >= min_count])
    if "O" not in vocab:
        vocab.append("O")
    return sorted(vocab)

def featurize(table: pd.DataFrame, element_vocab: list[str]) -> tuple[np.ndarray, np.ndarray]:
    element_to_idx = {element: i for i, element in enumerate(element_vocab)}
    composition = np.zeros((len(table), len(element_vocab)), dtype=np.float32)
    for row_idx, counts in enumerate(table["counts"]):
        total = sum(counts.values())
        if total <= 0:
            continue
        for element, amount in counts.items():
            if element in element_to_idx:
                composition[row_idx, element_to_idx[element]] = float(amount) / float(total)
    conditions = table[["stability_utility", "gap_utility"]].to_numpy(dtype=np.float32)
    features = np.hstack([composition, conditions]).astype(np.float32)
    return features, conditions

def split_indices(n: int, seed: int = 42, train_frac: float = 0.8, val_frac: float = 0.1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, conditions: np.ndarray, indices: Iterable[int]):
        self.features = torch.tensor(features[list(indices)], dtype=torch.float32)
        self.conditions = torch.tensor(conditions[list(indices)], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.conditions[idx]

class CompositionCVAE(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int = 2, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.enc1 = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.dec1 = nn.Linear(latent_dim + cond_dim, hidden_dim // 2)
        self.dec2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.enc1(torch.cat([x, c], dim=-1)))
        h = F.gelu(self.enc2(h))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.dec1(torch.cat([z, c], dim=-1)))
        h = F.gelu(self.dec2(h))
        return torch.sigmoid(self.out(h))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

def cvae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    composition_dim: int,
    epoch: int,
    max_epochs: int,
    property_weight: float = 2.0,
    beta_max: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    comp_loss = F.mse_loss(recon_x[:, :composition_dim], x[:, :composition_dim], reduction="sum")
    prop_loss = F.mse_loss(recon_x[:, composition_dim:], x[:, composition_dim:], reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    anneal_epochs = max(1, min(50, max_epochs // 2))
    beta = min(beta_max, beta_max * (epoch / anneal_epochs))
    free_bits = 0.5 * mu.shape[1]
    loss = comp_loss + property_weight * prop_loss + beta * torch.clamp(kl, min=free_bits)
    return loss, comp_loss, prop_loss, kl

def formula_from_counts(counts: dict[str, int]) -> str:
    ordered = [element for element in counts if element != "O"] + (["O"] if "O" in counts else [])
    parts = []
    for element in ordered:
        value = counts[element]
        if value == 1:
            parts.append(element)
        else:
            parts.append(f"{element}{value}")
    return "".join(parts)

def charge_balance_possible(counts: dict[str, int]) -> bool:
    states = []
    for element in counts:
        if element not in OXIDATION_STATES:
            return False
        states.append(OXIDATION_STATES[element])
    for combo in product(*states):
        charge = sum(counts[element] * state for element, state in zip(counts, combo))
        if charge == 0:
            return True
    return False

def decode_composition_vector(
    vector: np.ndarray,
    element_vocab: list[str],
    top_cations: int = 3,
    max_cation_count: int = 4,
    max_oxygen_count: int = 12,
) -> dict[str, object]:
    arr = np.asarray(vector, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    if arr.sum() <= 0:
        return {"valid": False, "reason": "zero composition vector"}
    arr = arr / arr.sum()

    if "O" not in element_vocab:
        return {"valid": False, "reason": "oxygen not in vocabulary"}

    o_idx = element_vocab.index("O")
    cation_candidates = [
        (element, arr[idx])
        for idx, element in enumerate(element_vocab)
        if element != "O" and element not in NOBLE_GASES
    ]
    cation_candidates.sort(key=lambda item: item[1], reverse=True)
    cations = [element for element, frac in cation_candidates[:top_cations] if frac > 0.01]
    if not cations:
        return {"valid": False, "reason": "no cation selected"}

    cation_fracs = np.array([arr[element_vocab.index(element)] for element in cations])
    cation_fracs = cation_fracs / max(cation_fracs.sum(), 1e-12)
    raw_counts = np.maximum(1, np.rint(cation_fracs * max_cation_count).astype(int))
    gcd_value = int(raw_counts[0])
    for value in raw_counts[1:]:
        gcd_value = math.gcd(gcd_value, int(value))
    raw_counts = raw_counts // max(gcd_value, 1)

    cation_counts = {element: int(count) for element, count in zip(cations, raw_counts)}
    best_counts = None
    best_distance = float("inf")
    decoded_o_frac = float(arr[o_idx])
    for oxygen_count in range(1, max_oxygen_count + 1):
        counts = dict(cation_counts)
        counts["O"] = oxygen_count
        total = sum(counts.values())
        oxygen_frac = oxygen_count / total
        if charge_balance_possible(counts):
            distance = abs(oxygen_frac - decoded_o_frac)
            if distance < best_distance:
                best_distance = distance
                best_counts = counts

    if best_counts is None:
        counts = dict(cation_counts)
        counts["O"] = max(1, int(round(decoded_o_frac * (sum(cation_counts.values()) + 3))))
        return {
            "valid": False,
            "formula": formula_from_counts(counts),
            "canonical_formula": canonical_formula({k: float(v) for k, v in counts.items()}),
            "reason": "no charge-balanced oxygen count found",
            "elements": " ".join(counts),
            "oxygen_fraction": decoded_o_frac,
        }

    elements = set(best_counts)
    high_risk = sorted(elements & HIGH_RISK_ELEMENTS)
    process_risk = sorted(elements & PROCESS_RISK_ELEMENTS)
    unusual = sorted(elements - COMMON_CERAMIC_ELEMENTS)
    return {
        "valid": not high_risk and not unusual,
        "formula": formula_from_counts(best_counts),
        "canonical_formula": canonical_formula({k: float(v) for k, v in best_counts.items()}),
        "reason": "charge-balanced oxide candidate" if not high_risk and not unusual else "risk elements present",
        "elements": " ".join(best_counts),
        "high_risk_elements": " ".join(high_risk),
        "process_risk_elements": " ".join(process_risk),
        "unusual_elements": " ".join(unusual),
        "oxygen_fraction": decoded_o_frac,
        "charge_balanced": True,
    }