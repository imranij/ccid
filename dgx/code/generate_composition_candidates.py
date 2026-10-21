
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from composition_cvae_common import (
    CompositionCVAE,
    decode_composition_vector,
    load_material_table,
    parse_formula,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="composition_cvae")
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--target-stability", type=float, default=0.9)
    p.add_argument("--target-gap", type=float, default=0.8)
    p.add_argument("--condition-jitter", type=float, default=0.08)
    p.add_argument("--top-cations", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()

def composition_vector(formula: str, element_vocab: list[str]) -> np.ndarray:
    counts = parse_formula(formula)
    total = sum(counts.values())
    arr = np.zeros(len(element_vocab), dtype=np.float32)
    if total <= 0:
        return arr
    for idx, element in enumerate(element_vocab):
        arr[idx] = float(counts.get(element, 0.0)) / float(total)
    return arr

def mean_pairwise_distance(vectors: np.ndarray, max_pairs: int = 5000) -> float:
    if len(vectors) < 2:
        return 0.0
    pairs = list(combinations(range(len(vectors)), 2))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(5)
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=max_pairs, replace=False)]
    distances = [float(np.linalg.norm(vectors[i] - vectors[j])) for i, j in pairs]
    return float(np.mean(distances)) if distances else 0.0

def load_model(project_root: Path, model_name: str) -> tuple[CompositionCVAE, dict[str, object]]:
    ckpt_path = project_root / "outputs_dgx" / f"{model_name}_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metadata = ckpt["metadata"]
    model = CompositionCVAE(
        input_dim=int(metadata["input_dim"]),
        cond_dim=int(metadata["condition_dim"]),
        latent_dim=int(metadata["latent_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
    )
    state = {key.replace("module.", ""): value for key, value in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, metadata

def write_latex_table(path: Path, top: pd.DataFrame, metrics: dict[str, object]) -> None:
    def esc(value: object) -> str:
        return str(value).replace("_", r"\_")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Composition-CVAE candidate proposals after oxide, charge-balance, novelty, and risk filters. These are computational proposals requiring database, DFT, uMLIP, or experimental validation.}",
        r"\label{tab:ceramic_composition_cvae_candidates}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Formula & Status & Target stability & Target gap & Score & Nearest known distance \\",
        r"\midrule",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"{esc(row['formula'])} & {esc(row['status'])} & "
            f"{row['target_stability_utility']:.3f} & {row['target_gap_utility']:.3f} & "
            f"{row['candidate_score']:.3f} & {row['nearest_known_distance']:.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\vspace{0.3em}\footnotesize{"
                f"Validity: {metrics['validity_rate']:.1%}; ".replace("%", r"\%")
                + f"uniqueness among valid: {metrics['uniqueness_rate']:.1%}; ".replace("%", r"\%")
                + f"novelty among unique valid: {metrics['novelty_rate']:.1%}; ".replace("%", r"\%")
                + f"mean pairwise composition distance: {metrics['mean_pairwise_composition_distance']:.4f}."
                + r"}"
            ),
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    out_dir = project_root / "outputs_dgx"
    manuscript_dir = workspace_root / "manuscript"

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    model, metadata = load_model(project_root, args.model)
    element_vocab = list(metadata["element_vocab"])
    known_table = load_material_table(workspace_root, include_double=bool(metadata.get("include_double_perovskites", True)))
    known_formulas = set(known_table["canonical_formula"])
    known_vectors = np.vstack([composition_vector(formula, element_vocab) for formula in known_table["formula"]])

    n = args.n_samples
    z = torch.randn(n, int(metadata["latent_dim"]))
    conditions = np.column_stack(
        [
            np.clip(rng.normal(args.target_stability, args.condition_jitter, n), 0.0, 1.0),
            np.clip(rng.normal(args.target_gap, args.condition_jitter, n), 0.0, 1.0),
        ]
    ).astype(np.float32)

    with torch.no_grad():
        decoded = model.decode(z, torch.tensor(conditions, dtype=torch.float32)).numpy()

    composition_dim = int(metadata["composition_dim"])
    rows = []
    for idx in range(n):
        decoded_comp = decoded[idx, :composition_dim]
        result = decode_composition_vector(decoded_comp, element_vocab, top_cations=args.top_cations)
        formula = result.get("formula", "")
        canonical = result.get("canonical_formula", "")
        vector = composition_vector(str(formula), element_vocab)
        if vector.sum() > 0 and len(known_vectors):
            nearest_distance = float(np.linalg.norm(known_vectors - vector, axis=1).min())
        else:
            nearest_distance = float("nan")
        is_valid = bool(result.get("valid", False))
        is_novel = bool(canonical and canonical not in known_formulas)
        score = 0.5 * float(conditions[idx, 0]) + 0.5 * float(conditions[idx, 1])
        rows.append(
            {
                "sample_id": idx,
                "formula": formula,
                "canonical_formula": canonical,
                "status": result.get("reason", ""),
                "valid": is_valid,
                "novel": is_novel,
                "charge_balanced": bool(result.get("charge_balanced", False)),
                "target_stability_utility": float(conditions[idx, 0]),
                "target_gap_utility": float(conditions[idx, 1]),
                "candidate_score": score,
                "nearest_known_distance": nearest_distance,
                "elements": result.get("elements", ""),
                "high_risk_elements": result.get("high_risk_elements", ""),
                "process_risk_elements": result.get("process_risk_elements", ""),
                "unusual_elements": result.get("unusual_elements", ""),
            }
        )

    candidates = pd.DataFrame(rows)
    valid = candidates[candidates["valid"]].copy()
    unique_valid = valid.drop_duplicates("canonical_formula").copy()
    novel_unique = unique_valid[unique_valid["novel"]].copy()

    vectors = np.vstack([composition_vector(formula, element_vocab) for formula in novel_unique["formula"]]) if len(novel_unique) else np.zeros((0, len(element_vocab)))
    metrics = {
        "total_samples": int(n),
        "valid_count": int(len(valid)),
        "unique_valid_count": int(len(unique_valid)),
        "novel_unique_count": int(len(novel_unique)),
        "validity_rate": float(len(valid) / max(1, n)),
        "uniqueness_rate": float(len(unique_valid) / max(1, len(valid))),
        "novelty_rate": float(len(novel_unique) / max(1, len(unique_valid))),
        "mean_pairwise_composition_distance": mean_pairwise_distance(vectors),
        "target_stability": args.target_stability,
        "target_gap": args.target_gap,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = out_dir / f"{args.model}_generated_candidates.csv"
    metrics_path = out_dir / f"{args.model}_generated_metrics.json"
    candidates.to_csv(candidates_path, index=False, quoting=csv.QUOTE_MINIMAL)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    top = novel_unique.sort_values(["candidate_score", "nearest_known_distance"], ascending=[False, False]).head(12)
    write_latex_table(manuscript_dir / f"generated_{args.model}_candidates.tex", top, metrics)

    print(f"Wrote {candidates_path}")
    print(f"Wrote {metrics_path}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()