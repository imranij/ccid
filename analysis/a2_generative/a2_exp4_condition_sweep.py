"""Experiment 4 - does the condition vector actually steer the decoded chemistry?

A grid of (stability utility, gap utility) conditions is decoded with a SHARED,
FIXED latent draw, so every difference between cells is attributable to the
condition vector alone. Effect sizes reported:

  * mutual information I(cell ; decoded canonical formula), in bits, with a
    label-permutation null to correct for the positive bias of the plug-in estimator
  * mutual information I(cell ; decoded element), same null
  * total-variation / Jensen-Shannon distance between per-cell element-frequency
    distributions
  * mean decoded element fraction per cell (which elements move, and how far)
  * Jaccard overlap of the sets of decoded formulas between cells
  * paired formula-switch rate: for the same z, how often does changing the
    condition change the decoded formula?
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import torch

import a2_common as C

N_PER_CELL = 4000
STAB = [0.2, 0.5, 0.8, 1.0]
GAP = [0.5, 0.8, 1.0]
SEED = 123


def mutual_information(labels: np.ndarray, values: np.ndarray) -> float:
    """Plug-in MI in bits between a cell label and a categorical outcome."""
    df = pd.crosstab(labels, values).to_numpy(dtype=float)
    p = df / df.sum()
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p * np.log2(p / (px * py))
    return float(np.nansum(term))


def permutation_null(labels: np.ndarray, values: np.ndarray, n_perm: int = 20,
                     seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [mutual_information(rng.permutation(labels), values) for _ in range(n_perm)]
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return float(np.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))


def main() -> None:
    vocab = C.element_vocab()
    model, meta = C.get_model()
    comp_dim = int(meta["composition_dim"])
    torch.manual_seed(SEED)
    z = torch.randn(N_PER_CELL, int(meta["latent_dim"]))  # SHARED across all cells

    cells, frames, mean_fracs = [], [], []
    for s, g in itertools.product(STAB, GAP):
        c = np.tile([s, g], (N_PER_CELL, 1)).astype(np.float32)
        with torch.no_grad():
            dec = model.decode(z, torch.tensor(c)).numpy()[:, :comp_dim]
        cand = C.decode_batch(dec, vocab)
        cand["cell"] = f"s{s}_g{g}"
        cand["stability_utility"] = s
        cand["gap_utility"] = g
        m = C.score_frame(cand, vocab)
        m.update({"cell": f"s{s}_g{g}", "stability_utility": s, "gap_utility": g})
        cells.append(m)
        frames.append(cand)
        norm = dec / np.clip(dec.sum(axis=1, keepdims=True), 1e-12, None)
        mean_fracs.append(dict(zip(vocab, norm.mean(axis=0))))
        print(f"s={s} g={g} valid={m['validity_rate']:.3f} "
              f"uniq_all={m['unique_formula_count_all']} modal={m['modal_formula_share_all']:.3f} "
              f"top={cand['canonical_formula'].value_counts().index[0]}")

    allc = pd.concat(frames, ignore_index=True)
    allc.to_csv(C.RESULTS_DIR / "a2_condition_sweep_samples.csv.gz", index=False,
                compression="gzip")
    pd.DataFrame(cells).to_csv(C.RESULTS_DIR / "a2_condition_sweep_cells.csv", index=False)
    mf = pd.DataFrame(mean_fracs, index=[c["cell"] for c in cells])
    mf.to_csv(C.RESULTS_DIR / "a2_condition_sweep_mean_element_fractions.csv")

    labels = allc["cell"].to_numpy()
    formulas = allc["canonical_formula"].astype(str).to_numpy()

    # element-level table: one row per (sample, element)
    el_rows_label, el_rows_el = [], []
    for lab, elems in zip(labels, allc["elements"].astype(str).to_numpy()):
        for e in elems.split():
            if e != "O":
                el_rows_label.append(lab)
                el_rows_el.append(e)
    el_lab = np.array(el_rows_label)
    el_el = np.array(el_rows_el)

    def entropy(v):
        p = pd.Series(v).value_counts(normalize=True).to_numpy()
        return float(-(p * np.log2(p)).sum())

    mi_f = mutual_information(labels, formulas)
    mi_f_null, mi_f_null_sd = permutation_null(labels, formulas)
    mi_e = mutual_information(el_lab, el_el)
    mi_e_null, mi_e_null_sd = permutation_null(el_lab, el_el)

    # per-cell element frequency distributions and pairwise JS distance
    freq = pd.crosstab(el_lab, el_el, normalize="index")
    cell_names = list(freq.index)
    js = pd.DataFrame(
        [[js_distance(freq.loc[a].to_numpy(), freq.loc[b].to_numpy()) for b in cell_names]
         for a in cell_names], index=cell_names, columns=cell_names)
    js.to_csv(C.RESULTS_DIR / "a2_condition_sweep_js_matrix.csv")
    freq.to_csv(C.RESULTS_DIR / "a2_condition_sweep_element_frequency.csv")

    # Jaccard overlap of decoded formula sets
    sets = {c: set(allc.loc[allc["cell"] == c, "canonical_formula"]) for c in cell_names}
    jac = pd.DataFrame(
        [[len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b])) for b in cell_names]
         for a in cell_names], index=cell_names, columns=cell_names)
    jac.to_csv(C.RESULTS_DIR / "a2_condition_sweep_jaccard.csv")

    # paired switch rate: same z, different condition
    pivot = allc.pivot_table(index="sample_id", columns="cell",
                             values="canonical_formula", aggfunc="first")
    switch = {f"{a}|{b}": float((pivot[a] != pivot[b]).mean())
              for a, b in itertools.combinations(cell_names, 2)}

    # marginal effect of each condition channel with the other held at its mid value
    out = {
        "n_per_cell": N_PER_CELL,
        "grid": {"stability_utility": STAB, "gap_utility": GAP},
        "shared_latent_draw": True,
        "mutual_information_bits": {
            "cell_vs_formula": mi_f,
            "cell_vs_formula_permutation_null_mean": mi_f_null,
            "cell_vs_formula_permutation_null_sd": mi_f_null_sd,
            "cell_vs_element": mi_e,
            "cell_vs_element_permutation_null_mean": mi_e_null,
            "cell_vs_element_permutation_null_sd": mi_e_null_sd,
            "max_possible_bits_12_cells": float(np.log2(len(cell_names))),
            "H_formula_bits": entropy(formulas),
            "H_element_bits": entropy(el_el),
            "normalised_MI_formula": mi_f / max(1e-12, entropy(formulas)),
            "normalised_MI_element": mi_e / max(1e-12, entropy(el_el)),
        },
        "js_distance_between_cells": {
            "mean_offdiag": float(js.to_numpy()[~np.eye(len(js), dtype=bool)].mean()),
            "max": float(js.to_numpy().max()),
        },
        "jaccard_formula_sets": {
            "mean_offdiag": float(jac.to_numpy()[~np.eye(len(jac), dtype=bool)].mean()),
            "min": float(jac.to_numpy()[~np.eye(len(jac), dtype=bool)].min()),
        },
        "paired_formula_switch_rate": {
            "mean": float(np.mean(list(switch.values()))),
            "min": float(np.min(list(switch.values()))),
            "max": float(np.max(list(switch.values()))),
            "per_pair": switch,
        },
        "modal_formula_per_cell": {
            c: allc.loc[allc["cell"] == c, "canonical_formula"].value_counts().index[0]
            for c in cell_names
        },
        "per_cell": cells,
        "mean_element_fraction_range_across_cells": {
            e: float(mf[e].max() - mf[e].min())
            for e in mf.columns[np.argsort(-(mf.max() - mf.min()).to_numpy())[:15]]
        },
    }
    (C.RESULTS_DIR / "a2_condition_sweep_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_cell",)}, indent=2)[:4000])


if __name__ == "__main__":
    main()
