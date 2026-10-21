"""Step 4: the rediscovery test.

Hold out known good wide-gap dielectric materials (Stage-1 Pareto set: SrHfO3,
BaHfO3, CaHfO3, BaZrO3, LaAlO3, YAlO3) and ask: does a COMPOSITION-ONLY approach,
searching for compositions whose SURROGATE-PREDICTED properties match the held-out
material's TRUE target properties, recover the material itself or a close chemical
neighbour?

Design (documented explicitly because there is no single canonical way to do this):
  1. All six targets are, in fact, present in composition_cvae_training_records.csv
     (confirmed by exact canonical-formula match; each formula appears under BOTH
     Castelli and Wolverton, and under both cation writing orders, i.e. 3-4 rows
     each) — they are NOT held out of the CVAE's own training set. We therefore
     retrain HOLDOUT versions of the stability and band-gap surrogates with EVERY
     row matching any of the six canonical formulas removed, so this is a genuine
     out-of-sample test of the surrogate, independent of the CVAE.
  2. We do not have a composition-only generative decoder available in this agent's
     scope (that is the CVAE itself, Stage 2, owned by a different part of the
     pipeline) — so "a composition-only approach conditioned on property targets"
     is implemented here as SURROGATE-GUIDED NEAREST-PROPERTY SEARCH: we build a
     large composition-only search universe (every unique formula in the training
     pool NOT touching the six held-out compounds, unioned with a combinatorial
     ABO3 generator over the 41 whitelisted "common ceramic" cations, charge-balance
     filtered), score every candidate in that universe with the HOLDOUT surrogate,
     and rank candidates by distance-in-predicted-property-space to the held-out
     material's TRUE (data-derived) target stability/gap.
  3. We report, for each held-out material: whether the exact formula is reachable
     in the search universe at all, its rank there by predicted-property distance,
     and the identity/chemistry of the top-5 nearest candidates (same A-site or
     B-site family = "close chemical neighbour", by our reading of the task).
  4. We report the result STRAIGHT, including if it is negative — a composition-
     only surrogate is not the same thing as a trained generative decoder, and if
     it fails to prioritise these known-good compounds near the top, that is
     itself an important, honest finding for the authors.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import itertools
import numpy as np
import pandas as pd

from a5_common import (
    DGX_OUT, RESULTS_DIR, magpie_features, BootstrapEnsemble, rf_factory_default,
    save_json, canonical_formula as canon, parse_formula,
    COMMON_CERAMIC_ELEMENTS, OXIDATION_STATES,
)
from composition_cvae_common import charge_balance_possible
import functools

rf_factory = functools.partial(rf_factory_default, n_estimators=80, max_depth=20, min_samples_leaf=2)

RNG_SEED = 42
N_ENSEMBLE = 6

TARGETS = ["SrHfO3", "BaHfO3", "CaHfO3", "BaZrO3", "LaAlO3", "YAlO3"]


def target_canonical(formula: str) -> str:
    return canon(parse_formula(formula))


def build_synthetic_abo3_universe() -> list[str]:
    """All charge-balanced ABO3 combinations over the 41 COMMON_CERAMIC_ELEMENTS
    cations that have a known oxidation state (reuses the project's own
    charge_balance_possible / OXIDATION_STATES tables)."""
    cations = sorted(e for e in COMMON_CERAMIC_ELEMENTS if e != "O" and e in OXIDATION_STATES)
    formulas = set()
    for a, b in itertools.product(cations, cations):
        if a == b:
            continue
        counts = {a: 1, b: 1, "O": 3}
        if charge_balance_possible(counts):
            formulas.add(canon({a: 1.0, b: 1.0, "O": 3.0}))
    return sorted(formulas)


def main():
    df = pd.read_csv(DGX_OUT / "composition_cvae_training_records.csv")
    target_canons = {t: target_canonical(t) for t in TARGETS}
    print("Target canonical formulas:", target_canons)

    holdout_mask = df["canonical_formula"].isin(set(target_canons.values()))
    print(f"{holdout_mask.sum()} rows across the training table match one of the six held-out compounds "
          f"(removed from ALL surrogate training data for this test):")
    print(df[holdout_mask][["formula", "canonical_formula", "source_dataset", "stability", "gap"]].to_string())

    df_train = df[~holdout_mask].reset_index(drop=True)

    # ---- featurize the (reduced) training table ----
    cached = np.load(_CCID_ROOT + "/analysis/a5_surrogate/a5_training_magpie_features.npz", allow_pickle=True)
    X_all_cached = cached["X"]
    assert cached["ok"].all()
    X_train = X_all_cached[~holdout_mask.to_numpy()]

    # ---- retrain HOLDOUT stability models (per source) ----
    known = df_train["stability_known"].astype(bool).to_numpy()
    holdout_stab_models = {}
    for source in ["castelli_perovskites", "wolverton_oxides"]:
        mask = (df_train["source_dataset"] == source).to_numpy() & known
        model = BootstrapEnsemble(rf_factory, n_members=N_ENSEMBLE, random_state=RNG_SEED)
        model.fit(X_train[mask], df_train.loc[mask, "stability"].to_numpy())
        holdout_stab_models[source] = model
    print("Retrained holdout stability models (Castelli, Wolverton).")

    # ---- retrain HOLDOUT gap model ----
    src_dummies = pd.get_dummies(df_train["source_dataset"], prefix="src")
    for col in ["src_castelli_perovskites", "src_double_perovskites_gap", "src_wolverton_oxides"]:
        if col not in src_dummies:
            src_dummies[col] = 0.0
    src_dummies = src_dummies[["src_castelli_perovskites", "src_double_perovskites_gap", "src_wolverton_oxides"]]
    X_train_gap = np.hstack([X_train, src_dummies.to_numpy(dtype=np.float64)])
    holdout_gap_model = BootstrapEnsemble(rf_factory, n_members=N_ENSEMBLE, random_state=RNG_SEED)
    holdout_gap_model.fit(X_train_gap, df_train["gap"].to_numpy())
    print("Retrained holdout gap model.")

    # ---- build search universe ----
    # The raw Castelli/Wolverton pool contains many formulas with elements far
    # outside sane ceramic-dielectric chemistry (actinides like Pa/U/Np, risky
    # elements, etc. -- this is that same pool the paper's Stage 1 draws from
    # without an element-risk filter, the audit brief). We therefore build
    # TWO universes: the full (unrestricted) pool+synthetic union, and a
    # "practical" whitelist-only union restricted to COMMON_CERAMIC_ELEMENTS
    # (the same 41-element whitelist Stage 2's own post-filter uses), so the
    # "close chemical neighbour" reading is not contaminated by radioactive or
    # otherwise impractical near-matches.
    def elements_of(canonical: str) -> set[str]:
        return set(parse_formula(canonical).keys())

    pool_formulas_all = set(df_train["canonical_formula"].unique().tolist())
    pool_formulas_practical = {
        f for f in pool_formulas_all if elements_of(f) <= COMMON_CERAMIC_ELEMENTS
    }
    synth = set(build_synthetic_abo3_universe())  # already whitelist-only by construction
    universe_all = sorted(pool_formulas_all | synth)
    universe_practical = sorted(pool_formulas_practical | synth)
    print(f"Full search universe: {len(pool_formulas_all)} raw training-pool formulas (any element) + "
          f"{len(synth)} synthetic ABO3 combinations = {len(universe_all)} unique compositions.")
    print(f"Practical (whitelist-only) search universe: {len(pool_formulas_practical)} whitelist-only pool "
          f"formulas + {len(synth)} synthetic = {len(universe_practical)} unique compositions.")
    universe = universe_practical  # use the practical universe as the primary reported result

    X_uni, ok = magpie_features(universe)
    universe = [u for u, o in zip(universe, ok) if o]
    X_uni = X_uni[ok]
    print(f"Featurized {len(universe)} universe compositions.")

    stab_mean = {}
    for source, model in holdout_stab_models.items():
        mean, std = model.predict(X_uni)
        stab_mean[source] = mean

    gap_preds = []
    for i in range(3):
        dummy = np.zeros((len(X_uni), 3))
        dummy[:, i] = 1.0
        mean, _ = holdout_gap_model.predict(np.hstack([X_uni, dummy]))
        gap_preds.append(mean)
    gap_mean = np.mean(gap_preds, axis=0)

    universe_df = pd.DataFrame({
        "canonical_formula": universe,
        "pred_stability_castelli": stab_mean["castelli_perovskites"],
        "pred_stability_wolverton": stab_mean["wolverton_oxides"],
        "pred_gap": gap_mean,
    })

    # ---- per-target search ----
    report_rows = []
    detail_rows = []
    for target_name, tcanon in target_canons.items():
        rows = df[df["canonical_formula"] == tcanon]
        # use the most-stable (min stability) entry per source as the "true" target,
        # documenting the ambiguity that composition alone maps to >1 polymorph.
        true_by_source = {}
        for source in ["castelli_perovskites", "wolverton_oxides"]:
            src_rows = rows[rows["source_dataset"] == source]
            if len(src_rows):
                best = src_rows.loc[src_rows["stability"].idxmin()]
                true_by_source[source] = {"stability": float(best["stability"]), "gap": float(best["gap"])}
        print(f"\n=== {target_name} (canonical {tcanon}) ===")
        print("True (most-stable-entry) properties by source:", true_by_source)

        reachable = tcanon in set(universe)
        print(f"Reachable via synthetic ABO3 generator or training pool: {reachable}")

        for source, truth in true_by_source.items():
            col = "pred_stability_castelli" if source == "castelli_perovskites" else "pred_stability_wolverton"
            # normalize stability & gap distance by the training-pool spread so
            # the two axes are comparable
            stab_scale = df_train.loc[known, "stability"].std()
            gap_scale = df_train["gap"].std()
            dist = (
                ((universe_df[col] - truth["stability"]) / stab_scale) ** 2
                + ((universe_df["pred_gap"] - truth["gap"]) / gap_scale) ** 2
            ) ** 0.5
            ranked = universe_df.assign(distance=dist).sort_values("distance").reset_index(drop=True)
            rank_of_true = None
            if reachable:
                match_idx = ranked.index[ranked["canonical_formula"] == tcanon]
                if len(match_idx):
                    rank_of_true = int(match_idx[0]) + 1

            top5 = ranked.head(5)["canonical_formula"].tolist()
            print(f"  [{source}-style target] true stability={truth['stability']:.3f}, gap={truth['gap']:.3f} "
                  f"-> rank of exact formula in {len(ranked)}-candidate universe: {rank_of_true}; "
                  f"top-5 nearest by predicted properties: {top5}")

            report_rows.append({
                "target": target_name, "canonical_formula": tcanon, "style": source,
                "true_stability": truth["stability"], "true_gap": truth["gap"],
                "reachable_in_universe": reachable,
                "rank_of_exact_formula": rank_of_true,
                "universe_size": len(ranked),
                "top5_nearest": "; ".join(top5),
            })
            for k in range(5):
                detail_rows.append({
                    "target": target_name, "style": source, "search_rank": k + 1,
                    "candidate": top5[k],
                    "pred_stability": float(ranked.iloc[k][col]),
                    "pred_gap": float(ranked.iloc[k]["pred_gap"]),
                    "distance": float(ranked.iloc[k]["distance"]),
                })

    report_df = pd.DataFrame(report_rows)
    detail_df = pd.DataFrame(detail_rows)
    report_df.to_csv(RESULTS_DIR / "a5_rediscovery_summary.csv", index=False)
    detail_df.to_csv(RESULTS_DIR / "a5_rediscovery_top5_neighbors.csv", index=False)

    n_exact_top1 = int((report_df["rank_of_exact_formula"] == 1).sum())
    n_exact_top10 = int((report_df["rank_of_exact_formula"] <= 10).sum())
    n_reachable = int(report_df["reachable_in_universe"].sum())
    summary = {
        "n_target_source_pairs": len(report_df),
        "n_reachable_in_search_universe": n_reachable,
        "n_exact_formula_ranked_top1": n_exact_top1,
        "n_exact_formula_ranked_top10": n_exact_top10,
        "interpretation": (
            "See a5_REPORT.md for the full discussion. Reported straight, including if negative."
        ),
    }
    save_json(summary, RESULTS_DIR / "a5_rediscovery_summary.json")
    print("\nSUMMARY:", summary)

    # LaTeX table
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Rediscovery test: for each Stage-1 Pareto material, held out of "
           r"surrogate training entirely, we search a composition-only universe "
           r"(training-pool formulas outside the holdout, unioned with a charge-balanced "
           r"$ABO_3$ generator over the 41 whitelisted cations) for the composition whose "
           r"surrogate-predicted stability/gap is closest to the held-out material's true "
           r"values. ``Rank'' is the position of the exact formula in that search "
           r"(\textemdash\ if unreachable); ``Nearest neighbour'' is the top search hit.}",
           r"\label{tab:a5_rediscovery}",
           r"\begin{tabular}{llrl}", r"\toprule",
           r"Target & Style & Rank of exact formula & Nearest search hit \\", r"\midrule"]
    for _, r in report_df.iterrows():
        rank_str = str(r["rank_of_exact_formula"]) if pd.notna(r["rank_of_exact_formula"]) else r"\textemdash"
        nearest = r["top5_nearest"].split("; ")[0]
        style_short = "Castelli" if "castelli" in r["style"] else "Wolverton"
        tex.append(f"{r['target']} & {style_short} & {rank_str} & {nearest} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(RESULTS_DIR / "a5_table_rediscovery.tex", "w") as f:
        f.write("\n".join(tex) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
