# Agent 5 — Independent, composition-only property surrogates

**Scope.** Two reviewer objections converge on one fix: the paper's "target-proximity
score" (`0.5*conditions[idx,0] + 0.5*conditions[idx,1]`, the audit brief) carries
no information about the *candidate*, and the paper's title promises dielectric design
while the pipeline never evaluates permittivity. This deliverable builds three
independent, **composition-only** surrogates — stability, band gap, dielectric
permittivity — trained without ever looking at the CVAE's conditioning vector, and
uses them to (a) replace the meaningless ranking with a real property-based one and
(b) run a rediscovery test of known good dielectrics. All numbers below come from
scripts in `analysis/a5_surrogate/` that were actually executed against the
data in this repository; nothing here is a literature value or an assumption.

Composition-only features: 132-dimensional Magpie elemental-statistics descriptors
(matminer's `ElementProperty.from_preset("magpie")` over a `pymatgen.Composition`
built from the parsed formula — mean/range/min/max/mode/avg-deviation of ~11
elemental properties). No structure, no CIF, no POSCAR is ever touched — the
generated candidates have none of that, so this is the only honest choice.

---

## 1. Stability and band-gap surrogates

**Data / split.** `dgx/outputs_dgx/composition_cvae_training_records.csv` (25,148
rows), split with the *exact* `split_indices(n, seed=42)` from
`composition_cvae_common.py` applied directly to the CSV row order (verified: the
CSV is written in the same concatenation order — Castelli, Wolverton, double
perovskites — that `load_material_table` produces, so this reproduces the CVAE's
own split exactly, not an approximation of it). Model class throughout: a 6-member
bootstrap ensemble of `RandomForestRegressor` (80 trees, max depth 20); the
ensemble spread (`std` across members) is the uncertainty reported everywhere,
and it should be read as **estimation/model uncertainty from resampling, not a
calibrated predictive interval**.

### The stability trap — verified and handled

`stability` pools two different physical quantities: Castelli formation energy
(`e_form`, referenced to elemental phases) and Wolverton energy above hull
(`e_hull`, referenced to the convex hull). We confirmed the offset directly from
this data: mean stability is **1.471 eV/atom for Castelli** vs **0.741 eV/atom for
Wolverton** — a systematic ~0.73 eV/atom difference in this table (this is not
identical to, but corroborates the same trap as, the 2.37 eV/atom offset another
agent measured elsewhere; the exact number depends on which sub-populations are
compared, but the qualitative finding — these are not the same quantity — is the
same). `double_perovskites_gap` rows carry `stability=0` as a placeholder
(`stability_known=0`); these were excluded entirely from stability training.

Two things were fit and both are reported, on purpose:

| Fit | n(test) | MAE | RMSE | R² |
|---|---|---|---|---|
| **Pooled (naive, cautionary — do not use)** | 2382 | 0.521 | 0.698 | **0.160** |
| Per-source, Castelli only (e_form) | 1872 | 0.521 | 0.698 | **0.074** |
| Per-source, Wolverton only (e_hull) | 510 | 0.430 | 0.584 | **0.117** |

**This is a documented negative result, not a success to talk around.** The naive
pooled model's R²=0.16 is *higher* than either honest per-source model — but that
lift is an artifact: the model is partly just learning "which of the two clusters
(with very different means) this formula probably belongs to," not real formation
chemistry. Once you separate the sources, composition-only Magpie features explain
only 7–12% of stability variance. **Composition-only stability prediction from this
particular training pool is weak.** This matters for the rediscovery test below and
should be stated plainly to the authors: a composition-only surrogate is not a
substitute for DFT-quality stability screening; it is at best a very rough filter.

### Band gap

Single model, all three sources pooled with a one-hot `source_dataset` feature
(gap is the same physical quantity everywhere, computed with GLLB-SC for
Castelli/double-perovskites and PBE for Wolverton).

| Subset | n(test) | MAE (eV) | RMSE (eV) | R² |
|---|---|---|---|---|
| All test rows | 2516 | 0.156 | 0.454 | **0.831** |
| **Non-zero-gap subset only** | 305 | 0.681 | 0.981 | **0.725** |

~88% of both the train/val and test splits have `gap == 0` exactly (train/val:
87.8%, test: 87.9% — matches the ~22,000/25,148 figure in the audit brief). The overall
R²=0.83 is real but is dominated by correctly predicting near-zero for the
majority class; **R²=0.725 on the non-zero subset is the honest number to quote**
for "can this model rank wide-gap candidates." It is a decent, useful surrogate for
that purpose, not an excellent one.

Parity plots: `a5_parity_stability_castelli_perovskites.pdf`,
`a5_parity_stability_wolverton_oxides.pdf`, `a5_parity_gap_all.pdf`,
`a5_parity_gap_nonzero.pdf`. Table: `a5_table_stability_gap.tex`. Full numbers:
`a5_stability_gap_metrics.json`. Models: `a5_stability_models.pkl` (dict keyed by
source), `a5_gap_model.pkl`.

---

## 2. Dielectric surrogate

**Data.** Primary: Materials Project `dielectric_constant.json.gz`, 1,056 records
(`formula`, `poly_electronic`, `poly_total`). Auxiliary: `phonon_dielectric_mp.json.gz`
(1,296 records) — **523 of these share an mp-id with the primary set** (i.e., they
are largely the same underlying MP calculations, repackaged with a phonon DOS
peak), so only the **772 non-overlapping rows** (one further row dropped as a
`>1000` soft-mode-divergence outlier) were added, strictly as **training-only**
auxiliary rows — they are never used to compute a reported metric, to avoid
conflating two possibly-different eps conventions in the evaluation itself.

**Method.** 1,056 primary rows, 10-fold CV (never a single train/test split — too
small for one split to be trustworthy), targets `log1p`-transformed, 8-member
bootstrap ensemble per fold. Metrics reported both in log space (what the model is
actually optimized on) and back-transformed to linear space via `expm1` (what a
materials scientist actually cares about).

| Target | n | MAE (log) | R² (log) | MAE (linear) | R² (linear) |
|---|---|---|---|---|---|
| ε∞ (poly_electronic) | 1056 | 0.146 | **0.801** | 2.03 | **0.224** |
| ε₀ (poly_total) | 1056 | 0.248 | **0.600** | 5.12 | **0.200** |

**The gap between log-space R² (0.60–0.80, good) and linear-space R² (0.20–0.22,
poor) is itself the finding.** A handful of very high-permittivity outliers
(poly_electronic up to 257, poly_total up to 278 in the primary set) dominate the
linear-space error; the model is good at getting the *order of magnitude* /
relative ranking right (hence decent log-space R²) but should not be trusted for
literal permittivity values, especially near the high tail.

### Verdict — stated plainly, as required

**This surrogate is good enough to REJECT the obviously bad candidates (screen out
low-permittivity compositions), but it is NOT good enough to finely RANK/prioritise
among plausible dielectric candidates.** Linear-space R² of ~0.20–0.22 means a
factor-of-2 to factor-of-5 error is common on individual predictions; treat its
output as a coarse "plausible vs implausible" gate, and treat its ranking among
already-plausible candidates as indicative, not authoritative. This verdict is saved
machine-readably in `a5_dielectric_verdict.json`.

Parity plots: `a5_parity_dielectric_poly_electronic.pdf`, `a5_parity_dielectric_poly_total.pdf`
(log-log axes). Table: `a5_table_dielectric.tex`. Full numbers:
`a5_dielectric_metrics.json`. Model: `a5_dielectric_models.pkl`.

---

## 3. Applying the surrogates to the 66 candidates

Scored: all `valid==True` rows of `composition_cvae_generated_candidates.csv`,
deduplicated on `canonical_formula` — **66 unique compositions**, matching
the audit brief exactly. Full per-candidate CSV:
`a5_candidate_predictions.csv`. Ranking table (top 15):
`a5_table_candidate_ranking.tex`.

The candidates have no known source label (Castelli-style vs Wolverton-style), so
both per-source stability predictions are reported, and the band-gap prediction is
averaged over the three possible one-hot source settings (with the across-setting
spread folded into the reported uncertainty as `pred_band_gap_source_sensitivity`).
The new ranking (`predicted_property_rank_score`) combines normalized predicted
stability (Wolverton/e_hull-style, 40%), band gap saturating at ~4 eV (30%), and
log-total-permittivity (30%) — a transparent, documented replacement for the old
meaningless proximity score, which is retained in the CSV (`candidate_score`
column) purely for side-by-side comparison, not as an input to anything.

### Band-gap / insulator flag — surfaced, not buried

Using the paper's own Eg ≥ 1.0 eV insulator threshold (the same filter Stage-1
applies to the dielectric pool):

- **0/66** candidates have a *point-estimate* predicted band gap below 1.0 eV.
- **However, 58/66 (88%)** have `mean − 1·std` fall below 1.0 eV — i.e. the
  surrogate's own uncertainty is wide enough that "confidently insulating" cannot
  be claimed for the great majority of candidates. This is flagged explicitly as
  `flag_uncertain_insulator` in the CSV, precisely so it is not buried behind a
  point estimate that looks reassuring.
- The lowest-confidence, lowest-gap-point-estimate candidates cluster very
  strongly around **Ti- and Sn-containing formulas**: of the 20 lowest predicted
  band gaps, 17 contain Ti or Sn. **35/66 (53%)** of all valid candidates contain
  Ti or Sn at all. This is exactly consistent with the independently-reported
  finding (cited in the task) that 36% of the "charge-balanced" candidates require
  open-shell Ti³⁺ or Sn²⁺ — those oxidation states plausibly narrow the gap, and
  the surrogate (trained on real DFT gaps) is picking up a version of that same
  signal, without ever being told about oxidation states directly. **This should
  be read as corroborating evidence for the reviewers' concern, not a clean
  refutation of it** — the surrogate's band-gap R² on non-zero gaps is only 0.72,
  so individual low-gap calls should be verified with DFT before being treated as
  ground truth.

---

## 4. The rediscovery test

**Question.** Hold out SrHfO₃, BaHfO₃, CaHfO₃, BaZrO₃, LaAlO₃, YAlO₃ (the Stage-1
Pareto set) and ask whether a composition-only approach, conditioned on their
*true* property targets, recovers them or close chemical neighbours.

**A finding before the test even runs:** all six of these compounds are, in fact,
**already present in the CVAE's own 25,148-row training table** (confirmed by
canonical-formula match — e.g. `SrHfO3` → `HfO3Sr` appears from both Castelli and
Wolverton, in both cation-order spellings, 4 rows total; 22 rows total across all
six). They are not held out of the CVAE's training set at all. So this is not
simply "did the CVAE ever see this" — it manifestly did — the question this test
actually answers is whether a *pure surrogate-guided, composition-only search*
(no decoder) can find them from scratch.

**Method.** We removed all 22 rows matching any of the six canonical formulas from
*all* surrogate training data (not just the test split) and retrained fresh
holdout stability and band-gap models. We then built a composition-only search
universe: every unique formula remaining in the training pool, unioned with a
combinatorial `ABO3` generator over the 41 whitelisted "common ceramic" cations,
filtered to charge-balanced combinations only (reusing the project's own
`charge_balance_possible`/`OXIDATION_STATES` tables — the same chemistry rules
Stage 2's decoder uses). We built and checked **two** versions of this universe:
an unrestricted one (12,152 compositions, allowing any element from the raw
Castelli/Wolverton pool — which, as the audit brief notes, has no
element-risk filter applied in Stage 1) and a **practical, whitelist-only** one
(1,311 compositions, every formula restricted to the same 41-element
`COMMON_CERAMIC_ELEMENTS` set Stage 2's own post-filter uses). The unrestricted
universe's nearest neighbours for some targets turned out to include Pa/U/Np
(actinide) formulas (e.g. `BaO3Pa`, `BaNpO3`) inherited from the raw training
pool — chemically "close" in predicted-property space but obviously impractical
as dielectric candidates, and a small illustration in its own right of why an
element-risk filter matters at the Stage-1 pool level, not just at Stage 2's
decode step. **We report the whitelist-only universe below as the primary,
practically-meaningful result**; the unrestricted-universe numbers are in
`a5_rediscovery_test.py`'s git history / re-runnable but are not the headline.
Every universe candidate was scored with the *holdout* surrogates and ranked by
distance (in normalized predicted-property space) to each held-out compound's
**true** stability/gap (taken from its most-stable recorded polymorph per source,
since composition alone maps to more than one recorded structure/energy for
several of these formulas — itself a small reminder that composition-only
stability targets are ambiguous).

**Result — reported straight (whitelist-only, 1,311-candidate universe):**

- All six compounds were reachable in the search universe (the charge-balanced
  `ABO3` generator does produce them).
- **None (0/12) of the target/source combinations had the exact formula ranked
  #1, and none ranked in the top 10** out of 1,311 candidates. Ranks for the
  Castelli-style targets ranged from 37 (LaAlO₃) to 160 (BaHfO₃); Wolverton-style
  ranks ranged from 18 (LaAlO₃) to 180 (BaZrO₃) — no consistent pattern between
  the two stability conventions here, consistent with both being weak
  composition→stability fits (R²=0.07–0.12) rather than one being reliably worse.
- **But the top-5 nearest hits are, in most cases, extremely close chemical
  relatives**: for every Castelli-style target the top search hits are drawn from
  `{AlGdO3, CaO3Zr, O3PrSc, O3SrZr, NdO3Sc, AlDyO3, ...}` — i.e. rare-earth
  aluminates/scandates and alkaline-earth zirconates that are isovalent, same-
  family substitutions on the A- or B-site of the actual target (Sr↔Ca↔Ba on the
  A-site of a zirconate/hafnate; La↔Nd↔Pr↔Gd↔Y on the A-site of an aluminate/
  scandate). For YAlO₃ specifically, the top-5 (Wolverton-style) are
  `AlGdO3, AlDyO3, AlHoO3, AlO3Tb, AlErO3` — literally the lanthanide-aluminate
  family YAlO₃ belongs to. The Wolverton-style neighbours for BaHfO₃ and BaZrO₃
  are somewhat less clean (`MgO6SrTiZr`, `CaO3Ti`, etc. — still alkaline-earth
  titanate/zirconate chemistry, but drifting toward Ti rather than staying on
  Hf/Zr), which we read as the weaker Wolverton-style stability fit showing up
  as noisier neighbour selection, not as a different qualitative result.

**Honest interpretation.** A composition-only surrogate search does **not**
rediscover the *exact* known-good compound at high rank — with stability R² of
only 0.07–0.12, that is not surprising, and should not be oversold. What it *does*
reliably do is land the search in the immediately correct chemical neighbourhood —
the right family of isovalent substitutions on the right sublattice. **This is a
genuinely positive, reportable result for the "close chemical neighbours" half of
the question, and a genuinely negative result for the "recovers them" half.** We
recommend the authors describe the rediscovery test this way rather than as an
unqualified success: the surrogate (and by extension a property-conditioned
generative approach built on the same composition-only physical information) can
be expected to find the right chemical family, not the literal best-known
member of it, given how weak the underlying composition→stability signal is.

Per-target detail: `a5_rediscovery_summary.csv` (rank + top-5 per target/source),
`a5_rediscovery_top5_neighbors.csv` (full nearest-neighbour table with predicted
properties), `a5_rediscovery_summary.json` (aggregate counts), and
`a5_table_rediscovery.tex`.

---

## What each surrogate can and cannot support (summary for the authors)

1. **Stability surrogate**: use only as a same-source, same-convention screen
   (e.g., "is this more or less stable than typical Castelli perovskites").
   Composition alone explains only ~7–12% of variance once the Castelli/Wolverton
   pooling trap is removed — do **not** quote a pooled R² of 0.16 as if it were a
   real predictive result; that number is a pooling artifact, not a stability
   signal, and we say so on the record here.
2. **Band-gap surrogate**: reasonably trustworthy for the "is this likely to be a
   wide-gap insulator" question (R²=0.72 on the non-zero-gap subset), and this is
   the surrogate we would lean on hardest of the three. Still verify any specific
   low-gap call with DFT before using it to reject a candidate outright, given the
   Ti/Sn open-shell risk already flagged elsewhere in the project.
3. **Dielectric surrogate**: screening only. Log-space R² is good (0.60–0.80) but
   linear-space R² (0.20–0.22) means point predictions of ε can be off by a factor
   of several. Use it to throw out clearly-bad candidates; do not use it to break
   ties between two plausible candidates.
4. **Rediscovery test**: supports the claim "a composition-only, property-targeted
   search lands in the right chemical family" but does **not** support the claim
   "recovers the exact known-best material." State it that way in the revision.

## Files delivered (all under `results/`, `a5_` prefix)

- Scripts (also kept in `analysis/a5_surrogate/`, importable/re-runnable):
  `a5_common.py`, `a5_featurize_training.py`, `a5_train_stability_gap.py`,
  `a5_train_dielectric.py`, `a5_score_candidates.py`, `a5_rediscovery_test.py`.
- Models: `a5_stability_models.pkl`, `a5_gap_model.pkl`, `a5_dielectric_models.pkl`.
- Metrics/JSON: `a5_stability_gap_metrics.json`, `a5_dielectric_metrics.json`,
  `a5_dielectric_verdict.json`, `a5_rediscovery_summary.json`.
- Parity plots (vector PDF): `a5_parity_stability_castelli_perovskites.pdf`,
  `a5_parity_stability_wolverton_oxides.pdf`, `a5_parity_gap_all.pdf`,
  `a5_parity_gap_nonzero.pdf`, `a5_parity_dielectric_poly_electronic.pdf`,
  `a5_parity_dielectric_poly_total.pdf`.
- LaTeX tables (booktabs): `a5_table_stability_gap.tex`, `a5_table_dielectric.tex`,
  `a5_table_candidate_ranking.tex`, `a5_table_rediscovery.tex`.
- CSVs: `a5_candidate_predictions.csv` (all 66 candidates, full predictions +
  flags + new ranking), `a5_rediscovery_summary.csv`,
  `a5_rediscovery_top5_neighbors.csv`.
