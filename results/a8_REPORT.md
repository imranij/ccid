# a8 — Fixing the free-bits bug and retraining the Composition-CVAE

**MTCOMM-D-26-04679.** Working directory `analysis/a8_retrain`; outputs prefixed `a8_`.

---

## Direct answer

**The fix works, and it works dramatically — but the model still does not beat random.**

* The bug is real and is the whole cause of the collapse. Correct per-dimension free bits
  take the latent from **1 active dimension out of 32 to 32 out of 32**, total KL from
  **0.175 to 29.8 nats**, and test reconstruction MSE from **0.1279 to 0.0195** (6.5× better).
* Downstream, distinct valid formulas from 20,000 samples go from **69 ± 2 to 1,752 ± 12**
  (25×), and the fixed-condition collapse is gone: with the condition pinned at the paper's
  own target (0.9, 0.8), 20,000 latent draws now decode to **1,821 distinct valid formulas
  instead of 1**.
* Validity can be pushed **above** the deposited model's — 0.4415 ± 0.0031 at λ = 0.5 versus
  0.3302 ± 0.0018 — while simultaneously giving 16× more unique formulas.
* **But random selection through the identical filter still wins on both axes:**
  0.7274 ± 0.0017 validity and 11,821 ± 51 unique valid formulas. The best retrained cell on
  validity reaches 0.511; the best on diversity reaches 1,923 unique valid. Neither is close.

So Audit finding **B** (posterior collapse, one-line bug) is *fixed and was worth fixing*, and
Audit finding **A** (the generative model is outperformed by random sampling) **survives the
fix**. The honest framing of the paper does not change: composition-only generation through a
rule-based charge-balance filter does not beat random selection from the same element
whitelist. What does change is that the paper can now say this about a *correctly trained*
model, which is a far stronger and more defensible claim than saying it about a model with a
dead latent space.

---

## 1. Reproducing the pathology on the deposited checkpoint

Script: `a8_exp0_baseline_pathology.py` → `a8_baseline_pathology.json`.
Computed by encoding all 25,148 records of the material table through
`dgx/outputs_dgx/composition_cvae_checkpoint.pt` (epoch 150).

| Quantity | Value |
|---|---|
| Total KL of the aggregate posterior | **0.1746 nats/sample** |
| Latent dimensions with KL > 0.01 nats | **1 of 32** |
| KL of the single surviving dimension | 0.17437 nats |
| KL of the next-largest dimension | 2 × 10⁻⁵ nats |
| Test-split composition MSE (deposited file) | 0.12786 |

**Fixed-condition collapse.** Condition pinned at (0.9, 0.8), 20,000 independent draws
z ~ N(0, I):

* **1 unique *valid* formula** (Al₂HfO₇Zr) — this reproduces the audit's "20,000 draws decode
  to one formula" exactly.
* 4 distinct decoded formulas in total, of which the modal one (Al₂O₄ScZr, rejected by the
  filter) accounts for **90.74 %** of all draws.
* The decoder output standard deviation across the 20,000 latent draws is 6.5 × 10⁻⁴ per
  element on average — i.e. the decoder is numerically almost independent of z.
* At (0.5, 0.5) the same test gives 7 unique valid formulas and a modal share of 91.3 %.

**Independent confirmation that the loss is the cause.** I retrained on CPU with the
*unmodified* deposited `cvae_loss` (`buggy_b0.1`, 150 epochs, seed 42, identical settings) and
recovered the pathology from scratch: total KL 0.1732 nats, **1 of 32** active units, test
composition MSE 0.12710 — against the deposited checkpoint's 0.1746 / 1 / 0.12786. The bug
reproduces; nothing else about the deposited run is needed to explain the collapse.

A second control makes the mechanism unambiguous. Setting the corrected per-dimension budget
to **λ = 0** (i.e. correct code, no free bits at all) at β_max = 0.1 gives total KL 0.1696 and
1 active unit — statistically the same model as the buggy clamp. **The deposited "free-bits"
guard is worth exactly as much as having no guard at all.** At β_max = 0.5 and λ = 0 the KL
reaches 1.6 × 10⁻⁶ nats, i.e. total collapse.

---

## 2. The fix

Delivered as a clean unified diff against the authors' repo:
**`a8_composition_cvae_common_free_bits.patch`** (applies with `patch -p1` from the project
root; verified with `patch --dry-run`, and it preserves the file's CRLF line endings).

```python
# before
kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())   # summed over ALL dims
free_bits = 0.5 * mu.shape[1]                                  # one scalar = 16.0 nats
loss = comp_loss + property_weight*prop_loss + beta*torch.clamp(kl, min=free_bits)

# after
kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())       # (batch, latent)
kl = kl_elem.sum()                                             # unchanged, for logging
kl_per_dim = kl_elem.mean(dim=0)                               # (latent,)
kl_term = x.shape[0] * torch.clamp(kl_per_dim, min=free_bits_per_dim).sum()
loss = comp_loss + property_weight*prop_loss + beta*kl_term
```

Two points the authors should note in the rebuttal:

1. The floor is applied **per dimension before summing** (Kingma et al., NeurIPS 2016). The
   deposited code clamped the *sum*, which makes the term a constant with zero gradient
   whenever the total KL is below 16 nats — the opposite of the intended effect.
2. The multiplication by `x.shape[0]` is deliberate: `comp_loss` and `prop_loss` use
   `reduction="sum"` and the old `kl` was also a sum over batch *and* dimensions, so this keeps
   the corrected term in the same units and `beta_max` keeps its previous meaning. Without it
   the effective β would silently change by a factor of 128.

The signature gains one keyword argument, `free_bits_per_dim` (default 0.05), and is otherwise
backward compatible.

Reference implementation used by the sweep: `a8_loss.py`. Trainer: `a8_train.py`, a
single-process CPU re-implementation of `train_composition_cvae_dgx.py` with everything except
the KL term held identical — same data table (25,148 records), same `min_count=5` element
vocab, same seed-42 80/10/10 split, same 79→256→128→32 architecture, same
`AdamW(lr=1e-3)`, same `CosineAnnealingLR(T_max=150)`, same grad-norm clip 1.0, batch 128,
150 epochs, same linear β warm-up over 50 epochs, same `property_weight=2.0`.

**Wall-clock cost (measured).** A full 150-epoch run takes **147–433 s** (median 320 s,
≈ 5.3 min) on this 2-core CPU box with two runs sharing the two cores, plus ≈ 20 s to build
the data table. 31 checkpoints were trained for a total of **2.8 CPU-hours**. A full retrain
is entirely tractable here; no GPU is needed.

---

## 3. The sweep and the honest trade-off curve

Scripts `a8_run_sweep.sh`, `a8_run_sweep2.sh`, `a8_run_sweep3.sh`, `a8_train.py`,
evaluated by `a8_eval.py`, merged by `a8_merge_sweep.py`.
Full results: **`a8_sweep_all.csv`** (31 retrained checkpoints + the deposited one; stage files
`a8_sweep_results.csv`, `..._stage2.csv`, `..._stage3.csv`).
LaTeX: **`a8_tables.tex`**, table `tab:a8_freebits_sweep`.

Grid: per-dimension budget λ ∈ {0, 0.05, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 4, 8} nats ×
β_max ∈ {0.001, 0.01, 0.1, 0.5, 1.0}. Generation metrics are 20,000 samples through the
project's own `decode_composition_vector` at the archived settings (target 0.9/0.8, jitter 0.30
— the setting the audit established actually produced the archived run).

Selected rows at β_max = 0.01 (the full grid is in the CSV):

| λ (nats/dim) | Recon MSE | Total KL | Active units | Validity | Unique valid | Fixed-cond. unique valid |
|---|---|---|---|---|---|---|
| deposited (buggy) | 0.1279 | 0.17 | 1 | 0.3313 | 68 | **1** |
| 0 | 0.0793 | 1.70 | 2 | 0.3854 | 127 | 11 |
| 0.05 | 0.0773 | 2.64 | 32 | 0.3446 | 148 | 5 |
| 0.25 | 0.0677 | 7.40 | 32 | 0.3710 | 393 | 36 |
| 0.5 | 0.0429 | 14.80 | 32 | **0.4401** | 1,095 | 849 |
| 0.75 | 0.0294 | 22.38 | 32 | 0.3565 | 1,852 | 1,821 |
| 1.0 | 0.0195 | 29.82 | 32 | 0.2925 | 1,735 | 1,821 |
| 1.25 | 0.0135 | 37.49 | 32 | 0.2244 | 1,856 | 1,970 |
| 1.5 | 0.0149 | 45.14 | 32 | 0.1904 | 1,844 | 1,878 |
| 2.0 | 0.0096 | 60.56 | 32 | 0.0597 | 820 | 832 |
| 4.0 | 0.0089 | 120.51 | 32 | 0.0303 | 504 | 496 |
| 8.0 (β=0.1) | 0.0171 | 229.38 | 32 | 0.0051 | 97 | 40 |

**What the curve says.**

* Reconstruction improves monotonically with λ (0.128 → 0.009) — unusually, there is *no*
  reconstruction/KL trade-off in the usual direction over most of the range, because the
  deposited model was so far inside the collapsed regime that any latent information helps.
* **Active units saturate at 32/32 for every λ ≥ 0.05.** Free bits, applied correctly, are an
  extremely effective guard. That alone answers the reviewers' posterior-collapse objection.
* The real trade-off is **validity versus diversity**, and it is sharp. Validity peaks near
  λ ≈ 0.5 (0.44–0.51) and falls off a cliff beyond λ ≈ 1.5 (0.06 at λ = 2, 0.005 at λ = 8),
  while unique valid formulas peak on a broad plateau at λ ≈ 0.75–1.5 (~1,850–1,920) and then
  also collapse. At λ ≥ 2 the latent is so information-rich that samples from the N(0, I) prior
  land in regions the decoder never sees during training, and the compositions it emits stop
  being charge-balanceable.
* β_max matters much less than λ. Lowering β_max improves reconstruction and diversity
  slightly; raising it to 0.5–1.0 improves validity slightly at the cost of unique formulas.
  The exception is λ = 0, where raising β_max drives the model to *total* collapse — exactly
  the failure the free bits are supposed to prevent.

**Two recommended settings, depending on what the paper wants to claim:**

| | λ | β_max | Validity | Unique valid | Recon MSE |
|---|---|---|---|---|---|
| Best joint validity/diversity | 0.5 | 0.01 | 0.4415 ± 0.0031 | 1,108 ± 11 | 0.0429 |
| Most diverse | 1.0 | 0.01 | 0.2913 ± 0.0028 | 1,752 ± 12 | 0.0195 |

(The single highest unique-valid cell in the whole sweep is λ = 1.0, β_max = 0.001 at 1,923;
the single highest validity is λ = 0.5, β_max = 1.0 at 0.5113. Both are within the plateau.)

---

## 4. Full downstream re-evaluation, against the existing baselines

Script `a8_exp_downstream.py`, which imports `a2_common`, `a2_exp2_random_baseline` and
`a2_exp4_condition_sweep` verbatim, so every filter, novelty test and estimator is
byte-identical to the a2 baselines. 20,000 samples per run, 5 seeds for generation and
3 seeds for the controls. Outputs: `a8_downstream_fb0.5_b0.01_*`, `a8_downstream_fb1.0_b0.01_*`,
`a8_headline_comparison.csv`, LaTeX table `tab:a8_headline`.

### 4.1 Does the fixed model beat random selection? **No.**

| Generator | Validity | Unique valid | Novelty | Mean pairwise dist. | Distinct decoded |
|---|---|---|---|---|---|
| Deposited CVAE | 0.3302 ± 0.0018 | 69 ± 2 | 0.9855 | 0.2785 | 305 ± 9 |
| **Retrained, λ = 0.5** | **0.4415 ± 0.0031** | 1,108 ± 11 | 0.9581 | 0.3226 | 5,395 ± 27 |
| **Retrained, λ = 1.0** | 0.2913 ± 0.0028 | **1,752 ± 12** | 0.9688 | 0.3265 | 10,011 ± 53 |
| Random, uniform over 41-element whitelist | **0.7274 ± 0.0017** | **11,821 ± 51** | 0.9977 | 0.3437 | 16,512 ± 30 |
| Random, Dirichlet (no model input at all) | 0.6778 ± 0.0042 | 11,636 ± 96 | 0.9789 | 0.3755 | 17,305 ± 41 |
| Random, training-frequency weighted (whitelist) | 0.6606 ± 0.0028 | 8,850 ± 47 | 0.9934 | 0.3457 | 13,739 ± 61 |

The random arms were recomputed inside this experiment from the *retrained* model's decoded
fraction profiles (block B of `a8_exp_downstream.py`) and land on the same numbers as a2's
(0.709–0.716 validity, 12,175–12,311 unique valid for `rand_uniform41`), so the comparison is
not an artefact of reusing a2's decoded profiles.

Quantitatively, after the fix the model still produces **1.6× fewer valid candidates** and
**6.7× fewer distinct valid formulas** than uniform random selection from its own whitelist.
Against the enumerated reachable output space of `decode_composition_vector`
(129,205 valid formulas, from a2's reachability enumeration) the retrained model covers
**1.36 %** and random covers **9.1 %**; the deposited model covered 0.05 %.

The gap narrowed from 180× (deposited) to 6.7× (fixed), which is a large and reportable
improvement, but the sign of the comparison does not change.

### 4.2 The unconditional control — the collapse to one formula is gone

Table `tab:a8_unconditional`. Latent draw shared within each seed, so every difference is
attributable to the condition vector alone.

| Condition arm | Deposited validity | Deposited unique valid | Retrained (λ=0.5) validity | Retrained unique valid |
|---|---|---|---|---|
| Archived conditioning | 0.3302 | 69 ± 2 | 0.4411 | 1,106 ± 14 |
| Same conditions, permuted across samples | 0.3299 | 73 ± 2 | 0.4406 | 1,119 ± 14 |
| Constant (0.5, 0.5) | 0.9887 | 6 ± 1 | 0.3730 | 1,403 ± 14 |
| **Constant (0.9, 0.8)** | 0.0713 | **1 ± 0** | 0.4744 | **865 ± 18** |
| Constant training mean | 0.0000 | 0 ± 0 | 0.0129 | 210 ± 12 |
| Uniform (0,1)² | 0.3749 | 76 ± 1 | 0.3473 | 1,345 ± 14 |

The decisive row is "constant (0.9, 0.8)": the deposited model yields exactly one formula from
20,000 latent draws, the retrained model yields 865 (and 1,821 at λ = 1.0). The latent space
is doing real work now.

**However**, the "shuffled" row is still indistinguishable from the "conditioned" row
(0.4406 vs 0.4411 validity; 1,119 vs 1,106 unique valid). Breaking the pairing between z and
its condition costs nothing. The conditioning channel remains weak after the fix; it is not
what the diversity is coming from.

### 4.3 Conditional-generation sweep

Block D, 12 cells on a (stability utility, gap utility) grid, 4,000 samples per cell, shared
latent draw, mutual information with a 20-fold label-permutation null.

| | Deposited | Retrained λ=0.5 | Retrained λ=1.0 |
|---|---|---|---|
| I(cell ; formula), bits | 3.045 (null 0.012) | 1.757 (null 0.832) | 1.849 (null 1.569) |
| I(cell ; element), bits | 1.775 (null 0.0008) | 0.450 (null 0.0028) | 0.229 (null 0.0041) |
| H(formula), bits | 3.948 | 10.106 | 12.683 |
| Normalised MI (formula) | 0.771 | 0.174 | 0.146 |
| Paired formula-switch rate | 0.959 | 0.916 | 0.906 |

This needs careful reading and is easy to misreport. The deposited model's *normalised* MI is
high (0.77) not because it is well conditioned but because its output entropy is tiny
(3.95 bits): when a model emits four formulas, knowing the condition tells you almost
everything. The retrained model's absolute element-level MI (0.450 bits at λ = 0.5) is still
**160× its permutation null**, so the condition genuinely steers the chemistry; it simply
accounts for a smaller share of a now much larger output entropy (10.1 bits). Validity also
varies systematically and strongly across the grid for the retrained model (0.15 at
s=1.0, g=0.5 up to 0.79 at s=0.2, g=1.0), which the deposited model could not do meaningfully.
Per-cell numbers: `a8_downstream_*_condition_sweep_cells.csv`.

### 4.4 Novelty and diversity

Novelty (exact canonical-formula match against the 25,148-record training table) *falls*
slightly after the fix, from 0.9855 to 0.9581 (λ = 0.5) / 0.9688 (λ = 1.0). This is expected
and is a good sign: a model that has actually learned the training distribution rediscovers
more of it. Mean pairwise composition distance among novel unique valid candidates rises from
0.2785 to 0.3226 / 0.3265, approaching but not reaching random's 0.3437–0.3755.

The caveat from audit finding E stands unchanged: this novelty is exact string matching only.
**External verification against Materials Project, OQMD, AFLOW, COD or Alexandria could not be
performed here** — every provider is blocked by this environment's egress policy. No absolute
novelty claim is supportable from these numbers.

---

## 5. Is the decoder a bottleneck? Partly, and only after the latent is fixed

Block E, `a8_downstream_*_top_cations.csv`, LaTeX table `tab:a8_top_cations`.
20,000 samples, 3 seeds, varying `top_cations` in `decode_composition_vector`
(`max_cation_count = 4` left at its deposited value).

| Model | top_cations | Validity | Unique valid | Distinct decoded |
|---|---|---|---|---|
| Deposited | 2 | 0.5584 | 25 ± 1 | 126 |
| | 3 (deposited) | 0.3298 | 69 ± 2 | 308 |
| | 4 | 0.3771 | 65 ± 4 | 313 |
| | 5 | 0.5210 | 80 ± 2 | 343 |
| Retrained λ=0.5 | 2 | 0.4891 | 303 ± 3 | 1,662 |
| | 3 | 0.4411 | 1,106 ± 14 | 5,414 |
| | 4 | 0.3584 | 1,346 ± 33 | 7,020 |
| | 5 | 0.3431 | 1,418 ± 23 | 8,090 |
| Retrained λ=1.0 | 2 | 0.3781 | 469 ± 11 | 2,749 |
| | 3 | 0.2907 | 1,746 ± 10 | 9,983 |
| | 4 | 0.2298 | 2,134 ± 8 | 13,542 |
| | 5 | 0.1855 | 2,153 ± 36 | 15,460 |

**Interpretation.** On the collapsed model, `top_cations` is almost irrelevant — 3 → 5 moves
unique valid formulas from 69 to 80. On the retrained model the same change gives +28 %
(1,106 → 1,418) and +23 % (1,746 → 2,153), at a cost of roughly a third of the validity rate.
So the coarse decoder was a *secondary* bottleneck that only became visible once the primary
one (the dead latent) was removed. It is worth one sentence in the paper, but relaxing it does
not change the comparison with random: even at `top_cations = 5` the model reaches 2,153
unique valid formulas against random's 11,821.

---

## 6. What I would caution the authors about

1. **Report the fix and the negative result together.** "We identified and corrected a
   free-bits implementation error that had collapsed 31 of 32 latent dimensions; after
   correction the latent is fully active, reconstruction improves 6.5×, and distinct valid
   candidates increase 25-fold — and the model still does not outperform random selection
   through the same rule-based filter" is a strong, self-consistent, publishable story. Either
   half alone is weaker: the fix alone overclaims, the negative result alone invites the
   reviewer to ask whether a correctly trained model would have done better. **It would not,
   and now you can say so with evidence.**
2. **Do not quote the sweep's best cell as "the model's performance".** The trade-off is real
   and steep. Table `tab:a8_freebits_sweep` should go in the paper or the SI in full.
3. **The conditioning channel is still weak.** Shuffling the conditions across samples costs
   essentially nothing (§4.2). The free-bits fix repaired the *latent*, not the *conditional*
   part of the CVAE. If the paper wants to claim conditional inverse design, that needs
   separate work — the element-level MI is well above its null, but the paired-shuffle control
   is the one a reviewer will run, and it is not passed.
4. **Audit finding C is untouched by this work.** The "target-proximity score" in
   `generate_composition_candidates.py` line 149 is still the mean of two random condition
   draws and still contains no information about the decoded candidate. Retraining does not fix
   that; deleting or replacing the score does.
5. **The deposited checkpoint should be replaced in the repository**, and the archived
   `composition_cvae_generated_candidates.csv` regenerated, if any retrained result is cited.
   Mixing numbers from the two would be worse than either.
6. **Everything here is CPU-reproducible in under three hours.** There is no reason for the
   revision to report a number the referees cannot re-derive.

---

## Provenance

* **Computed in this session** (all tables and numbers above unless marked otherwise):
  `a8_baseline_pathology.json`, `a8_sweep_all.csv` (+ three stage files),
  `a8_headline_comparison.csv`, `a8_downstream_*` CSV/JSON, `a8_tables.tex`,
  31 checkpoints in `analysis/a8_retrain/checkpoints/`.
* **Read from existing files:** the deposited test metrics (`composition_mse` 0.12786,
  `kl` 0.17979) from `dgx/outputs_dgx/composition_cvae_test_metrics.json`; the random-baseline
  arms and the 129,205-formula decoder reachability count from a2's
  `a2_random_baseline_runs.csv`, `a2_random_baseline_summary.json`,
  `a2_unconditional_runs.csv`, `a2_condition_sweep_summary.json`, `a2_decoder_reachability.json`.
* **Interpretation** is confined to the paragraphs labelled as such and to §6.
* **Could not be computed:** external novelty verification against structure databases — all
  providers blocked by this environment's egress policy (unchanged from a2/a3).

## Files

Scripts (`analysis/a8_retrain/`): `a8_common.py`, `a8_loss.py`, `a8_train.py`,
`a8_exp0_baseline_pathology.py`, `a8_eval.py`, `a8_merge_sweep.py`, `a8_exp_downstream.py`,
`a8_make_tables.py`, `a8_run_sweep.sh`, `a8_run_sweep2.sh`, `a8_run_sweep3.sh`.
Checkpoints and per-run histories: `analysis/a8_retrain/checkpoints/` (45 MB, 31 runs,
each with `*_history.csv` and `*_test_metrics.json`). Training logs: `.../logs/`.

Results (`results/`): `a8_composition_cvae_common_free_bits.patch`,
`a8_baseline_pathology.json`, `a8_sweep_all.csv`, `a8_sweep_results.csv`,
`a8_sweep_results_stage2.csv`, `a8_sweep_results_stage3.csv`, `a8_headline_comparison.csv`,
`a8_downstream_fb0.5_b0.01_*.csv/json`, `a8_downstream_fb1.0_b0.01_*.csv/json`,
`a8_downstream_deposited_top_cations.csv`, `a8_downstream_fb0.5_b0.01_candidates_seed123.csv`,
`a8_downstream_fb1.0_b0.01_candidates_seed123.csv`,
`a8_tables.tex`, `a8_REPORT.md`.
