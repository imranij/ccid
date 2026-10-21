# a2 — Composition-CVAE: reproduction, controls, and what the generator demonstrably does

Agent a2, MTCOMM-D-26-04679. All numbers below were **computed** by the scripts in
`analysis/a2_generative/` on the deposited checkpoint
`dgx/outputs_dgx/composition_cvae_checkpoint.pt`, unless explicitly labelled
"read from file" or "interpretation". Every candidate passes through the project's own
`decode_composition_vector`, `charge_balance_possible` and `COMMON_CERAMIC_ELEMENTS`
whitelist, imported directly from `dgx/code/composition_cvae_common.py`, so the
baselines are filtered identically to the manuscript's candidates.

---

## Executive summary

1. **The deposited checkpoint does reproduce the archived run — but only at
   `--condition-jitter 0.30`, which is not the deposited script's default (0.08).**
   At the default the same checkpoint gives 22.3 % validity and ~7 unique formulas,
   not 33 % and ~69. This must be corrected in the methods section.
2. **The random control is decisive and unfavourable.** Uniform random element
   selection from the same 41-element whitelist, pushed through the identical decode
   and filter path, gives **72.7 % validity and 11,821 unique valid formulas**, versus
   the CVAE's **33.0 % and 69**. The CVAE is beaten by random selection by 2.2× on
   validity and by 171× on unique formulas. Validity is a property of the rule-based
   post-filter, exactly as Reviewer 2 suspects — and the learned model makes it *worse*.
3. **Conditioning does steer chemistry — strongly.** `I(condition cell; decoded formula)`
   = 3.04 bits out of a maximum 3.58 (permutation null 0.012 bits); 96 % of samples
   change formula when only the condition changes. So the paper's qualitative claim
   survives. What does *not* survive is the claim that this is a generative model:
   with the condition held fixed, 20,000 latent draws produce as few as **1 unique
   valid formula**.
4. **The cause of low diversity is total posterior collapse, not a coarse decoder.**
   31 of 32 latent dimensions have exactly zero KL to the prior; the surviving one
   carries 0.174 nats (0.25 bits). The decoder can emit **129,205** distinct
   charge-balanced whitelist formulas; the CVAE reaches **68** of them (0.05 %) while
   random selection through the same decoder reaches 11,744 (9.1 %). *This contradicts
   the prior in the audit brief — the decoder is not the bottleneck.*
5. The paper's element-risk filter looks harmless (108/20,000 rejections) only because
   the charge-balance step runs first. **16.5 % of decoded samples contain at least one
   non-whitelist element** (Ac 2,184; Pa 1,646; Eu 820 occurrences), and 96.7 % of those
   are silently booked as charge-balance failures instead.

---

## 1. Reproduction (`a2_exp1_reproduce.py`, `a2_seed_forensics.py`)

### 1.1 The archived run used a jitter the deposited script does not default to

The archived `composition_cvae_generated_candidates.csv` condition columns have s.d.
0.207 (stability) and 0.236 (gap) after clipping — far too wide for `jitter = 0.08`.
The clipping mass at 1.0 implies σ = 0.3005 and 0.3055 respectively. Searching seeds
0–199,999, `np.random.default_rng(123).normal(0.9, 0.30, ·)` reproduces the archived
condition column **to 1.1e-16**. So the archived run was
`--seed 123 --condition-jitter 0.30`.

Per-sample formulas do *not* reproduce (54.7 % agreement) because the latent draw
`torch.randn(...)` was made on a different RNG/device (almost certainly the DGX GPU).
Aggregate metrics do reproduce.

### 1.2 What the deposited checkpoint actually produces (8 seeds × 20,000 samples)

| Source | Valid / 20,000 | Validity | Unique valid | Novel unique | Mean pairwise dist. |
|---|---|---|---|---|---|
| Manuscript (as printed; read from file, not computed here) | 6,577 | 0.329 | 71 | 69 | 0.278 |
| Deposited metrics JSON (read from file) | 6,768 | 0.3384 | 66 | 64 | 0.2789 |
| **Rerun, jitter 0.08 (script default)** | 4,458 ± 43 | **0.2229 ± 0.0021** | **7.1 ± 0.8** | 7.1 ± 0.8 | 0.2183 ± 0.0049 |
| **Rerun, jitter 0.30 (archived setting)** | 6,608 ± 28 | **0.3304 ± 0.0014** | **69.1 ± 2.0** | 68.1 ± 2.0 | 0.2782 ± 0.0014 |

Run-to-run variation at the archived setting is small: validity ± 0.14 pp, unique valid
66–72, novel unique 65–71, distance 0.2757–0.2793. **The number the authors can stand
behind is validity = 33.0 % ± 0.1 %, 69 ± 2 unique valid formulas, 68 ± 2 novel,
mean pairwise distance 0.278 ± 0.001, over 8 seeds at jitter 0.30.** The archived
33.84 % sits slightly above that band and the manuscript's 32.9 % slightly below;
both are single runs.

LaTeX: `a2_table1_reproduction.tex`. Data: `a2_reproduce_seeds.csv`,
`a2_reproduce_summary.json`, `a2_seed_forensics.json`.

---

## 2. Random element-selection baseline — the decisive experiment (`a2_exp2_random_baseline.py`)

**Design.** Each control keeps the CVAE's own decoded fraction profile for a sample
(so the number of selected cations, the sorted cation-fraction profile and the decoded
oxygen fraction are held *identical* to the CVAE's) and randomises only **which element
carries each fraction**. Vectors then go through the same `decode_composition_vector`.
The `rand_dirichlet41` arm uses no CVAE information at all. 5 seeds × 20,000 samples.

| Generator | Validity | Unique valid | Novel unique | Uniqueness | Mean pairwise dist. |
|---|---|---|---|---|---|
| Composition-CVAE (deposited checkpoint) | 0.3302 ± 0.0018 | **69** | 68 | 0.0104 | 0.2785 |
| Random: uniform over 41-element whitelist | **0.7274 ± 0.0017** | **11,821** | 11,793 | 0.8126 | 0.3437 |
| Random: training-frequency weighted, whitelist | 0.6606 ± 0.0028 | 8,850 | 8,791 | 0.6698 | 0.3457 |
| Random: whitelist + random fractions (no model input) | 0.6778 ± 0.0042 | 11,636 | 11,390 | 0.8584 | 0.3755 |
| Random: uniform over 77-element vocabulary | 0.1009 ± 0.0018 | 1,964 | 1,959 | 0.9732 | 0.3449 |
| Random: training-frequency weighted, full vocabulary | 0.0446 ± 0.0017 | 865 | 857 | 0.9688 | 0.3469 |

**Reading (interpretation).**
* On the paper's own headline metrics the CVAE is **worse than random**: 2.2× lower
  validity, 171× fewer unique valid formulas, 19 % lower mean pairwise distance than
  uniform whitelist sampling. There is no metric in the manuscript's Table on which
  the CVAE beats a whitelist-restricted random control.
* The one thing the model does beat is the *naive* prior: sampling elements by training
  frequency over the full 77-element vocabulary yields 4.5 % validity, and uniform over
  the full vocabulary 10.1 %, against the CVAE's 33.0 %. So the model has learned
  something — it has learned to emit a small set of well-behaved ceramic cations. But
  the identical benefit is obtained for free by restricting the sampler to the whitelist
  the code already contains, and the whitelist-restricted sampler does not pay for it
  with a 171× loss of diversity.
* The CVAE's failure mode is charge balance, not exotic elements: 66.5 % of CVAE
  samples fail "no charge-balanced oxygen count found" versus 27.3 % for uniform
  whitelist sampling. Its single most frequent decoded formula overall, Al₂ScZrO₄
  (23 % of all 20,000 samples), is itself not charge-balanceable and is discarded.

LaTeX: `a2_table2_random_baseline.tex`. Data: `a2_random_baseline_runs.csv`,
`a2_random_baseline_summary.{csv,json}`, `a2_baseline_<arm>_seed123.csv`.

---

## 3. Unconditional / condition-destroyed controls (`a2_exp3_unconditional.py`)

Same checkpoint, same decoder, **same latent draw**; only the condition vector changes.
5 seeds × 20,000 samples.

| Condition signal | Validity | Unique valid | Distinct formulas (all samples) | Modal share | Mean pairwise dist. |
|---|---|---|---|---|---|
| Conditioned, archived path | 0.3302 | 69.0 | 305.0 | 0.235 | 0.2785 |
| Conditions shuffled across samples | 0.3299 | 73.2 | 295.2 | 0.236 | 0.2777 |
| Conditions ~ U(0,1)² | 0.3749 | 76.4 | 497.6 | 0.057 | 0.2762 |
| Condition held at (0.5, 0.5) | 0.9887 | 6.0 | 8.4 | 0.911 | 0.1624 |
| Condition held at (0.9, 0.8) — the paper's target | **0.0713** | **1.0** | 3.8 | 0.912 | 0.0000 |
| Condition held at the training-set mean (0.654, 0.040) | **0.0000** | **0.0** | 27.6 | 0.496 | 0.0000 |

**Reading (interpretation).** Shuffling the conditions changes nothing, which is
expected (the conditions are i.i.d. and independent of z) and is therefore a weak test.
The informative control is holding the condition **constant**: at the paper's own
nominal target (0.9, 0.8), 20,000 independent latent draws collapse to **one** unique
valid formula and 7.1 % validity. The 69 unique formulas reported in the paper are
therefore not generated by the latent space — they are generated by the *spread of the
random condition draw*. The generator is, operationally, a lookup table from the 2-dim
condition to a formula.

Note also that the model is degenerate exactly at the mean of its own training
conditions: 0 % validity.

LaTeX: `a2_table3_unconditional.tex`. Data: `a2_unconditional_runs.csv`,
`a2_unconditional_summary.json`.

---

## 4. Does conditioning do anything at all? (`a2_exp4_condition_sweep.py`)

Grid of U_stab ∈ {0.2, 0.5, 0.8, 1.0} × U_gap ∈ {0.5, 0.8, 1.0}, 4,000 samples per
cell, with a **single shared fixed latent draw** across all 12 cells, so every
between-cell difference is attributable to the condition alone.

**Effect sizes (computed).**
* `I(cell; decoded canonical formula)` = **3.045 bits**; label-permutation null
  0.0122 ± 0.0005 bits; maximum possible for 12 cells 3.585 bits; normalised MI 0.77.
* `I(cell; decoded element)` = **1.775 bits** (null 0.0008); normalised MI 0.50.
* Mean pairwise Jensen–Shannon distance between per-cell element-frequency
  distributions **0.817** (max 1.0).
* Mean Jaccard overlap of decoded formula sets between cells **0.034** (min 0.0).
* Paired formula-switch rate (same z, different condition): mean **0.959**, min 0.106.
* Largest mean decoded element-fraction ranges across cells: O 0.182, Al 0.112,
  Nb 0.087, Ta 0.080, Sr 0.073, Zr 0.063.

| U_stab | U_gap | Validity | Distinct formulas / 4,000 | Modal share | Modal decoded formula |
|---|---|---|---|---|---|
| 0.2 | 0.5 | 0.006 | 5 | 0.929 | CsNb₂Ta₂O₄ |
| 0.2 | 0.8 | 0.009 | 11 | 0.964 | BaSr₂TaO₄ |
| 0.2 | 1.0 | 0.999 | 2 | 0.999 | BaCaSr₂O₄ |
| 0.5 | 0.5 | 0.990 | 7 | 0.917 | NbTaTiO₇ |
| 0.5 | 0.8 | 1.000 | 4 | 0.863 | CaSrZr₂O₆ |
| 0.5 | 1.0 | 0.971 | 4 | 0.910 | CaSrZr₂O₆ |
| 0.8 | 0.5 | 0.764 | 9 | 0.360 | AlScZrO₅ |
| 0.8 | 0.8 | 0.746 | 6 | 0.746 | Al₂HfZrO₇ |
| 0.8 | 1.0 | 0.993 | 4 | 0.993 | Al₂HfZrO₇ |
| 1.0 | 0.5 | 0.035 | 27 | 0.639 | **AcBaPaO₄** |
| 1.0 | 0.8 | 0.107 | 12 | 0.286 | Al₂HfScO₄ |
| 1.0 | 1.0 | 0.191 | 5 | 0.800 | Al₂ScZrO₄ |

**Reading (interpretation).** This is *not* a null result: the condition vector almost
entirely determines the decoded formula. But three things follow that the authors must
say plainly.

* The conditioning is **degenerate**, not distributional. Each cell is essentially a
  point mass (modal share 0.29–0.999 over 4,000 latent draws). A conditional generative
  model should give a *distribution* of compositions per condition; this gives one.
* The response is **non-monotone and chemically incoherent**. Validity swings from
  0.006 at (0.2, 0.5) to 0.999 at (0.2, 1.0) and back to 0.035 at (1.0, 0.5). The
  highest requested stability (U_stab = 1.0) produces the model's *worst* outputs,
  including a modal formula containing actinium and protactinium.
* Crucially, **none of this is evidence that the conditioning targets stability or
  band gap**. The grid shows c → composition is informative; it says nothing about
  whether the decoded composition actually has the requested properties. That
  requires an independent property evaluation of the generated formulas (agent a5 /
  uMLIP work), and until that exists the paper cannot claim property-targeted design.
  The existing "target-proximity score" cannot supply it — as established in
  the audit brief and confirmed here, it is `0.5*c₀ + 0.5*c₁`, a function of the
  random condition draw only.

LaTeX: `a2_table4_condition_sweep.tex`. Data: `a2_condition_sweep_summary.json`,
`a2_condition_sweep_cells.csv`, `a2_condition_sweep_{js_matrix,jaccard,element_frequency,mean_element_fractions}.csv`,
`a2_condition_sweep_samples.csv.gz`.

---

## 5. Diversity, mode concentration, and the true cause (`a2_exp5_diversity.py`, `a2_exp6_latent_diagnostics.py`)

### 5.1 Sweeps (3 seeds × 20,000 samples per setting)

| Sweep | Setting | Validity | Unique valid | Unique per 1,000 valid | Element entropy (bits) | Stoich. classes |
|---|---|---|---|---|---|---|
| Latent σ | 0.08 | 0.2631 | 43.3 | 8.2 | 3.12 | 12.0 |
| Latent σ | 0.3 | 0.2696 | 47.0 | 8.7 | 3.15 | 12.0 |
| Latent σ | 0.5 | 0.2811 | 51.0 | 9.1 | 3.17 | 12.0 |
| Latent σ | 1.0 (as published) | 0.3298 | 69.3 | 10.5 | 3.16 | 12.0 |
| Latent σ | 2.0 | 0.4094 | 130.0 | 15.9 | 3.06 | 18.7 |
| Condition jitter | 0.08 | 0.2215 | 7.7 | 1.7 | 1.85 | 4.0 |
| Condition jitter | 0.3 (as published) | 0.3298 | 69.3 | 10.5 | 3.16 | 12.0 |
| Condition jitter | 0.5 | 0.3084 | 71.0 | 11.5 | 3.30 | 12.3 |
| Condition jitter | 1.0 | 0.1996 | 60.0 | 15.0 | 3.18 | 12.7 |

Uniqueness per 1,000 **valid** samples is 10.5 for the CVAE at the published settings;
for uniform whitelist random selection it is 813. Element entropy over the CVAE's valid
outputs is 3.16 bits across only **15 distinct cations** (Zr 4,931; Al 3,747; Hf 3,395;
Ca 1,733; Sr 1,483; Sc 1,188; Ti 912; …); uniform random selection uses all 40.
Doubling latent σ to 2.0 nearly doubles unique formulas (130) — off-manifold sampling
recovers diversity, which is itself a collapse signature.

### 5.2 Decoder reachability — the decoder is *not* the bottleneck

Enumerating everything `decode_composition_vector` can emit (all 1-, 2- and 3-cation
subsets of the 40 whitelist cations × every gcd-reduced count pattern the rounding step
can actually produce — {1}, {1,1},{3,1},{4,1}, {1,1,1},{2,1,1},{2,2,1},{3,1,1},{4,1,1} —
× oxygen 1–12 with charge balance):

* **129,205** distinct charge-balanced, whitelist-clean formulas are reachable.
* The CVAE reaches **68** (0.053 %) in 20,000 samples.
* Uniform random element selection through the *same* decoder reaches **11,744** (9.1 %).
* Sanity check: zero CVAE outputs fall outside the enumerated set, so the enumeration
  is complete.

### 5.3 Posterior collapse, measured directly

Encoding all 25,148 training records with the deposited checkpoint:

* **31 of 32 latent dimensions have exactly zero KL to the prior.**
* The one surviving dimension carries 0.1744 nats; total KL = **0.175 nats = 0.25 bits**.
* Training used `beta ≤ 0.1` against a *sum-reduced* reconstruction loss and a
  `free_bits = 0.5 × latent_dim = 16 nats` clamp on the **total** KL, i.e. the objective
  stopped penalising KL long before 0.175 nats — the collapse is not caused by the KL
  term being too strong; the latent simply was never needed.

**Reading (interpretation) — and a correction to the working hypothesis.** the audit brief
item 7 proposed that the ~66 unique formulas are structurally forced by the top-3-cation
/ max-count-4 decoder. **The data do not support that.** The decoder's reachable set is
six orders of magnitude larger than what the model produces, and a random sampler
through the identical decoder covers 9 % of it. The correct diagnosis is *complete
posterior collapse*: a 32-dimensional latent carrying a quarter of a bit, and a decoder
that is a near-deterministic function of the 2-dimensional condition. This matters for
the revision because the two diagnoses call for different fixes — a coarse decoder would
be fixed by a finer stoichiometry decoder; posterior collapse requires a different
objective (e.g. KL annealing that actually binds, per-dimension free bits, a
mean-reduced reconstruction term, or a discrete/autoregressive composition decoder).

LaTeX: `a2_table5_diversity.tex`. Data: `a2_diversity_sweeps.csv`,
`a2_decoder_reachability.json`, `a2_latent_diagnostics.json`.

---

## 6. A filter-accounting problem the reviewers have not yet raised (`a2_whitelist_masking.py`)

In the reproduced 20,000-sample run, only **108** candidates are reported with status
"risk elements present". But **3,304 samples (16.5 %)** contain at least one element
outside the 41-element whitelist — Ac (2,184 occurrences), Pa (1,646), Eu (820), V (174),
F (134), N (133), Np (129), S (117), Cs (79), Mn, Tl, Rb, Cr, Hg. 96.7 % of them are
booked as "no charge-balanced oxygen count found" because `charge_balance_possible`
returns `False` for any element missing from `OXIDATION_STATES` and runs first. Any
statement in the paper of the form "only 114 candidates were removed for containing
risky elements" therefore understates, by a factor of ~30, how often the generator
proposes elements the authors consider unsuitable.

---

## 7. What I would warn the authors about

1. **Do not publish the current Table without the random baseline.** Any reviewer who
   spends an afternoon will find that uniform sampling from `COMMON_CERAMIC_ELEMENTS`
   beats the model on every reported metric. It is far better to report this yourselves,
   reframe validity as a property of the post-filter, and state what the model adds
   (concentration on a small set of ceramic cations) and what it costs (171× diversity).
2. **Fix the methods text on the generation settings.** The deposited script's default
   (`--condition-jitter 0.08`) does not reproduce any published number. State
   `--seed 123 --condition-jitter 0.30` and report the 8-seed band.
3. **Report per-sample irreproducibility honestly.** The archived CSV cannot be
   regenerated sample-for-sample from the deposited checkpoint on CPU (54.7 % formula
   agreement) because `torch.randn` was called on the GPU. Either deposit the latent
   draw or state that only aggregate metrics reproduce.
4. **Drop or redefine the "target-proximity score".** Confirmed again here: it is the
   mean of the two random condition draws. Table 11's ranking is a ranking of random
   numbers, and Al₂ZrHfO₇ "scoring 1.000" means only that both of its condition draws
   clipped at 1.0.
5. **Do not describe the low uniqueness as an artefact of the decoder.** It is
   posterior collapse, and it is measurable (31/32 dead dimensions). Saying "coarse
   decoder" in the rebuttal would be falsifiable in one command.
6. **"Novelty ≈ 97 %" is not a strength.** Random element selection attains 99.8 %
   novelty on the same test, because the test is only "canonical formula not in the
   25,148-row known table". Novelty at this definition measures nothing about the model.
7. **Claims of property-targeted design are still unsupported.** Conditioning does
   change the decoded chemistry (3.04 of 3.58 bits), but nothing here shows the changed
   chemistry has the requested stability or gap, and the model's response to the
   condition is non-monotone — the maximum-stability condition yields an Ac/Pa oxide.
   Link this to an independent property evaluation before claiming inverse design.

---

## Files

Scripts (`analysis/a2_generative/`): `a2_common.py`, `a2_seed_forensics.py`,
`a2_exp1_reproduce.py`, `a2_exp2_random_baseline.py`, `a2_exp3_unconditional.py`,
`a2_exp4_condition_sweep.py`, `a2_exp5_diversity.py`, `a2_exp6_latent_diagnostics.py`,
`a2_whitelist_masking.py`, `a2_make_tables.py`.

One deviation from the project code is documented in `a2_common.safe_mean_pairwise_distance`:
`generate_composition_candidates.mean_pairwise_distance` materialises every pair before
subsampling and exhausts memory once a baseline arm yields thousands of unique formulas.
The replacement returns the project function's exact value whenever n(n−1)/2 ≤ 5,000
(which covers every CVAE run reported here) and an unbiased 5,000-pair estimate above that.

Results (`results/`): `a2_table1..5_*.tex` (booktabs bodies),
`a2_reproduce_*`, `a2_seed_forensics.json`, `a2_random_baseline_*`, `a2_baseline_*_seed123.csv`,
`a2_unconditional_*`, `a2_condition_sweep_*`, `a2_diversity_sweeps.csv`,
`a2_decoder_reachability.json`, `a2_latent_diagnostics.json`, `a2_whitelist_masking.json`.
