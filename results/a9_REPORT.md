# a9: Four-arm comparison of composition-generation strategies

Analysis a9. Scripts are in `analysis/a9_fourarm/` and results in `results/a9_*`. Every number below was computed in this session by the scripts listed in §7, unless it is marked "read from file". Energetics (uMLIP) for arms B, C and D are run by a separate agent from the hand-off files in §5. They are not part of this report.

## What each step up the knowledge ladder buys

**On counts, no step beats random selection, and the steps get progressively worse.** Each step up the ladder lowers validity: 0.729 for random (A), 0.662 for element statistics (B), 0.579 for ion substitution (C) and 0.442 for the corrected CVAE (D). Unique valid formulas per 20,000 proposals go 11,849 → 8,857 → 681 → 1,112. Ion substitution is the least diverse arm by a wide margin. It has 0.059 uniqueness and 162 substitution families per run. Only 43 % of its unique valid formulas are absent from the training table, and 77 % of its valid draws are training-table compounds. The learned model recovers some diversity relative to substitution (1.6× the unique formulas, 96 % novel), but it still has 10.7× fewer unique valid formulas than random and the lowest validity of all four arms.

**On chemistry, every step helps, and the learned model helps most.** Open-shell d/f content falls from 81.8 % (A) to 70.7 % (B), 56.9 % (C) and 21.4 % (D). The wide-gap gate pass rate rises from 16.0 % to 25.2 %, 39.5 % and 67.5 %. Hf/Zr/Al content is flat until D (17.9, 23.0, 17.4 → 57.5 %). The paper's one positive CVAE finding, chemical plausibility, therefore survives on the corrected model: 21.4 % open-shell versus 81.8 % for random (Δ −60.4 pp [−62.2, −58.6]), compared with 27.3 % vs 82.1 % on the buggy model. It also survives against the substitution arm: Δ −35.5 pp [−39.3, −31.5], and −24.6 pp [−30.0, −19.2] when only novel formulas are compared. All Fisher p are ≤ 1×10⁻⁹. The exception is the reduced-cation defect, which gets worse at the last step. 16.7 % of D's formulas need Ti³⁺/Sn²⁺ to charge-balance, against 9.1 % (A), 12.0 % (B) and 5.8 % (C). That makes D the worst arm on this axis, although it improves on the buggy model's 36.4 %.

**On surrogate-predicted properties, the ladder buys almost nothing, and the element-statistics step makes things slightly worse.** Every pairwise median effect is negligible to small (|Cliff's δ| ≤ 0.19 on band gap and composite score). B is worse than A on band gap (δ −0.19) and composite score (δ −0.14). D is indistinguishable from A on band gap (δ −0.02 [−0.04, +0.01]) and composite (δ −0.03). D's predicted permittivity is lower (δ −0.17) and its E_hull-like stability is slightly better (δ −0.07, negligible).

At matched budget, D's top candidates beat equal-sized random draws (size-controlled top-N, composite score: p = 0.004 at N = 5, < 0.0005 at N = 20), and this survives when only novel formulas are compared (p = 0.11 / 0.002). Substitution's even stronger top-N result (p < 0.0005) is entirely carried by known compounds, which make up 19 of its top 20. Among novel formulas it vanishes (p = 0.59 / 0.99).

The honest summary is that substitution and learning both act as chemical-plausibility priors, and the learned prior is much the stronger of the two. Neither is better than random on counts. Neither moves the predicted-property distribution beyond negligible-to-small effects.

---

## 1. Arms and design

All four arms make 20,000 proposals per seed with seeds 0–4. Every proposal is converted to the 77-element fraction vector and passed through the project's `decode_composition_vector` via `a2_common.decode_batch`. Metrics come from `a2_common.score_frame`, so all four rows come from one code path (`a9_counts.py`).

| Arm | Definition | Code reused |
|---|---|---|
| A, random | a2 `rand_uniform41`. The 3 cations are drawn uniformly from the 40 whitelist cations. | `a2_exp2_random_baseline.relabel` |
| B, element statistics | a2 `rand_freq41`. Same as A, but cations ∝ training-table occurrence count. | same |
| C, ion substitution | new, see §1.1 | `a9_arm_c_substitution.py` |
| D, learned | a8 checkpoint `fb0.5_b0.01`, condition N(0.9, 0.30), N(0.8, 0.30) clipped to [0,1], z ~ N(0, I) | `a8_common.use_checkpoint`, `a2_common.cvae_decode` |

**Reproduction checks** (`a9_reproduction_check.csv`):
- Seeds 0–3 reproduce a2's `rand_uniform41`, `rand_freq41` and `rand_dirichlet41` runs exactly (valid and unique-valid counts identical).
- Seeds 0–3 also reproduce a8's `fb0.5_b0.01` generation runs exactly: valid 8,870 / 8,794 / 8,768 / 8,916 and unique valid 1,100 / 1,122 / 1,111 / 1,113.
- Seed 4 is new. Arm D over seeds 0–4 gives validity 0.4417 ± 0.0030 and 1,112 ± 8 unique valid. a8's headline, over seeds 123 and 0–3, was 0.4415 ± 0.0031 and 1,108 ± 11.

**Caveat on arms A and B (inherited from a2).** These arms are not model-free. Each sample keeps the deposited CVAE's decoded fraction profile: the number of cations, the cation-fraction pattern and the oxygen fraction. Only the identity of each element is randomised. A model-free sensitivity row, A′ (a2's `rand_dirichlet41`: Dirichlet cation fractions, uniform oxygen fraction, uniform whitelist cations), gives:
- validity 0.677, unique valid 11,619, novelty 0.979
- 3,519 families per run, against 2,030 for A, because it reaches more stoichiometries.

None of the conclusions depend on choosing A over A′.

### 1.1 Arm C design choices

These stand in for the ion-exchange baseline of Szymanski & Bartel (Materials Horizons 2025). The exact rules are in the docstring of `a9_arm_c_substitution.py`.

1. **Parents.** Parents are the distinct canonical formulas of the project's own training table (12,150 distinct of 25,148 records) that are pure oxides. Excluded:
   - 8,268 formulas containing N, F or S. These are the Castelli oxynitride, oxyfluoride and oxysulfide rows, and pymatgen sometimes guesses N as a cation, so they are excluded by element.
   - 854 formulas with no pymatgen oxidation-state guess.
   - 103 formulas with a species missing from the Hautier λ table.
   - 721 formulas with no admissible substitution (C4/C5 below).
   - 4 others.

   2,200 parents remain: 928 Castelli, 750 Wolverton and 522 double perovskites. By number of distinct cations: 1 cation 46, 2 cations 1,632, 3 cations 169, 4 cations 353. Note that "known" means present in this DFT training table. Many Castelli entries are hypothetical cubic perovskites.
2. **Oxidation states.** Assigned by pymatgen `Composition.oxi_state_guesses()` (ICSD-frequency ranked), taking the first guess.
3. **Substitution model.** pymatgen `SubstitutionProbability` (Hautier et al. 2011, default λ table, α = −5). The probability that parent ion s is replaced by t is `cond_prob(t, s)`. Keeping s unchanged is allowed. The joint probability of a tuple is the product over the parent's cation species, which is how `SubstitutionPredictor` scores a species list.
4. **Replacement ions.** Restricted to positive-valence species of the same 40 whitelist cations that arms A and B use (94 species). A parent ion may always stay unchanged, even if it is not on the whitelist; the filter then rejects it. Oxygen is never substituted.
5. **Threshold (C4).** Tuples are kept if their joint probability is ≥ 1×10⁻³, the `SubstitutionPredictor` default. Following pymatgen, tuples that map two sites onto the same species are dropped. The all-identity tuple is also dropped.
6. **Charge (C5).** The anion framework is kept, so Σ nᵢ q(tᵢ) = 2 n_O with the parent's own oxygen count. Aliovalent substitutions are allowed only in compensating pairs.
7. **Number of substitutions.** Not fixed. It follows from the Hautier joint model after the rules above. The realised mean is 1.26 sites per proposal (seed 0: 14,764 one-site, 5,225 two-site and 11 three-site out of 20,000).
8. **Sampling.** The parent is drawn uniformly with replacement. The tuple is drawn ∝ joint probability among that parent's admissible tuples (median 19 per parent).
9. **Filter.** The substituted formula goes through `composition_vector` and then `decode_composition_vector`. A formula with an element outside the 77-element vocabulary would be booked invalid rather than silently truncated. This occurred 0 times.

**Consequence of the shared filter for arm C.** The filter keeps only the top 3 cations and requantises counts to at most 4. Overall, 79 % of C's valid draws keep exactly the substituted formula. For 4-cation double-perovskite parents this figure is about 0 %: one cation is always dropped, and with equal fractions the dropped cation is decided by alphabetical order. Only 0.19 % of valid draws collapse back to their own parent. This is a property of the shared filter, applied identically to all arms, but it does handicap the substitution arm on 4-cation parents.

## 2. Counts (`a9_table_counts.tex`, `a9_counts_summary.{csv,json}`, `a9_counts_runs.csv`)

Values are means ± s.d. over seeds 0–4. Families are a3's `family_key`: the same anonymised stoichiometry and the same cation periodic-group multiset.

| Arm | Validity | Unique valid | Uniqueness | Novelty | Mean pairwise dist. | Families/run | Pooled unique valid | Pooled novel | Pooled families |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.7287 ± 0.0018 | 11,849 ± 36 | 0.813 | 0.998 | 0.344 | 2,030 | 31,171 | 31,104 | 3,100 |
| B | 0.6623 ± 0.0021 | 8,857 ± 54 | 0.669 | 0.993 | 0.346 | 2,224 | 20,034 | 19,959 | 3,100 |
| C | 0.5786 ± 0.0023 | 681 ± 6 | 0.059 | 0.429 | 0.351 | 162 | 765 | 364 | 175 |
| D | 0.4417 ± 0.0030 | 1,112 ± 8 | 0.126 | 0.958 | 0.323 | 332 | 2,135 | 2,081 | 487 |
| A′ (Dirichlet) | 0.6767 ± 0.0034 | 11,619 ± 80 | 0.858 | 0.979 | 0.375 | 3,519 | 41,210 | 40,831 | 6,816 |

- **Pooled families.** A and B both reach 3,100 pooled families, but not the same ones: 2,676 are shared. Both saturate the families reachable under the deposited CVAE's fraction profiles.
- **Why arm C is so narrow.** In seed 0, its 20,000 proposals contain only 1,865 distinct substituted formulas before the filter. The Hautier probabilities send many parents to the same few products, such as BaSnO₃, SrSnO₃, BaZrO₃ and BaTiO₃.
- **Arm C invalidity.** 11.8 % of C's proposals are rejected for non-whitelist elements (unsubstituted parent ions), and 30.4 % fail charge balance after requantisation.
- **Arm D invalidity.** D's invalidity is mostly charge balance (52.3 %).
- **Novelty for arm C** is low, as expected. Substitution into a dense ABO₃ table mostly lands on other entries of the same table.

## 3. Chemistry (`a9_table_chemistry.tex`, `a9_compare_proportions.csv`, `a9_chemical_sanity.csv`)

Each arm's pooled unique valid set was annotated with `a7_chem_sanity.annotate` and `nn_distance`, unchanged. All formulas had an assignable charge balance. Flags come from `a7_compare.load`, unchanged. The gate requires predicted E_g ≥ 1 eV, no open-shell cation and no forced Ti³⁺/Sn²⁺. Statistics come from `a7_stats`: Wilson/Newcombe 95 % CI and Fisher's exact test.

| Arm | n | Open-shell % | Ti³⁺/Sn²⁺ % | Hf/Zr/Al % | Gate % |
|---|---|---|---|---|---|
| A | 31,171 | 81.8 | 9.1 | 17.9 | 16.0 |
| B | 20,034 | 70.7 | 12.0 | 23.0 | 25.2 |
| C | 765 | 56.9 | 5.8 | 17.4 | 39.5 |
| D | 2,135 | 21.4 | 16.7 | 57.5 | 67.5 |

D − X differences in percentage points [Newcombe 95 % CI]:

| Comparison | Open-shell | Ti³⁺/Sn²⁺ | Hf/Zr/Al | Gate |
|---|---|---|---|---|
| D vs A | −60.4 [−62.2, −58.6] | +7.6 [+6.1, +9.3] | +39.6 [+37.4, +41.7] | +51.6 [+49.5, +53.6] |
| D vs B | −49.3 [−51.1, −47.4] | +4.7 [+3.1, +6.4] | +34.5 [+32.3, +36.7] | +42.3 [+40.2, +44.4] |
| D vs C | −35.5 [−39.3, −31.5] | +11.0 [+8.5, +13.2] | +40.1 [+36.6, +43.4] | +28.1 [+24.0, +32.0] |

- **Novel formulas only.** D vs C becomes −24.6 pp open-shell, +12.4 pp Ti³⁺/Sn²⁺ and +19.5 pp gate (C novel n = 364). D vs A and D vs B change by ≤ 0.3 pp.
- **Ladder steps.** B vs A on open-shell: −11.2 pp. C vs B: −13.8 pp. Both p < 1×10⁻¹⁴.
- **Element enrichment** (`a9_element_enrichment.csv`). D's odds ratios against A are Zr 4.29, Al 5.28 and Hf 3.29 (Fisher p ≤ 2×10⁻⁶⁷). Against C they are Zr 3.84, Al 4.83 and Hf 5.67.
- **Where D's open-shell advantage comes from.** D almost never emits Cu, Co, Ni, Zn, Ce, Pr or Nd (≤ 0.3 % each). Arms A and B emit each of these in 3–12 % of their formulas.
- **Distance to training data.** D sits slightly closer to the training data than A (median nearest-neighbour distance 0.141 vs 0.151; δ −0.31). C's median is 0, because most of its outputs are training compounds.

## 4. Surrogate predictions (`a9_table_surrogates.tex`, `a9_compare_continuous.csv`, `a9_steelman_size_controlled.csv`)

**Retraining.** The a5 models were retrained with the unmodified a5 scripts (`a9_retrain_a5.sh`), pinned to one core.
- The scripts rewrite a5's own metric files, so they were run against a shadow root with a separate `results/`, and the committed a5 results are untouched.
- The retrained models are close to a5 but not bit-identical:
  - The largest test-metric difference across all models is 0.0015 in R², and 0.0037 in RMSE (dielectric models, linear space) (`a9_a5_retrain_check.csv`).
  - On a7's 66 CVAE candidates, predictions correlate with a7's stored ones at Spearman 0.982–0.999. The largest band-gap difference is 0.054 eV (`a9_a5_retrain_check_a7_cvae_predictions.csv`).
- Predictions come from `a7_surrogates.predict`, unchanged, on all 54,105 pooled formulas (all featurised). The composite score uses a7's formula, normalised over the four arms pooled.

Medians:

| Arm | Ê_g (eV) | ε̂₀ | Ê_hull-like (eV/at) | Composite |
|---|---|---|---|---|
| A | 1.68 | 17.6 | 0.548 | 0.520 |
| B | 1.54 | 16.8 | 0.549 | 0.498 |
| C | 1.72 | 16.6 | 0.612 | 0.525 |
| D | 1.67 | 16.7 | 0.507 | 0.507 |

Cliff's δ, D vs X [bootstrap 95 % CI]:

| Comparison | Ê_g | ε̂₀ | Ê_hull-like (lower is better) | Composite |
|---|---|---|---|---|
| D vs A | −0.02 [−0.04, +0.01] | −0.17 [−0.20, −0.15] | −0.07 [−0.10, −0.04] | −0.03 [−0.06, −0.00] |
| D vs B | +0.16 [+0.14, +0.19] | −0.03 [−0.05, −0.01] | −0.07 [−0.10, −0.04] | +0.10 [+0.07, +0.13] |
| D vs C | −0.09 [−0.14, −0.03] | +0.00 [−0.05, +0.05] | −0.07 [−0.12, −0.02] | −0.02 [−0.07, +0.03] |

- **Novel formulas only, D vs C.** D is better on composite (+0.22), band gap (+0.16) and permittivity (+0.28). Arm C's novel formulas score lower than its known ones (median composite 0.471).
- **Electronic permittivity.** D is lower than every other arm (δ −0.33 to −0.43). This is the largest surrogate effect, and it runs against D.
- **Formation-energy head.** The Castelli E_form-like head favours D (δ −0.16 vs A). a7 found that this head is not corroborated by the Wolverton head or by uMLIP, so it should not be relied on.

**Size-controlled steelman** (a7's procedure). The pool is arm A. Budget = the compared arm's pooled unique count. 2,000 draws with replacement. The table gives p(A draw ≥ arm) on the composite score:

| Arm (budget) | N = 1 | N = 5 | N = 10 | N = 20 | Novel only, N = 5 / 20 |
|---|---|---|---|---|---|
| B (20,034) | 0.461 | 0.298 | 0.169 | 0.106 | 0.302 / 0.086 |
| C (765) | < 0.0005 | < 0.0005 | < 0.0005 | < 0.0005 | 0.588 / 0.993 |
| D (2,135) | 0.068 | 0.004 | < 0.0005 | < 0.0005 | 0.112 / 0.002 |

- **Band-gap axis.** D's advantage at equal budget is weaker: p = 0.34 at N = 5 and 0.021 at N = 20.
- **E_hull-like axis.** D's top-N is not better than random (p = 0.88–0.97).
- **Arm C.** The C rows should not be quoted without the novel-only column. The surrogates were fitted on the training table, so their predictions for C's known compounds are in-sample.
- **Budget premise (a7's caveat applies).** Per 20,000 proposals, A delivers 10.7× more unique valid formulas than D.

## 5. Hand-off for energetics (`a9_umlip_sample_{C,D}.csv`, READY flag written)

The selection follows a7's rule:
- **Eligible** formulas are those whose cations all lie in {Al Ga Sc Y La Zr Hf Ti Sn Si Ba Sr Ca Mg}.
- **Draw:** `default_rng(20260913).choice(min(50, n), replace=False)`, keeping the original order.
- **Pool:** each arm's pooled unique valid set (seeds 0–4).

a7 drew from a single 20,000-sample run. Pooling across seeds is a deviation from that, stated here.

| Arm | Unique valid | Eligible | Sampled | Sampled that are training compounds |
|---|---|---|---|---|
| C | 765 | 154 (20.1 %) | 50 | 23 |
| D | 2,135 | 839 (39.3 %) | 50 | 2 |

**Flags for the energetics agent:**
- The arm-C sample contains one binary oxide, ZrO₂ (row `O2Zr`). Its decomposition energy into binary oxides is trivially about 0 and would inflate "% ≤ 0". Arm C's pooled set has 13 single-cation formulas (0 in A, B and D).
- 23 of the 50 arm-C formulas are training-table compounds. Their stability says nothing about generation.

The eligible lists are in `a9_umlip_eligible_{C,D}.csv`, and `a9_umlip_sample_summary.json` holds the counts.

## 6. Limitations

- Arm C is one simple, documented operationalisation. It is not Szymanski & Bartel's exact pipeline: parents come from this training table rather than Materials Project, and there is no structure-level substitution. Its counts are also shaped by the shared top-3, max-count-4 filter (§1.1).
- Only one threshold (1×10⁻³) was run. Sensitivity to the threshold and to the parent pool was not tested.
- Surrogate predictions are composition-only. a5 judged the dielectric model fit only to reject candidates, not to rank finely, and the stability heads have test R² of 0.07–0.12.
- Novelty is exact formula matching against the training table only. External databases are blocked in this environment, as in a2/a3/a8.

## 7. Files

Scripts in `analysis/a9_fourarm/`, in run order:
- `a9_common.py` (paths; a3 `family_key` loaded without running a3's module code)
- `a9_arm_c_substitution.py` (writes the parent cache `a9_arm_c_parents.json`)
- `a9_counts.py`
- `a9_umlip_sample.py`
- `a9_chemistry.py`
- `a9_retrain_a5.sh`
- `a9_check_a5_retrain.py`
- `a9_surrogates.py`
- `a9_compare.py`
- `a9_tables.py`

Logs: `a9_counts.log`, `a9_retrain_a5.log`, `a9_surrogates.log`, `a9_compare.log`.

The other `a9_energetics_*` files in that directory and in `results/` belong to the energetics agent.

Results in `results/`:
- Counts: `a9_counts_runs.csv`, `a9_counts_summary.{csv,json}`, `a9_reproduction_check.csv`, `a9_population_{A,B,C,D}.csv`, `a9_arm_c_proposals_seed0.csv`
- Chemistry and surrogates: `a9_chemical_sanity.csv`, `a9_surrogate_predictions.csv`, `a9_merged_candidates.csv`, `a9_compare_{continuous,proportions}.csv`, `a9_compare_summary.json`, `a9_element_enrichment.csv`, `a9_steelman_size_controlled.csv`
- Surrogate retraining checks: `a9_a5_retrain_check.csv`, `a9_a5_retrain_check_a7_cvae_predictions.csv`
- Energetics hand-off: `a9_umlip_sample_{C,D}.csv`, `a9_umlip_eligible_{C,D}.csv`, `a9_umlip_sample_summary.json`, `a9_umlip_sample_READY.flag`
- Tables: `a9_table_{counts,chemistry,surrogates}.tex` (booktabs) and `a9_TABLE_{counts,chemistry,surrogates}.csv`

The retrained `.pkl` surrogate files are in `analysis/a5_surrogate/` (not for the repository: 69–313 MB).
