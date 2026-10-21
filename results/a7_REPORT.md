# a7 — Does the CVAE produce *better* candidates than random, conditional on passing the same filter?

**MTCOMM-D-26-04679 · Agent 7 (quality comparison) · all numbers computed in this session**

---

## Direct answer

**Partly, and not on the axis that matters most.** Conditional on passing the project's own
charge-balance-plus-whitelist filter, the CVAE's 66 candidates are **not better materials** than
randomly generated ones by the strongest physics-based measure available: under CHGNet, with an
identical prototype-substitution + random-symmetric-search protocol, the decomposition energy into
binary oxides is **+0.086 eV/atom (median) for the CVAE (n = 53) and +0.086 eV/atom for random
(n = 50)** — Cliff's δ = +0.04 (95 % CI −0.19 to +0.27), Mann–Whitney p = 0.73; MACE-MP-0 agrees
(δ = +0.11, p = 0.32). Worse for the model, the *fraction* of candidates that are actually stable
against decomposition into binary oxides — the only positive stability evidence this pipeline can
produce — is **9.4 % for the CVAE against 24.0 % for random** (CHGNet; MACE 7.5 % vs 24.0 %,
Fisher p = 0.029), and the single most promising composition found anywhere in this work is a
*random* one, MgScGaO₄ at ΔE_dec = −0.748 eV/atom, six times below the best CVAE candidate
(BaTiGaO₄, −0.127 eV/atom). The surrogate-predicted properties tell the same story: against a
size-matched random set the CVAE's median predicted band gap, permittivity and stability differ by
negligible-to-small effects (|δ| ≤ 0.27), and its predicted E_hull-like stability is slightly
*worse*. What the CVAE *does* do, clearly and reproducibly, is pick **more sensible chemistry**: only
27.3 % of its candidates contain an open-shell d/f cation against 82.1 % of random ones
(Δ = −54.8 pp, Fisher p = 4×10⁻²², and still −35.6 pp against a training-frequency-weighted random
control), it is enriched 8.4× in Zr, 4.2× in Al and 3.0× in Hf — precisely the hafnate / zirconate /
aluminate chemistries the Stage-1 real-data screen independently favoured — and 63.6 % of its
candidates pass a combined wide-gap-dielectric sanity gate against 15.9 % of random ones. At a
*matched generation budget* (66 candidates), the CVAE's top-5 composite predicted-property score
beats 98.9 % of equal-budget random draws (p = 0.011). **The honest summary for the manuscript is
therefore: the CVAE is a learned chemical-plausibility prior, not a materials-quality optimiser. It
concentrates the output on the right *families*, but within those families its candidates are
energetically indistinguishable from random ones, and because it explores ~180× less of the space it
delivers fewer genuinely promising compounds in absolute terms than the trivial control does.** That
is a real, defensible and publishable contribution — but it is a much weaker claim than the submitted
paper makes, and the sentence "the CVAE produces better candidates" cannot be written.

---

## 1. What was built, and how

### 1.1 Populations (all through a byte-identical filter path)

| Population | Definition | n (unique valid) |
|---|---|---|
| **CVAE** | the deposited archive `composition_cvae_generated_candidates.csv`, `valid==True`, deduplicated on canonical formula | **66** |
| **Random, uniform-41** | a2 arm `rand_uniform41`, seed 123: each sample keeps the CVAE's decoded fraction profile; only *which* whitelist element carries each fraction is randomised, then the project's own `decode_composition_vector` is applied | **11,744** |
| **Random, size-matched** | a uniform subsample of the above, n = 66, seed 20260913 | **66** |
| **Random, frequency-weighted** | a2 arm `rand_freq41`: cations drawn with probability ∝ their occurrence in the CVAE's *own* training table — the "learned element marginals only" control | **8,870** |

The random arms were *not* reimplemented: they are the a2 harness (`a2_exp2_random_baseline.py`),
which imports the project's `decode_composition_vector`, `OXIDATION_STATES`,
`COMMON_CERAMIC_ELEMENTS` and `HIGH_RISK_ELEMENTS` directly. Scripts: `a7_build_sets.py`.

Two facts worth noting before any comparison. **19 of the CVAE's 66 formulas also occur in the
11,744-formula random pool** — nearly a third of the model's entire output is reachable by chance in
a single 20,000-draw random run. And **2 of the 66 CVAE candidates are already in the training
table**, against 25 of 11,744 random ones (0.17 %).

### 1.2 Provenance of every number below

* **(a) computed here:** all decomposition energies for the random sample (374 CHGNet + 50 MACE
  relaxations run in this session), all surrogate predictions for both populations, all
  oxidation-state / open-shell assignments, all distances to training data, all statistics.
* **(b) read from existing files:** the 442 CHGNet / 67 MACE relaxations of the CVAE candidates and
  the binary-oxide reference energies (a4, not recomputed); the surrogate model objects (a5, not
  retrained); the validity/uniqueness figures quoted from the audit.
* **(c) interpretation:** clearly marked as such.

---

## 2. Energetics — the strongest evidence

### 2.1 What was run

For 50 random candidates, the a4 workflow was executed unchanged (`a7_run_energetics.py` imports
`a4_run_screen.run_candidates`, which calls `a4_hypotheses.all_hypotheses(n_random=4, seed=0)` and
`a4_relax.relax(fmax=0.10, steps=150)`): **374 CHGNet relaxations**, every candidate's hypothesis set
complete, then **50 MACE-MP-0 refinements** started from the CHGNet-relaxed geometry with
`fmax=0.10, steps=120` — a4's exact refinement policy (`a7_mace_refine.py`). Decomposition energies
were produced by calling `a4_analyze.analyze()` itself against a4's own already-relaxed binary-oxide
references. No energy from a4 was recomputed and no reference structure was invented.

**A stated constraint.** a4 could build trustworthy binary-oxide seed structures for only 14 cations
(Al Ga Sc Y La Zr Hf Ti Sn Si Ba Sr Ca Mg; Nb₂O₅ and Ta₂O₅ were excluded as not reconstructible
offline, and every structure database is proxy-blocked). Inventing seed structures for
Ce/Ni/Cu/Fe/Co/Bi/W/Mo/lanthanide oxides from memory would be exactly the kind of fabricated input the
analysis does not permit. The random sample was therefore drawn from the **331 of 11,744 random
candidates (2.8 %) whose cations all lie in that same 14-cation set**. This makes the energetics test
a *chemistry-matched* comparison — random is held to the CVAE's own element palette, which removes any
advantage the CVAE would otherwise get simply from choosing nicer elements. It is the conservative
choice, and the conclusion applies within that chemical support, not to the random population as a
whole. (`a7_energetics_sample.py`, 50 drawn with seed 20260913 from the 331.)

**A protocol asymmetry that had to be corrected.** a4's CVAE candidates received 2–16 structure
hypotheses (median 4) because textbook prototypes exist for their stoichiometries; the random ones
received 4–16 (median 6). Since ΔE_dec is a minimum over hypotheses, more hypotheses means a tighter
bound. Both a "full" and a "pyxtal-only" comparison are therefore reported, the latter restricting
both arms to the random symmetric search that *every* stoichiometry receives. A first MACE run that
started from the raw seed rather than the CHGNet geometry gave the random arm 72 % convergence
against a4's 99 % and produced a spurious "CVAE wins" result; it was discarded and is retained only as
`a7_hypothesis_relaxations_mace_SEEDSTART_discarded.csv`. After the correction, MACE convergence is
98 % (random) vs 99 % (CVAE), and the cross-potential MAD is **0.030 eV/atom for the CVAE arm and
0.030 eV/atom for the random arm** — i.e. the two arms are now equally well converged.

### 2.2 Result

ΔE_dec into binary oxides, eV/atom, lower is better. δ = Cliff's delta (CVAE − random), 95 % CI from
a 2,000-resample percentile bootstrap; p from a two-sided Mann–Whitney U test.

| Potential | Protocol | n CVAE | n rand | median CVAE | median rand | δ | 95 % CI | p | % ≤ 0 CVAE | % ≤ 0 rand |
|---|---|---|---|---|---|---|---|---|---|---|
| CHGNet | full | 53 | 50 | **+0.0861** | **+0.0864** | +0.040 | [−0.189, +0.273] | 0.73 | 9.4 | 24.0 |
| CHGNet | pyxtal-only | 53 | 50 | +0.0884 | +0.1081 | −0.084 | [−0.313, +0.145] | 0.47 | 5.7 | 16.0 |
| MACE-MP-0 | full | 53 | 50 | +0.0996 | +0.0955 | +0.115 | [−0.114, +0.338] | 0.32 | 7.5 | 24.0 |
| MACE-MP-0 | best-pyxtal subset | 41 | 36 | +0.1171 | +0.1162 | +0.070 | [−0.198, +0.329] | 0.60 | 2.4 | 16.7 |

(n CVAE is 53 not 66 because 13 CVAE candidates contain Nb or Ta, for which no binary reference
exists. The full per-candidate table is `a7_energetics_all.csv`.)

**Reading.** Under the identical, fully searched CHGNet protocol the two distributions are
indistinguishable: the median difference is −0.0004 eV/atom with a bootstrap CI of
[−0.037, +0.052] eV/atom, an order of magnitude smaller than the 0.030 eV/atom cross-potential error
and two orders below the 0.825 eV/atom structure-hypothesis spread that a4 identified as the dominant
uncertainty. The pyxtal-only comparison nudges in the CVAE's favour (δ = −0.084) but is still
negligible and far from significant. **This is a null result, and with n ≈ 50 per arm it has enough power to exclude
anything larger than a small CVAE advantage**: the CHGNet 95 % CI on δ excludes any CVAE advantage
beyond δ = −0.19 (full protocol) or δ = −0.31 (pyxtal-only), i.e. it rules out the CVAE being
*moderately* better; it does not rule out a small one, which would not matter.

**The one directional finding runs against the model.** Random produces 2.5× more candidates that are
genuinely stable against decomposition into binary oxides (12/50 vs 5/53 under CHGNet, Newcombe 95 %
CI on the difference [−29.0, −0.03] pp, Fisher p = 0.063; under MACE 12/50 vs 4/53, p = 0.029). Since
ΔE_dec ≤ 0 is the *only* affirmative stability statement this pipeline can make (a4's logic: ΔE_dec > 0
is inconclusive, ΔE_dec ≤ 0 is real evidence), this is the single most decision-relevant number in
the report, and it favours random.

The best candidates found, ranked by CHGNet ΔE_dec:

| Rank | Population | Formula | ΔE_dec (eV/atom) | best prototype |
|---|---|---|---|---|
| 1 | random | MgScGaO₄ | −0.748 | spinel Sc[GaMg]O₄ |
| 2 | random | Sr₂ZrTiO₆ | −0.178 | double perovskite |
| 3 | random | Sr₂CaSiO₅ | −0.129 | pyxtal sg14 |
| 4 | **CVAE** | BaTiGaO₄ | −0.127 | spinel Ti[BaGa]O₄ |
| 5 | random | SrTi₂SnO₅ | −0.124 | pyxtal sg4 |

*Caveat (interpretation).* MgScGaO₄ at −0.75 eV/atom is far outside the range a4 found for any CVAE
candidate and deserves independent checking before it is quoted; it is a spinel-prototype relaxation
and could reflect an unusually favourable prototype match rather than exceptional chemistry. Its
reference set (MgO, Sc₂O₃, Ga₂O₃ — the last an α-Ga₂O₃ metastable polymorph, flagged as such by a4)
is the same one used for the CVAE candidates, so the comparison is internally consistent, but the
absolute value should be confirmed by DFT.

---

## 3. Surrogate-predicted properties

The a5 model objects were loaded and applied unchanged (`a7_surrogates.py`; band gap averaged over
the three source one-hots, dielectric with `is_phonon_aux = 0`, both stability heads reported). The
composite score is a5's own 0.4·stability + 0.3·gap + 0.3·permittivity, renormalised on the **pooled**
population so the arms sit on one scale.

Medians (full table with effect sizes: `a7_compare_continuous.csv`, `a7_TABLE_overview.csv`):

| Population | n | Ê_g (eV) | ε̂₀ | Ê_hull-like (eV/at) | open-shell (%) | Ti³⁺/Sn²⁺ (%) | gate (%) | Hf/Zr/Al (%) |
|---|---|---|---|---|---|---|---|---|
| **CVAE** | 66 | **1.74** | **17.6** | 0.615 | **27.3** | 36.4 | **63.6** | **53.0** |
| Random, size-matched | 66 | 1.66 | 17.0 | 0.557 | 84.8 | 7.6 | 12.1 | 15.2 |
| Random, frequency-weighted | 8,870 | 1.49 | 16.6 | 0.557 | 62.9 | 13.0 | 31.8 | 25.8 |
| Random, uniform | 11,744 | 1.69 | 17.4 | 0.551 | 82.1 | 8.3 | 15.9 | 18.7 |

Against the **size-matched** random set the effects are negligible to small on every property axis:
band gap δ = +0.142 [−0.050, +0.350], p = 0.16; ε̂₀ δ = +0.273 [+0.081, +0.465], p = 0.007;
ε̂_∞ δ = −0.028, p = 0.78; E_hull-like δ = +0.015, p = 0.89 (positive δ here means *worse*, since
lower is better); composite δ = +0.152 [−0.050, +0.352], p = 0.13. Against the **frequency-weighted**
control the CVAE looks better — band gap δ = +0.390 (medium), composite δ = +0.266 — but that control
is itself poorer than uniform random on band gap, so this is a comparison against a weakened baseline
rather than evidence of strength.

The only *large* effect anywhere in this section is on the Castelli-style E_form-like head
(δ = −0.51 to −0.54, p ≤ 3×10⁻⁷, CVAE lower i.e. better). **Do not lean on it.** It is not
corroborated by the Wolverton-style head (δ ≈ +0.01 to +0.04, both null), it is not corroborated by
either uMLIP, and a5 itself notes that a candidate composition is not "from" either training source;
two heads that disagree this strongly on the same candidates are a sign that at least one of them is
extrapolating.

**Do not over-read the permittivity difference either.** a5's own verdict is that the dielectric
surrogate is good enough to reject bad candidates but not to rank finely. A median ε̂₀ gap of 0.6–1.1
(≈ 4–6 %) is well inside that stated limit.

---

## 4. Chemical sanity — where the CVAE actually wins

`a7_chem_sanity.py`. Oxidation-state assignment uses the project's **full** `OXIDATION_STATES` table
(a4's 16-cation table cannot assign states to the Fe/Co/Ni/Cu/W/Mo/lanthanide compositions that the
random arm produces in quantity); the implementation was verified to agree with
`a4_descriptors.reduction_requirement` on all 66 CVAE candidates, and reproduces the audit's
36.4 % (24/66) Ti³⁺/Sn²⁺ figure exactly.

| Property | CVAE | random (uniform) | Δ (pp) | 95 % CI | Fisher p |
|---|---|---|---|---|---|
| contains an open-shell d/f cation | 27.3 % | 82.1 % | **−54.8** | [−64.1, −43.0] | 4×10⁻²² |
| balance requires Ti³⁺ or Sn²⁺ | 36.4 % | 8.3 % | **+28.1** | [+17.5, +40.1] | 3×10⁻¹⁰ |
| passes combined wide-gap gate | 63.6 % | 15.9 % | **+47.7** | [+35.7, +58.3] | 4×10⁻¹⁸ |
| contains Hf, Zr or Al | 53.0 % | 18.7 % | **+34.4** | [+22.5, +45.9] | 5×10⁻¹⁰ |

(The gate = predicted E_g ≥ 1.0 eV **and** no open-shell cation **and** no forced Ti³⁺/Sn²⁺.)

**Answering the specific question asked.** The audit's 36 % Ti³⁺/Sn²⁺ defect is **a model problem, not
a filter problem**: random candidates through the identical filter show it at 8.3 % (and 13.0 % under
the frequency-weighted control). The CVAE is 4.4× more likely than random to reach charge balance only
by reducing a cation — a direct contradiction of its own wide-gap design target, and it happens
because the model concentrates on Ti- and Sn-bearing chemistries in the first place.

**But the broader open-shell picture is the reverse, and it is the larger effect.** Random draws
Ce, Ni, Pr, Cu, Yb, Tb, Fe, Co in ~1,100 candidates each; 82 % of random candidates contain a
partially-filled d or f shell, against 27 % of the CVAE's. On the electron-counting criterion — which
is exact arithmetic, not a prediction — the CVAE's population is much better suited to a wide-gap
dielectric target. This advantage survives against the frequency-weighted control (27.3 % vs 62.9 %,
Δ = −35.6 pp, p = 9×10⁻⁹), so it is **not** merely reproduced by knowing the training set's element
marginals.

**Distance to training data.** Median Euclidean distance in the project's own 77-dim element-fraction
space to the nearest of the 25,148 training compositions: CVAE 0.1446, random 0.1507
(δ = −0.228 [−0.399, −0.064], p = 0.001). The CVAE sits measurably closer to its training data, but
the shift is small and only 2 of its 66 candidates are exact training compounds. **This does not
support a "the model has merely memorised" reading** — the model is interpolating in a restricted
region, which is a different and less damaging criticism than memorisation.

---

## 5. The steelman — the fairest tests I could construct

### 5.1 Naive top-N: the CVAE loses, but the test is confounded

Taking the top-N of each population by composite predicted score, the random pool wins on every axis
at every N (e.g. top-20 band gap 3.74 vs 2.65 eV, δ = −0.66). **This is not a fair test**: the random
pool has 11,744 members, so its top-20 is selected from 178× more candidates. Reported for
completeness in `a7_steelman.csv`; it should not be quoted.

### 5.2 Size-controlled top-N: the CVAE wins

The correct version asks what would happen at the *same generation budget*. For each of 2,000 random
subsamples of size 66 from the random pool, the median composite score of the top N was computed and
the CVAE's value located in that null (`a7_steelman2.py`, `a7_steelman_size_controlled.csv`):

| Random pool | N | CVAE top-N | equal-budget random top-N (median) | p(random ≥ CVAE) |
|---|---|---|---|---|
| uniform | 1 | 0.830 | 0.773 | 0.066 |
| uniform | 5 | 0.789 | 0.717 | **0.011** |
| uniform | 10 | 0.728 | 0.686 | **0.027** |
| uniform | 20 | 0.685 | 0.653 | **0.020** |
| freq-weighted | 5 | 0.789 | 0.694 | **0.008** |
| freq-weighted | 20 | 0.685 | 0.627 | **0.0005** |
| freq-weighted | 20 (band gap) | 2.71 eV | 1.99 eV | **0.0000** |

**This is the CVAE's best result in the whole report, and it is a real one:** at matched budget, its
best candidates are better than the best of an equal number of random ones, about 98–99 times out of
100. Two honest qualifications. (i) The metric is a surrogate composite whose stability term is a
model a5 warns about and whose dielectric term a5 says cannot rank finely. (ii) The premise "at
matched budget" is exactly the premise the audit destroyed: the CVAE's budget is *not* matched in
practice, because it produces 66 unique valid formulas per 20,000 samples where random produces
11,744 (audit §A). An experimentalist choosing 20 compounds to synthesise would take the random
pool's top 20, which beats the CVAE's top 20 decisively.

### 5.3 Conditioning on chemical family

Within a stoichiometry class the CVAE does show one genuine, sizeable win: among **4-cation : 6-oxygen
(double-perovskite-like) compositions**, CVAE candidates have a median predicted band gap of 2.98 eV
against 1.95 eV for random (n = 13 vs 3,291, δ = +0.487 "large", p = 0.0024), and a higher composite
score (δ = +0.469, p = 0.0035). The other three families with enough members (3:4, 3:5, 4:7) show
negligible effects. Restricting both arms to candidates containing Hf/Zr/Al gives only a small band-gap
edge (δ = +0.197, p = 0.045).

### 5.4 Enrichment in the Stage-1-favoured chemistries

This is the clearest *positive* structural result. Odds ratios, CVAE vs uniform random
(`a7_stage1_enrichment.json`):

| Element | CVAE | random | odds ratio | Fisher p |
|---|---|---|---|---|
| Zr | 34.8 % | 6.0 % | **8.36** | 3×10⁻¹² |
| Al | 24.2 % | 7.2 % | **4.15** | 1×10⁻⁵ |
| Hf | 16.7 % | 6.3 % | **2.95** | 0.0029 |

The CVAE independently rediscovers the zirconate / aluminate / hafnate families that the Stage-1
real-data screen selected. **Interpretation:** this is the most defensible scientific claim available
about the generative stage — the model has learned *which chemistries* the screen rewards. It has not
learned to pick better members within them. Note also that the random pool's own top-scoring
candidates (BaHf₂SrO₆, BaCaHf₂O₆, HfSr₂ZrO₆, Ca₂HfZrO₆) are the same hafnate family; random finds
them too, just by sampling more.

---

## 6. What would change my conclusion, and what could not be done

* **Could not be done:** absolute stability (energy above hull). Every structure/thermochemistry
  database — Materials Project, OQMD, AFLOW, COD, Alexandria, NOMAD, Materials Cloud — is refused by
  this environment's egress policy, so the competing-phase set needed for a hull is unobtainable, as
  a4 already documented. ΔE_dec into binary oxides under one potential is the substitute, and it is a
  *lower* bound on the hull distance.
* **Could not be done:** extending the energetics comparison to the other 97.2 % of the random
  population. That would require binary-oxide reference structures for Ce, Ni, Cu, Fe, Co, Bi, W, Mo
  and the lanthanides; building those from memory would be fabrication, and the databases are blocked.
  This is the single largest limitation of §2, and it cuts *against* random (random is confined to the
  CVAE's own element palette and still matches it).
* **Would change the conclusion:** a DFT check of the ~17 candidates with ΔE_dec ≤ 0 across both arms;
  if the CVAE's five survived and random's twelve did not, §2 would need revisiting. On present
  evidence there is no reason to expect that.
* **Statistical power:** n = 53 vs 50 on the energetics axis bounds any CVAE advantage at
  δ ≥ −0.19 (CHGNet, full protocol) / δ ≥ −0.31 (pyxtal-only) at 95 % confidence. A medium or large
  CVAE advantage in ΔE_dec is ruled out; a small one is not, and would not change a screening
  decision.

---

## 7. What I would tell the authors to write

1. Do not claim the CVAE produces better candidates. On the decomposition-energy screen it is
   statistically indistinguishable from random through the same filter, and random yields 2.5× more
   compositions that are actually stable against binary-oxide decomposition.
2. **Do** claim, with the numbers in §4 and §5.4, that the model acts as a learned chemical-plausibility
   prior: it suppresses open-shell cations by 55 pp relative to random and is enriched 3–8× in the
   hafnate/zirconate/aluminate chemistries that the independent Stage-1 screen selected — and that this
   survives a training-frequency-weighted control, so it is more than first-order element statistics.
3. Report the Ti³⁺/Sn²⁺ defect as a **model** failure (36.4 % vs 8.3 % for random), not a filter one,
   and connect it to the design target explicitly.
4. If the size-controlled top-N result (§5.2) is used, state the budget premise honestly in the same
   sentence, because the model's budget disadvantage is 178:1.
5. The negative result is the paper's most useful contribution. Composition-only generative models
   are widely proposed and rarely benchmarked against random sampling through the same post-filter;
   this work now has that benchmark, with a uMLIP screen attached, for both arms.

---

## 8. Files produced

**Scripts** (`analysis/a7_quality/`): `a7_build_sets.py`, `a7_chem_sanity.py`,
`a7_surrogates.py`, `a7_stats.py`, `a7_compare.py`, `a7_steelman2.py`, `a7_energetics_sample.py`,
`a7_run_energetics.py`, `a7_mace_refine.py`, `a7_energetics_analyze.py`, `a7_tables.py`,
`a7_figures.py`, plus run logs `a7_chgnet.log`, `a7_mace2.log`, `a7_surrogates.log`, `a7_compare.log`.

**Results** (`results/`):

| File | Contents |
|---|---|
| `a7_population_{cvae,random,random_matched,random_freq,random_dirichlet}.csv` | the candidate populations |
| `a7_population_summary.json` | population sizes and overlap |
| `a7_chemical_sanity.csv` | per-candidate oxidation states, open-shell flags, distance to training |
| `a7_surrogate_predictions.csv` | a5 surrogates applied to both populations |
| `a7_merged_candidates.csv` | the merged per-candidate table used for every comparison |
| `a7_compare_continuous.csv` | property comparisons: medians, Cliff's δ, bootstrap CI, Mann–Whitney p |
| `a7_compare_proportions.csv` | chemical-sanity proportions: Wilson/Newcombe CIs, Fisher p |
| `a7_compare_summary.json`, `a7_TABLE_overview.csv` | the overview table |
| `a7_stage1_enrichment.json` | Hf/Zr/Al enrichment odds ratios |
| `a7_steelman.csv`, `a7_steelman_size_controlled.csv` | the CVAE-favourable tests |
| `a7_energetics_random_{sample,eligible}.csv` | the 50 relaxed random candidates and the 331 eligible |
| `a7_hypothesis_relaxations_{chgnet,mace}.csv` | 374 CHGNet + 50 MACE relaxations (this session) |
| `a7_hypothesis_relaxations_mace_SEEDSTART_discarded.csv` | the discarded seed-start MACE run, kept for the record |
| `a7_energetics_all.csv` | per-candidate ΔE_dec, both potentials, both protocols, both arms |
| `a7_energetics_comparison.csv`, `a7_energetics_proportions.csv`, `a7_energetics_cross_potential.csv`, `a7_cross_potential_check.csv` | the energetics statistics and the per-candidate MACE-vs-CHGNet check |
| `a7_tables.tex` | five `booktabs` LaTeX tables |
| `a7_quality_comparison.pdf/.png` | three-panel summary figure |
