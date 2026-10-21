# a9 — Four-arm uMLIP energetics: random vs element statistics vs ion substitution vs corrected CVAE

**Every number below was computed from relaxations run for this study (arms B, C, D, plus 8 spinel
re-builds) or read from a7/a4 relaxation tables. Nothing is estimated.**

## Answer

**Ion substitution (arm C) is the only arm whose candidates are energetically better as a distribution.
The advantage is large and holds under both potentials and all three protocol views. It does not
make C the better discovery engine.** C's median decomposition energy into binary oxides is
+0.012 eV/atom against +0.086 for the corrected CVAE (D). Cliff's δ(C − D) is −0.40 [95 % CI −0.58, −0.18]
with Mann–Whitney p = 0.0006 (Holm 0.002) under CHGNet, and −0.35 [−0.55, −0.12], p = 0.003 under
MACE-MP-0. Kruskal–Wallis across the four arms gives p = 0.0004, and this comes entirely from C: among
A, B and D alone, p = 0.69. But 23 of C's 50 candidates are compounds already in the training table,
including all five of its most favourable ones (BaTiO₃, BaHfO₃, Ba₂TiSnO₆, Ba₂HfTiO₆, Ba₂ZrTiO₆).
On the fraction of candidates that are actually stable against binary oxides (ΔE_dec ≤ 0), C is not
significantly ahead of D (46 % vs 30 %, Fisher p = 0.15; MACE 36 % vs 30 %, p = 0.67). On the
discovery-relevant count (**novel** compositions with ΔE_dec ≤ 0 under **both** potentials, per 50
relaxed), D has 12, A has 8, C has 8 and B has 7. None of these differ from D significantly
(Fisher p ≥ 0.31). **The corrected CVAE is energetically indistinguishable from uniform-random and
element-statistics sampling.** In the full protocol, all |δ| ≤ 0.09 and all p ≥ 0.44 under both
potentials. The largest gap in any view is B vs D in the MACE pyxtal-only subset: δ = −0.21, p = 0.11,
in B's favour. Its one clear energetic
gain is over the deposited CVAE: 30 % vs 9.4 % ≤ 0 (Fisher p = 0.012; MACE 30 % vs 7.5 %,
p = 0.005), with the same median. With n = 50 per arm, these tests can detect a medium-to-large shift
in the distribution, as C shows. They cannot resolve differences in the ≤ 0 fraction smaller than
about 20–25 percentage points. The ranking among A, B, D and novel-only C is therefore **unresolved,
not shown to be equal**.

**A correction that also affects a7.** The two most negative values in the whole study were artefacts
of a bug in a4's spinel prototype builder. One is a7's headline random candidate MgScGaO₄
(−0.748 eV/atom). The other is D's BaMgZrO₄ (−0.764 eV/atom). Details are in §2. The corrected
values are −0.023 (MgScGaO₄) and +0.059 eV/atom (BaMgZrO₄). **a7's sentence "the single most
promising composition found anywhere in this work is MgScGaO₄ at −0.748 eV/atom" must be withdrawn.**
a7's arm-level statistics are essentially unchanged: MgScGaO₄ stays ≤ 0, the ≤ 0 count stays 12/50
and the median stays the same.

---

## 1. What was run

| Arm | Population sampled from | eligible (14-cation set) | relaxed | CHGNet relaxations | MACE refinements |
|---|---|---|---|---|---|
| A random, uniform-41 | a7: a2 `rand_uniform41` seed 123, 11,744 unique valid | 331 (2.8 %) | 50 (a7, reused) | 374 (a7) + 4 re-built | 50 (a7), 1 replaced |
| B element statistics | a7/a2 `rand_freq41` seed 123, 8,870 unique valid (`a7_population_random_freq.csv`) | **680 (7.7 %)** | 50 | 372 | 50 |
| C ion substitution | generation agent: pooled unique valid seeds 0–4, 765 | 154 (20.1 %) | 50 | 331 | 50 |
| D corrected CVAE (fb0.5_b0.01) | generation agent: pooled unique valid seeds 0–4, 2,135 | 839 (39.3 %) | 50 | 318 + 4 re-built | 50, 1 replaced |
| deposited CVAE (reference row) | a4, 66 candidates, 53 with references | n/a | 53 | 442 (a4) | 67 (a4) |

**Selection rule (identical for every arm, a7's rule).** Keep only formulas whose cations all lie in
{Al Ga Sc Y La Zr Hf Ti Sn Si Ba Sr Ca Mg}. Then draw `default_rng(20260913).choice(n, min(50, n),
replace=False)` and keep the rows in their original order. For B this was done here
(`a9_energetics_sample_B.py`). For C and D the generation agent applied the same rule
(`a9_umlip_sample.py`). It was re-applied here as a check (`a9_energetics_sample_CD.py`) and kept all
50 formulas unchanged.

**Protocol (a7's corrected one, unchanged).**
- CHGNet: `a4_run_screen.run_candidates` → `a4_hypotheses.all_hypotheses(n_random=4, seed=0)`, then
  `a4_relax.relax(fmax=0.10, steps=150)` on every hypothesis.
- MACE-MP-0 (medium, float32): refinement of the best CHGNet hypothesis, started from the CHGNet
  geometry, with `fmax=0.10, steps=120`.
- ΔE_dec: `a4_analyze.analyze()` via a7's own staging helper (`a7_energetics_analyze.analyze_table`),
  against a4's existing reference files. No reference was re-relaxed.
- Completeness filter: as in a7, applied to A–D. No candidate was dropped.
- Jobs ran strictly one at a time, at one thread each.

The only operational difference is where the CIFs are written: `results/a9_structures/<arm>/` instead
of `a4_structures/`. This means a formula shared with a4 or a7 cannot overwrite their deposited CIFs.

**Every arm finished.** 50/50 candidates per arm have a complete hypothesis set and a reference set
under both potentials. Nothing was lost to time, convergence or missing references.

## 2. The spinel-builder bug (found here, corrected here, a4 code left untouched)

The two most negative values in the study came from spinel prototypes containing Mg, so I inspected
the relaxed CIFs. **`a4_GaMgO4Sc_spinel_Sc[GaMg]O4_chgnet.cif` contains Sc₄Ga₂O₈, with no Mg at
all.** `spinel_Zr[BaMg]O4` for BaMgZrO₄ contains Ba₂Zr₄O₈.

The cause is in `a4_hypotheses.prototype_hypotheses`, spinel branch. It writes cations B and C onto the
16d ("Al") sites and *then* replaces every "Mg" in the cell by cation A. When B or C is Mg, the Mg just
placed is overwritten as well. The wrong-composition energy per atom is then used as if it belonged to
the candidate. MACE refines the same wrong cell, so the two potentials "agree".

- **Audit** (`a9_energetics_hypothesis_audit.py`, `a9_energetics_hypothesis_audit.csv`): all
  1,837 prototype and pyxtal hypotheses across A, B, C, D and the deposited CVAE were regenerated and
  their compositions checked. Exactly **8 hypotheses in 2 formulas** are wrong: 4 in MgScGaO₄ (A) and
  4 in BaMgZrO₄ (D). No other formula and no deposited-CVAE candidate is affected.
- **Fix** (`a9_energetics_spinel_fix.py`): the same template, site assignment and tags were used, but
  substitution was done by site index. Composition was asserted before relaxing. CHGNet was run with
  `fmax 0.10 / 150 steps`. MACE refined each formula's best *valid* CHGNet hypothesis.
- **Effect.** MgScGaO₄ goes from −0.748 to **−0.023** (CHGNet) and from −0.726 to −0.015 (MACE), so it is
  still ≤ 0. BaMgZrO₄ goes from −0.764 to **+0.059** (CHGNet) and from −0.735 to +0.072 (MACE). D's
  ≤ 0 count drops from 16 to 15.

All headline numbers here are the corrected ones. The uncorrected, a7-identical numbers are kept in
`*_asrun.csv` and `a9_energetics_table_asrun.tex`. In the as-run version C vs D is δ = −0.37,
p = 0.0016, and D is 32 % ≤ 0. No conclusion changes.

## 3. Results

ΔE_dec in eV/atom; lower is better. δ is Cliff's delta (arm − D, so negative means the arm is lower,
i.e. better), with a 95 % CI from 2,000 bootstrap resamples. p_MW is from a two-sided Mann–Whitney
test; p_F is Fisher's exact test on the ≤ 0 fraction. Wilson CIs are given for the fractions. Full
tables: `a9_energetics_summary.csv` and `a9_energetics_pairwise_vs_D.csv`. LaTeX:
`a9_energetics_table.tex`.

### 3.1 Full protocol (prototypes + 4 pyxtal), n = 50 per arm

| Arm | CHGNet median | CHGNet ≤ 0 [95 % CI] | δ vs D [CI] | p_MW / p_F | MACE median | MACE ≤ 0 | δ vs D (MACE) | p_MW / p_F |
|---|---|---|---|---|---|---|---|---|
| A random | +0.086 | 24 % [14, 37] | −0.02 [−0.24, +0.23] | 0.90 / 0.65 | +0.095 | 24 % | −0.03 | 0.79 / 0.65 |
| B element stats | +0.068 | 24 % [14, 37] | −0.09 [−0.31, +0.15] | 0.44 / 0.65 | +0.079 | 18 % | −0.06 | 0.61 / 0.24 |
| **C substitution** | **+0.012** | **46 % [33, 60]** | **−0.40 [−0.58, −0.18]** | **0.0006** / 0.15 | **+0.026** | 36 % | **−0.35 [−0.55, −0.12]** | **0.003** / 0.67 |
| D corrected CVAE | +0.086 | 30 % [19, 44] | ref. | — | +0.108 | 30 % | ref. | — |
| deposited CVAE (n = 53) | +0.086 | 9.4 % [4, 20] | +0.00 [−0.23, +0.24] | 0.98 / **0.012** | +0.100 | 7.5 % | +0.08 | 0.47 / **0.005** |

Kruskal–Wallis over A–D: CHGNet H = 18.1, p = 0.0004; MACE H = 13.7, p = 0.003. Over A, B and D only:
p = 0.69 (CHGNet) and 0.86 (MACE). A χ² test on the ≤ 0 fractions over A–D gives p = 0.056 (CHGNet)
and 0.21 (MACE). Holm-adjusted p for C vs D: 0.002 (CHGNet) and 0.008 (MACE).

### 3.2 PyXtal-only (removes the hypothesis-count asymmetry)

Median hypotheses in the full protocol are 6 for A, B and C, and 4 for D and the deposited CVAE. When
every arm is restricted to the 4 pyxtal trials (CHGNet), the C advantage *grows*: δ(C − D) = −0.47
[−0.65, −0.26], p = 5×10⁻⁵, and ≤ 0 is 42 % vs 16 %, Fisher p = 0.008. This is the only cell where the
fraction difference is significant. MACE (formulas whose best CHGNet structure was pyxtal: 37 C vs
41 D) gives δ = −0.42, p = 0.0014. So C's advantage is **not** an artefact of C's perovskite formulas
receiving a textbook prototype. In the same view, D's full-protocol ≤ 0 fraction falls from 30 % to 16 %,
the same as A. Part of D's 30 % therefore comes from prototype hypotheses (pyrochlore, double
perovskite, spinel) that fit its stoichiometries.

### 3.3 Novel only (not in the training table, ≥ 2 cations)

| Arm | n | CHGNet median | ≤ 0 | δ vs D [CI], p_MW | MACE ≤ 0 | δ vs D (MACE), p_MW |
|---|---|---|---|---|---|---|
| A | 45 | +0.094 | 22 % | +0.03 [−0.20, +0.28], 0.80 | 22 % | +0.01, 0.94 |
| B | 48 | +0.068 | 23 % | −0.09 [−0.32, +0.16], 0.45 | 17 % | −0.06, 0.59 |
| **C** | **26** | **+0.008** | 46 % | **−0.38 [−0.62, −0.14], 0.007** | 31 % | −0.27 [−0.52, −0.02], 0.059 (Holm 0.18) |
| D | 48 | +0.086 | 29 % | ref. | 29 % | ref. |
| deposited CVAE | 51 | +0.086 | 9.8 % | −0.01, 0.95 | 7.8 % | +0.07, 0.54 |

Removing known compounds roughly halves C (50 → 26). The C-vs-D median shift survives under CHGNet,
but under MACE it weakens to borderline (p = 0.059). The ≤ 0 fraction under MACE is identical
(31 % vs 29 %).

**Discovery yield per 50 relaxed** (novel, ≥ 2 cations, ΔE_dec ≤ 0 under both potentials;
`a9_energetics_novel_yield.csv`):

| | A | B | C | D | deposited |
|---|---|---|---|---|---|
| count | 8 | 7 | 8 | 12 | 4/53 |
| Fisher p vs D | 0.45 | 0.31 | 0.45 | — | 0.029 |

Including known compounds, C has 17 that are ≤ 0 under both potentials; 9 of those 17 are training
compounds.

### 3.4 Cross-potential agreement and convergence (full protocol)

| Arm | MACE − CHGNet MAD | max abs | sign agreement | CHGNet: all hypotheses converged | CHGNet: best structure converged | MACE refinement converged |
|---|---|---|---|---|---|---|
| A | 0.029 | 0.222 | 92 % | 71 % | 80 % | 98 % |
| B | 0.026 | 0.074 | 90 % | 68 % | 64 % | 98 % |
| C | 0.036 | 0.324 | 86 % | 65 % | 68 % | 98 % |
| D | 0.035 | 0.139 | 92 % | 59 % | 62 % | 96 % |
| deposited | 0.030 | 0.219 | 98 % | 71 % | 62 % | 99 % |

All arms are converged about equally well after MACE refinement (96–98 %), and all agree across
potentials at the a4/a7 level (~0.03 eV/atom). C's largest disagreement is BaSiO₃: CHGNet +0.031,
MACE −0.293, a single 0.32 eV/atom outlier. The 150-step CHGNet budget leaves 30–40 % of hypotheses
unconverged in every arm, and slightly more in D. This is the shared a4 protocol's limitation, not an
arm-specific one.

### 3.5 Most favourable compositions (CHGNet full protocol; MACE in brackets; **K** = in training table)

- **A random:** Sr₂ZrTiO₆ −0.178 [−0.178] **K**; Sr₂CaSiO₅ −0.129 [−0.101]; SrTi₂SnO₅ −0.124
  [−0.106]; BaLa₂SiO₆ −0.051 [−0.004]; SrHfSi₂O₇ −0.049 [−0.139] (also drawn in D). Corrected
  MgScGaO₄ (−0.023) is now 7th.
- **B element stats:** Ba₂ZrTiO₆ −0.266 [−0.253] **K** (also in C); Ba₂TiSiO₆ −0.200 [−0.184];
  BaTiSi₂O₇ −0.112 [−0.090]; BaHfZr₂O₇ −0.069 [−0.041] (also in D); BaSr₂SnO₅ −0.060 [−0.017].
- **C substitution:** Ba₂TiSnO₆ −0.278 [−0.241] **K**; BaTiO₃ −0.276 [−0.275] **K**; Ba₂HfTiO₆
  −0.267 [−0.254] **K**; Ba₂ZrTiO₆ −0.264 [−0.252] **K**; BaHfO₃ −0.261 [−0.243] **K**. Its best
  *novel* ones are SrCaSiO₄ −0.132 [−0.095], BaSr(SiO₃)₂ −0.132 [−0.146], Ba₂SiSnO₆ −0.116
  [−0.082], BaSrSnO₄ −0.109 [−0.076] and BaTiSiO₅ −0.074 [−0.024].
- **D corrected CVAE:** Ba₂HfZrO₆ −0.254 [−0.236] **K**; Ba₂HfSiO₆ −0.113 [−0.090]; SrCaSnO₄
  −0.096 [−0.070]; BaHfZr₂O₇ −0.069 [−0.041] (also in B); SrMg(SnO₃)₂ −0.068 [−0.051] (also in C).
- **Deposited CVAE (reference):** BaTiGaO₄ −0.127 [−0.066]; SrCaZrO₄ −0.047 [−0.047]; Ba₂ScAlO₅
  −0.038 [−0.008]; SrHfZr₂O₇ −0.017 [−0.002]; BaZrAl₂O₆ −0.013 [+0.016].

Training-table membership among the 50 relaxed per arm: A 5, B 2, **C 23**, D 2, deposited 2. Of C's
23 CHGNet ≤ 0 candidates, 11 are known. Two of A's twelve are known; in a7 Sr₂ZrTiO₆ was listed as a
random "discovery" without being marked as known.

## 4. Caveats (read before quoting)

1. **Different sampling pools.** A and B were drawn from the single seed-123 a2 runs, as the brief
   specified (a7's population). C and D were drawn from the pooled unique-valid sets of seeds 0–4. The
   selection rule is identical, and a uniform draw over eligible unique formulas should depend only
   weakly on the pool. It is still a difference and should be stated.
2. **Chemistry is restricted to the 14 reference cations.** This keeps 2.8 % (A), 7.7 % (B), 20 % (C)
   and 39 % (D) of each arm's unique-valid output. Conclusions apply within that chemical support only.
3. **C's composition mix differs.** C has 14 two-cation formulas (12 known) and one binary oxide.
   ZrO₂ is itself a reference oxide, so its +0.148 is a structure-search miss and not a candidate; it is
   excluded from the novel-only view. Restricted to 3-cation formulas, C is still 51 % ≤ 0 (18/35;
   `a9_energetics_by_ncations.csv`), so the advantage is not driven by small ABO₃ cells.
4. **ΔE_dec is measured only against binary oxides.** It is an upper bound from a 4-trial search and
   not a hull distance (a4 §3.2). A known perovskite coming out ≤ 0 confirms the method works; it is
   not a discovery.
5. **Statistical power.** With n = 50 per arm, a median shift of δ ≈ 0.35–0.47 is detected
   reliably. Differences of about 20 points or less in the ≤ 0 fraction, and differences in discovery
   yield, are not.

## 5. Files

**Scripts** (`analysis/a9_fourarm/`):
- `a9_energetics_sample_B.py`, `a9_energetics_sample_CD.py`: sampling.
- `a9_energetics_run.py`: CHGNet and MACE per arm.
- `a9_energetics_driver.sh`: sequential driver with flag polling.
- `a9_energetics_hypothesis_audit.py`, `a9_energetics_spinel_fix.py`: bug audit and fix.
- `a9_energetics_analyze.py`: corrected by default; `--as-run` gives the uncorrected numbers.
- Logs: `a9_energetics_*.log`.

**Results** (`results/`):
- Samples: `a9_energetics_sample_{B,C,D}.csv`, `a9_energetics_eligible_B.csv`.
- Relaxations: `a9_hypothesis_relaxations_{B,C,D}_{chgnet,mace}.csv`,
  `a9_hypothesis_relaxations_fix_{chgnet,mace}.csv`. CIFs: `a9_structures/{B,C,D,fix}/`.
- Analysis: `a9_energetics_all.csv` (per candidate, all arms, potentials and protocols),
  `a9_energetics_summary.csv`, `a9_energetics_pairwise_vs_D.csv`, `a9_energetics_kruskal.csv`,
  `a9_energetics_cross_potential.csv`, `a9_energetics_cross_potential_per_candidate.csv`,
  `a9_energetics_ranked_candidates.csv`, `a9_energetics_top5_per_arm.csv`,
  `a9_energetics_novel_yield.csv`, `a9_energetics_by_ncations.csv`, `a9_energetics_overlap.csv`,
  `a9_energetics_run_status.csv`, `a9_energetics_hypothesis_audit.csv`.
- Uncorrected versions: `*_asrun.csv`.
- LaTeX: `a9_energetics_table.tex` (booktabs tabular body, three panels), plus
  `a9_energetics_table_asrun.tex`.
