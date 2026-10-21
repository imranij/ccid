# Stage-1 forensics, reproduction and correction — MTCOMM-D-26-04679

Agent `a1`. Working directory `analysis/a1_stage1`, outputs in
`results/` with prefix `a1_`.

Everything below was computed by scripts in this deliverable, run on the datasets
already present in `<repo root>/`. No network access was used
(the figshare download in `ensure_dataset()` was never triggered — the local
`dielectric_constant.json.gz` is present, so no patch was needed).

Scripts (all kept, all re-runnable in this order):

| script | purpose |
|---|---|
| `a1_common.py` | shared loaders, the harmonised exclusion list, exact re-implementations of the published utilities/Pareto/acquisition/gate logic |
| `a1_corrected_stage1.py` | runs the four scoring variants + the imputation sweep + the dielectric pool |
| `a1_reproduce_check.py` | audits the deposited CSVs against a fresh unmodified re-run, and the manuscript tables against the CSVs |
| `a1_make_tables.py` | writes the deliverable CSVs and `a1_tables.tex` |

Throughout: **(C)** = computed here, **(F)** = read from an existing file, **(I)** = my interpretation.

---

## Part 1 — What the Stage-1 scoring rules actually are

The manuscript prints two different, mutually inconsistent equations and neither is
what the code does. Eq. (3) is a five-term expression with weights `w1..w5` that
exists nowhere in the code. Eq. (4), `S = 0.5·U_stability + 0.5·U_gap`, exists in the
code but only in `composition_cvae_common.load_material_table` — i.e. it builds the
**CVAE training table**, not any Stage-1 table in the paper.

The Stage-1 tables are produced by `inverse_design_baseline.py`, which uses **four
different scoring rules on four different populations**. Itemised (C, from reading and
re-running the code; also in `a1_true_scoring_rules.csv` and Table `tab:a1_true_rules`
of `a1_tables.tex`):

| # | Population | Scalarisation actually used | Utilities | Min–max reference population | E_g filter |
|---|---|---|---|---|---|
| 1 | MP dielectric (1,056 records → 757 after filter) | `inverse_design_score = 0.80·D + 0.20·N` | `D = 0.45·mm(poly_total) + 0.35·mm(n) + 0.20·mm(band_gap)`; `N` = local novelty | the **757 post-filter** records | **yes**, `band_gap ≥ 1.0` |
| 2 | Castelli perovskites (18,928) | `0.45·U_s + 0.35·U_g + 0.20·U_d` | `U_s = mm(−e_form)`, `U_g = mm(gap gllbsc)`, `U_d = 1[gap is direct]` | **Castelli rows only** | **no** |
| 3 | Wolverton oxides (4,914) | `0.65·U_s + 0.35·U_g` | `U_s = mm(−e_hull)`, `U_g = mm(gap pbe)`, `U_d ≡ 0` | **Wolverton rows only** | **no** |
| 4 | Double perovskites (1,306) | `candidate_score = U_g` **only** | `U_s ≡ 0.5` hard-coded, `U_g = mm(gap gllbsc)`, `U_d ≡ 0` | **double-perovskite rows only** | **no** |
| 5 | CVAE training table (25,148) | `0.5·U_s + 0.5·U_g` — the paper's Eq. (4) | `U_s` pooled over known-stability rows, `U_g` pooled over all rows | pooled | **no** |

Detail that must go into the revised Methods:

* **`safe_scaled(series, invert)`** (`inverse_design_baseline.py:49`). Coerces to numeric;
  if `invert`, negates *first*; then `MinMaxScaler` to [0,1]. **If the series has ≤1
  distinct non-NaN value it returns the constant 0.5 for every row** — so a degenerate
  objective silently becomes a uniform 0.5 rather than an error. Inversion therefore
  means `U = (x_max − x)/(x_max − x_min)`: utility 1.0 goes to the *most negative*
  formation energy **within that one file**, with no absolute meaning.
* **`local_novelty_score`** (only used for the dielectric pool). Build a 6-feature matrix
  `[nsites, space_group, volume, band_gap, n, poly_total]`, min–max each column over the
  757 filtered records, compute the full Euclidean pairwise-distance matrix, blank the
  diagonal, take the mean of the **10 smallest** distances per row, then min–max that
  mean over the 757 records. Note `space_group` (an arbitrary integer label) is used as a
  continuous coordinate — (I) that is not a meaningful metric dimension.
* **Objective-sensitivity profiles are *not* the ranking weights.** The ranking uses the
  four rules above. The sensitivity table re-scores *the already-pooled table* with a
  single three-term form `w_s·U_s + w_g·U_g + w_d·U_d` and five weight sets:
  balanced (0.45/0.35/0.20), stability_first (0.70/0.20/0.10), gap_first (0.25/0.65/0.10),
  direct_gap_bonus (0.35/0.35/0.30), risk_averse (0.80/0.15/0.05). So the "balanced"
  profile equals the Castelli rule but not the Wolverton or double-perovskite rule, and
  the "top-50 overlap" column compares against `candidate_score`, which was produced by a
  *different* (per-dataset) formula. The reported overlaps (12/15/17/3/14) are therefore
  not weight sensitivity alone — part of the disagreement is the branch inconsistency.
* **Diverse-acquisition batch.** Sort the pooled table by `candidate_score` descending,
  truncate to the top `min(1500, n)`, feature vector = `[U_s, U_g, U_d, candidate_score]`
  (NaN→0). Seed the batch with pool index 0 (the single highest-scoring record).
  Then repeat 19 times: for every remaining pool member compute
  `acquisition = 0.65·candidate_score + 0.35·min-Euclidean-distance-to-already-selected`
  and take the argmax. Batch size 20. Note `candidate_score` enters the feature vector
  *and* the utility term, so it is double-counted.
* **Pareto front.** Restricted to `stability_known == 1` (i.e. double perovskites are
  excluded). Sort by `(U_s desc, U_g desc)` and keep a row iff its `U_g` strictly exceeds
  the running maximum. I verified this sweep against an independent non-dominated-set
  computation: it is correct (7 = 7 on the published run). See §4.3 for a caveat about it.

---

## Part 2 — Reproduction

### 2.1 The deposited code reproduces the deposited CSVs, exactly

I copied the project, ran `inverse_design_baseline.py` and `candidate_validation_pack.py`
**unchanged**, and compared every output against the archived
`dgx/data/processed_results/*.csv` column by column (C, `a1_rerun_vs_deposited.csv`):

| artifact | result |
|---|---|
| `ceramic_ranked_real_candidates.csv` (757 × 13) | identical, max \|Δ\| = 1.2e-15 |
| `ceramic_pareto_perovskite_candidates.csv` (7 × 10) | identical, Δ = 0 |
| `ceramic_candidate_pool_summary.csv` (3 × 7) | identical, Δ = 0 |
| `ceramic_objective_sensitivity.csv` (5 × 11) | identical, Δ = 0 |
| `ceramic_diverse_acquisition_batch.csv` (20 × 8) | identical, Δ = 0 |
| `ceramic_validation_gate_summary.csv` (5 × 5) | identical, Δ = 0 |

**Stage 1 does not have the Stage-2 reproducibility problem.** The pipeline is
deterministic and the deposited artifacts are the ones the code produces. That is good
news and should be stated in the response to reviewers.

### 2.2 The manuscript tables, however, do not all match those CSVs

(C, `a1_reproduction_status.csv`; LaTeX version = `tab:a1_repro`.)

* **Table 5 (top dielectric candidates) — DOES NOT REPRODUCE.** The deposited code's
  top three are mp-27832 Tl₂SnCl₆ (0.644), mp-5606 AlTlF₄ (0.628), mp-5967 TiCoSb (0.518).
  The manuscript prints mp-5606 (0.282), mp-27832 (0.259), mp-22908 (0.230). Neither the
  ordering nor the score column matches. TiCoSb, P₂Pt, Ta₃N₅ (code ranks 3–5) do not
  appear in the manuscript table at all. I tried to reverse-engineer the printed scores
  from the deposited columns (least-squares over min–max `band_gap`, `n`, `poly_total`,
  `poly_electronic`, `log poly_total`, on both the filtered and unfiltered population,
  and a set of explicit candidate weightings including `0.5·U_gap + 0.5·U_ε` and equal
  thirds): **best residual 0.054 — no rule in or near the deposited code reproduces
  these numbers.** (C)
  Worse: **mp-5683 (Al₂O₃) and mp-2821 (MgF₂) are not in the deposited dielectric dataset
  at all.** Its only Al₂O₃ record is mp-7048 with E_g = 4.46 eV and ε = 8.98, not the
  5.21 / 8.45 printed; there is no MgF₂ record. (C) These two rows cannot be supported
  by the deposited data and must be removed or re-sourced with a citation.
  Also note the table currently recommends SiH₄ (silane, a pyrophoric gas) as a "real
  dielectric candidate".
* **Table 6 (top perovskite/oxide) — PARTIALLY REPRODUCES.** Rows 1–12 match the code
  exactly. Rows 13–14 print SrZrYAlO₆ (0.915) and YAlO₃ (0.912); the code's ranks 13–14
  are HoAlO₃ (0.9169) and CaHfMgZrO₆ (0.9167), with LuAlO₃ (0.9165) at 15. Three rows
  were silently dropped and two lower-scoring rows promoted.
* **Table 7 (Pareto front) — reproduces exactly** (all 7 rows, all utilities).
* **Table 8 (Pareto by dataset) — reproduces exactly** (Castelli 5 / 0.733 / 0.758;
  Wolverton 2 / 0.873 / 0.888).
* **Table 9 (pool summary) — reproduces exactly.**
* **Table 10 (objective sensitivity) — reproduces exactly** (all five profiles).
* **Table 11 (acquisition batch) — reproduces, but truncated.** The 12 printed rows are
  rows 1–12 of a **20**-row batch. The surrounding text calls them "the 12 candidates".
* **Table 12 (validation gates) — reproduces the code, but the code is wrong.** See §4.1.

---

## Part 3 — The corrected Stage-1 screen

### 3.0 What "corrected" means here

Four changes, each isolable (all four variants are written out so the authors can see
each step):

| variant | prefix | E_g ≥ 1 on all pools | harmonised exclusion | scalarisation | band-gap normalisation | stability metric |
|---|---|---|---|---|---|---|
| V0 published | `a1_legacy_*` | no | no | per-dataset (4 rules) | per-dataset | e_form / e_hull mixed |
| V1 filters only | `a1_filtersonly_*` | **yes** | **yes** | per-dataset (4 rules) | per-dataset | e_form / e_hull mixed |
| **V2 corrected (recommended)** | `a1_corrected_*` | **yes** | **yes** | **`S = 0.5·U_s + 0.5·U_g` for all three sources** | **pooled** | e_form / e_hull mixed, labelled |
| V3 metric probe | `a1_eformprobe_*` | yes | yes | 0.5/0.5 | pooled | **e_form for both**, pooled |

V2 is the recommendation because (i) it makes the code match the equation the paper
prints, (ii) it removes the gap-only double-perovskite branch, (iii) it removes the
direct-gap term that only one of three datasets can supply, and (iv) pooled band-gap
normalisation removes the artifact that each dataset's own widest-gap member is
automatically awarded `U_g = 1.000`.

V3 is reported as a robustness probe, **not** as a recommendation — see §4.2.

### 3.1 (a) E_g ≥ 1.0 eV on the perovskite/oxide pool

(C) 22,086 of the 25,148 pooled records have `band_gap` **exactly 0.0** (87.8 %).
Applying the paper's own insulator criterion:

| source | published n | E_g ≥ 1 eV | E_g ≥ 1 eV **and** element-clean |
|---|---|---|---|
| Castelli perovskites | 18,928 | 472 | **398** |
| Wolverton oxides | 4,914 | 679 | **444** |
| Double perovskites | 1,306 | 1,287 | **1,048** |
| total | 25,148 | 2,438 | **1,890** |

This is the single biggest change: the "25,148-record candidate pool" the abstract
advertises contains 1,890 records that satisfy the screen's own stated criterion. The
abstract's phrasing ("screens ... 25,148 perovskite and oxide records ... enforcing a
minimum band gap of E_g ≥ 1.0 eV") is not what the code did and must be reworded.

### 3.2 (b) Harmonised element exclusion

Final list (`a1_element_exclusion_list.csv`, `tab:a1_exclusions`), applied identically to
**both** Stage-1 pools and recommended for Stage 2:

1. **No stable isotope / radioactive** — Tc, Pm, Po, At, Rn, Fr, Ra, Ac, Th, Pa, U, Np,
   Pu, Am–Lr and the transactinides. *Justification:* these are not procurable for
   non-nuclear ceramic processing at any scale; licensing, shielding and waste handling
   make synthesis infeasible regardless of computed score. This is the criterion that
   removes KPaO₃, NaPaO₃ and NaTcO₃, which the manuscript itself already concedes are
   "screening artifacts" — the fix is to exclude them rather than to print them and
   apologise in the caption.
2. **RoHS / REACH-restricted** — Pb, Cd, Hg, Tl, As, Be. *Justification:* this is exactly
   the existing Stage-2 `HIGH_RISK_ELEMENTS` set minus its radioactive members, which are
   already covered by group 1. Using it in Stage 1 too is what makes the two stages
   consistent; keeping it out of Stage 1 is why PbHfO₃ reached the published acquisition
   batch while Pb-containing CVAE outputs were rejected.
3. **Noble gases** — He, Ne, Ar, Kr, Xe, Rn. *Justification:* cannot occupy a cation site;
   already excluded in the Stage-2 decoder, so this is a harmonisation, not a new rule.

I deliberately did **not** exclude the halogens/chalcogens F, Cl, Br, I, S, Se, Te. They
are the Stage-2 `PROCESS_RISK_ELEMENTS` set, which Stage 2 *flags* rather than rejects;
I kept that semantics and carry a `process_risk_elements` column through every output.

Effect (C): 5,990 of 25,148 perovskite/oxide records and 132 of the 757 filtered
dielectric records contain an excluded element. Specifically removed from published
tables: **Pareto front** KPaO₃, NaPaO₃; **acquisition batch** NaTcO₃ (rank 3), KPaO₃
(rank 4), PbHfO₃ (rank 12); **dielectric Table 5** Tl₂SnCl₆, AlTlF₄, TlF (and BeF₂ in
the manuscript's version of that table).

### 3.3 (c) The double-perovskite stability imputation

The `double_perovskites_gap` file has six columns — `formula, a_1, b_1, a_2, b_2,
gap gllbsc` — and **no energy field of any kind** (C). So `stability_utility = 0.5` is
not a modelling choice, it is a placeholder for a quantity that does not exist in the
source.

Two things compound in the published result:

1. `candidate_score` for that branch is `gap_utility` **alone** — the 0.5 is carried in
   the table but never enters the score.
2. `gap_utility` is min–maxed **within the double-perovskite file**, so its widest-gap
   member is assigned exactly 1.000 by construction.

**Therefore "CaSiYAlO₆ scores 1.000" carries no information beyond "CaSiYAlO₆ has the
largest band gap (8.343 eV) in `double_perovskites_gap.json.gz`".** It is not a
stability result, it is not comparable to the Castelli/Wolverton scores (which are capped
below 1.0 because they mix in a stability term), and its first place is guaranteed a
priori: no three-term record can reach 1.000 unless it simultaneously maximises all three
of its utilities. (I, supported by the arithmetic above.)

Sensitivity over the imputed value ρ, in the corrected V2 pipeline
(C, `a1_imputation_sensitivity.csv`, `tab:a1_imputation`; per-ρ top-50 tables and
acquisition batches are in `a1_corrected_rho*_top50.csv` and
`a1_imputation_acquisition_batches.csv`):

| ρ | top-ranked | source | top S | rank of CaSiYAlO₆ | S(CaSiYAlO₆) | DP in top 20 | DP in batch of 20 |
|---|---|---|---|---|---|---|---|
| 0.00 | BaHfO₃ | Castelli | 0.824 | **216** | 0.500 | 0 | 1 |
| 0.25 | BaHfO₃ | Castelli | 0.824 | **83** | 0.625 | 0 | 1 |
| 0.50 | BaHfO₃ | Castelli | 0.824 | **14** | 0.750 | 1 | 3 |
| 0.75 | CaSiYAlO₆ | Double perov. | 0.875 | **1** | 0.875 | 18 | 13 |
| 1.00 | CaSiYAlO₆ | Double perov. | 1.000 | **1** | 1.000 | 20 | 19 |

Plainly stated for the paper: **CaSiYAlO₆'s rank moves from 1 to 216 across a range of
the imputed constant that the data cannot distinguish between.** The double-perovskite
share of the top 20 moves from 0/20 to 20/20 over the same range. Nothing in the
published result about that compound survives the removal of the placeholder. The honest
options are (i) report the double perovskites as a *gap-only* shortlist in a clearly
separate table with no cross-dataset score, or (ii) keep them in the pooled table with
ρ = 0.5 and print this sensitivity table alongside. I recommend (i) as the primary
presentation and (ii) as supplementary.

### 3.4 Corrected results

**Top-ranked perovskite/oxide candidates** (V2, ρ = 0.5; `a1_TABLE_corrected_top_perovskite.csv`,
`tab:a1_corrected_top`). Top 10: BaHfO₃ 0.824, SrHfO₃ 0.821, CaHfO₃ 0.798, SrZrO₃ 0.797,
BaZrO₃ 0.794, LaAlO₃ 0.770, CaZrO₃ 0.764, SrAlO₂F 0.760, CaSiO₃ 0.757, CaAlO₂F 0.757.
CaSiYAlO₆ falls to rank 14 (0.750). (I) This is a far more defensible table than the
published one — it is a list of recognised wide-gap dielectric perovskites rather than a
list of the widest band gaps in one small auxiliary file.

**Corrected Pareto front** (`a1_TABLE_corrected_pareto.csv`, `tab:a1_corrected_pareto`):
7 members, all Castelli — BaHfO₃, SrHfO₃, CaHfO₃, BaZrO₃, NaTaO₃, KTaO₃, LiTaO₃. No Pa,
no Tc. (The V1 "filters-only" front, for comparison, has 9 members and is Wolverton-led;
`a1_filtersonly_pareto.csv`.)

**Pool summary** (`a1_TABLE_corrected_pool_summary.csv`): 1,048 / 444 / 398 as in §3.1;
max S now 0.824 (Castelli), 0.757 (Wolverton), 0.750 (double perovskites) — no branch
attains 1.000 any more.

**Objective sensitivity** (`a1_TABLE_corrected_objective_sensitivity.csv`): balanced,
stability-first and direct-gap-bonus all select LaAlO₃; gap-first selects CaSiYAlO₆;
risk-averse selects KTaO₃. Top-50 overlap with the balanced profile is 26/26/30/10/24 —
higher and more coherent than the published 12/15/17/3/14, because the base ranking and
the sensitivity re-scoring now use commensurate utilities.

**Acquisition batch** (`a1_TABLE_corrected_acquisition_batch.csv`): 20 candidates led by
BaHfO₃, LaAlO₃, CaSiYAlO₆, KTaO₃, NaTaO₃, CaHfO₃, CsTaO₃, SrHfO₃, BaZrO₃, SrZrO₃ …
Three double perovskites remain (all flagged `computational_validation_required`), no
excluded element appears.

**Validation gates** (`a1_TABLE_corrected_validation_gate_summary.csv`): standard review
14, computational validation required 3, priority expert review 2, expert review required
1 — **summing to 20**, as it must (see §4.1).

**Dielectric pool** (`a1_TABLE_dielectric_before_after.csv`, `tab:a1_dielectric`). After
element exclusion, 625 of 757 records remain and the top of the list becomes BiCl₃ 0.676,
Ta₃N₅ 0.622, TiCoSb 0.600, SnCl₂ 0.588, P₂Pt 0.532. (I) **This is still not a table of
ceramic dielectrics** — TiCoSb is a half-Heusler, P₂Pt and Si₄P₄Ru are phosphides, and
the chlorides are not insulator ceramics. The exclusion list removes the toxicity
problem but not the category problem, because the objective rewards large `poly_total`
and `n`, which favours soft, polarisable halides. I therefore also generated an
oxide-only restriction (`a1_dielectric_corrected_oxideonly.csv`, third column pair of the
same table): BiBrO 0.506, TiO₂ 0.439, Ca₄P₂O 0.357, SrSeO₃ 0.339, MnO₂ 0.320, ZrO₂ 0.313,
NaMnO₂ 0.309, HfO₂ 0.300, La₂TeO₂ 0.282, LiAlO₂ 0.279. I offer this as a supplementary
view and flag the choice explicitly rather than silently restricting the pool.

### 3.5 Before/after summary

(`a1_TABLE_before_after_top10.csv`, `tab:a1_before_after`.) The perovskite/oxide top 10
changes completely: published = seven double perovskites + LaAlO₃ + ErAlO₃ + CaZrYAlO₆;
corrected = nine Castelli perovskites + one Wolverton entry. Of the published 20-member
acquisition batch, 3 members are removed outright by the exclusion list and the
double-perovskite share drops from 8/20 to 3/20. The paper must state that the Stage-1
tables changed, and why — §3.1–3.3 give the three reasons in a citable form.

---

## Part 4 — Further inconsistencies the reviewers did **not** raise

### 4.1 The published validation-gate counts sum to 23 for a 20-candidate batch

`candidate_validation_pack.load_candidates` reads the 20-row acquisition batch and then
left-merges it onto `ceramic_ranked_perovskite_candidates.csv` **on `formula`**. That
ranked table has 25,148 rows but only 22,669 unique formulas, so any batch member whose
formula is not unique is duplicated by the merge. Three are: BaNbO₃, KTaO₃ and LaScO₃
each appear twice. The review pack therefore has 23 rows, and the published Table 12
counts (9 + 8 + 4 + 1 + 1 = 23) describe 23 rows for a 20-candidate batch (C). The
manuscript text compounds this by calling the batch "the 12 candidates". My
`validation_gate()` classifies the batch directly and the corrected counts sum to 20.

### 4.2 The pooled "stability utility" compares two physically different quantities

`U_s` is `mm(−e_form)` for Castelli but `mm(−e_hull)` for Wolverton, computed
independently per file, and then the two are pooled into one ranking, one Pareto front,
one sensitivity table and one acquisition batch as though they were the same axis. They
are not: formation energy and energy above hull are different quantities with different
zeros.

Quantitative evidence (C): **2,479 formulas appear in both the Castelli and the Wolverton
file.** For those same formulas the two files' *own* `e_form` columns have mean +1.185 and
−1.182 eV/atom respectively — a systematic offset of ≈2.37 eV/atom — and correlate at
only **r = 0.645**. So even the identically-named field is not the same quantity across
the two sources; the `e_form` vs `e_hull` comparison the code actually performs is weaker
still. (Independently: 258 of 4,914 Wolverton `e_hull` values are **negative**, which is
impossible for a true hull distance, so that column's semantics are themselves unclear.)

To quantify the consequence I ran V3, which uses `e_form` for both sources and pools the
normalisation. **The top-20 overlap between V2 and V3 is 7/20** (C) and the V3 ranking is
entirely Wolverton-led (ErAlO₃, TmAlO₃, LuAlO₃ …) because Castelli's positive-offset
`e_form` values are crushed to low utility. Changing only the stability field — a choice
the manuscript never states — reorders two thirds of the shortlist.

My recommendation (I): keep the per-dataset normalisation (V2), but say explicitly in the
Methods that `U_s` is a **within-dataset relative rank**, not an absolute stability, that
it is not comparable across sources, and carry the `stability_metric_type` column into the
published tables (I have done this in `tab:a1_corrected_top`). Anything stronger requires
recomputing a single consistent energy for all records, which is outside a revision.

### 4.3 One published Pareto-front member survives on a 2·10⁻¹⁶ rounding difference

In the published run KTaO₃ (Castelli) has `stability_utility = 1.0000000000000002` while
KPaO₃ (Wolverton) has exactly `1.0` — both print as "1.000" in Table 7. Because the sweep
sorts by `U_s` descending, that 2.2·10⁻¹⁶ puts KTaO₃ *first*, so it is retained and then
KPaO₃ (larger `U_g`) is also retained. In exact arithmetic the two tie at `U_s = 1.0`,
KTaO₃ has the smaller `U_g` (0.629 vs 0.681), and **KTaO₃ would be dominated and dropped**
(C — my independent non-dominated-set routine drops it when the utilities are rounded to
12 decimals and keeps it when they are not). The front is stable in the corrected run,
but the published Table 7 has a member whose presence is a floating-point accident across
two incommensurate metrics (§4.2).

### 4.4 A ranking objective with only one absolute anchor

`U_s`, `U_g` and the dielectric `D` are all min–max normalised over the *screened
population*, so every score is relative to the pool and changes when the pool changes.
This is why the dielectric design scores change between the published run and the
element-filtered run even for compounds that were never removed (BiCl₃ goes 0.470 → 0.676
purely because Tl₂SnCl₆ and AlTlF₄ left the normalisation population). (C) The paper
should say that design scores are pool-relative and are not comparable between tables.
Related: Castelli's `e_form` is positive for 96 % of its records (median +1.36 eV/atom),
so a "stability utility" of 0.9 there does not mean the compound is stable.

### 4.5 Two mutually inconsistent "common ceramic element" whitelists in the codebase

`composition_cvae_common.COMMON_CERAMIC_ELEMENTS` has **41** elements;
`candidate_validation_pack.COMMON_CERAMIC_ELEMENTS` has **28**, and they are not nested:
Dy, Er, Ga, Gd, Ho, In, Lu, Nd, Pr, Sc, Sm, Tb, Tm, Yb are in the first but not the
second, and Mn is in the second but not the first (C). So a Stage-2 candidate containing
Er is "valid", while a Stage-1 candidate containing Er is flagged "less common ceramic
screening element" and pushed to `expert_review_required`. Whichever list the revision
keeps, it should be defined once and imported by both scripts.

### 4.6 Smaller items worth a sentence each

* The pooled 25,148-record table contains only **22,669 unique formulas** (C); 2,479
  formulas are duplicated across Castelli and Wolverton, generally with different band
  gaps (e.g. KTaO₃ appears at 4.4 eV in Castelli and 2.324 eV in Wolverton). The pool is
  a concatenation, not a deduplicated candidate set, and the duplicate records compete
  against each other in the ranking and in the acquisition batch.
* `direct_gap_utility` is hard-set to 0 for Wolverton and double-perovskite records
  because those files do not report gap directness. It carries weight 0.20 in the
  "balanced" ranking rule and 0.30 in the `direct_gap_bonus` profile, so those two
  profiles systematically penalise two of the three sources for a missing field rather
  than an unfavourable value. I preserved this behaviour in the corrected sensitivity
  table (for continuity) but flagged it in the caption; the cleaner fix is to drop the
  term or restrict that profile to Castelli.
* `pot_ferroelectric` is loaded for the dielectric pool and never used.
* The dielectric novelty feature matrix uses `space_group` (a categorical label) as a
  Euclidean coordinate.

---

## Deliverables

LaTeX: **`a1_tables.tex`** — 12 `booktabs` tables, `\toprule`/`\midrule`/`\bottomrule`
only, no `\hline`, no blank rows, plain `table` environments (correct for the
manuscript's single-column `3p` elsarticle class). Compiled with `pdflatex` at
`textwidth = 460pt` (elsarticle 3p/12pt): **zero overfull boxes**. Requires
`\usepackage{booktabs}` only.

Paste-ready CSVs matching those tables: `a1_true_scoring_rules.csv`,
`a1_element_exclusion_list.csv`, `a1_reproduction_status.csv`,
`a1_TABLE_corrected_top_perovskite.csv`, `a1_TABLE_corrected_pareto.csv`,
`a1_TABLE_corrected_pool_summary.csv`, `a1_TABLE_corrected_objective_sensitivity.csv`,
`a1_TABLE_corrected_acquisition_batch.csv`,
`a1_TABLE_corrected_validation_gate_summary.csv`, `a1_TABLE_imputation_sensitivity.csv`,
`a1_TABLE_before_after_top10.csv`, `a1_TABLE_dielectric_before_after.csv`.

Full machine-readable outputs: `a1_{legacy,filtersonly,corrected,eformprobe}_*`
(top-200 ranking, Pareto, pool summary, sensitivity, acquisition batch, expert-review
pack, gate summary for each variant), `a1_corrected_rho*_top50.csv`,
`a1_imputation_{sensitivity,acquisition_batches}.csv`,
`a1_dielectric_{legacy_ranked,corrected_ranked,corrected_oxideonly,removed_by_exclusion}.csv`,
`a1_rerun_vs_deposited.csv`.

## Caution to the authors

1. Stage 1 is reproducible; the **manuscript's Table 5 is not**, and two of its rows cite
   material IDs absent from the deposited dataset. That must be fixed before resubmission
   independently of anything a reviewer asked for.
2. Do not describe the corrected screen with Eq. (3). Print the itemised rules (§1) or,
   if you adopt V2, print `S = 0.5·U_s + 0.5·U_g` and say so consistently for all three
   sources.
3. Every design score in this paper is pool-relative and dimensionless. Avoid any
   phrasing that implies an absolute property prediction — in particular avoid
   "CaSiYAlO₆ scores 1.000" and any sentence of the form "X scores highest because it
   combines a wide band gap with a low formation energy" for a record that has no
   formation energy.
