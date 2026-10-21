# a6 — Figures for MTCOMM-D-26-04679 (revision)

**Agent 6 · figure production · every number traced to a deposited result file**

Deliverables: eight vector PDFs plus 600 dpi PNGs in
`results/figures/`. Scripts and the shared style module in
`analysis/a6_figures/` (`a6_style.py`, `a6_fig*.py`, `a6_make_all.sh`).
`sh a6_make_all.sh` rebuilds the whole set from the result files.

---

## 0. The style system

`a6_style.py` is imported by every figure script and is the only place any visual
decision is made.

* **Type.** Liberation Sans (Helvetica metrics) throughout — 8 pt body, 8 pt axis
  labels, 7 pt tick labels, 9 pt bold panel letters, 5.7–6.6 pt for in-plot
  annotation. Figures are built at their **final print size**, so nominal = effective
  size. Include them with the widths in §1; scaling them in LaTeX will shrink the type
  below 8 pt.
* **Canvas.** `COL = 3.30 in` (include at `width=\columnwidth`) and
  `PAGE = 6.00 in` (include at `width=0.85\textwidth`, which is what the manuscript
  already writes for Fig. 1).
* **Palette.** Colour-blind-safe *and* greyscale-safe: the two arms compared in every
  figure differ in hue **and** lightness — CVAE `#1C3F66` (L\* ≈ 27) against random
  control `#E8A33C` (L\* ≈ 72). Supporting series are `#9B2D20` brick (L\* ≈ 38),
  `#3E7B62` teal (L\* ≈ 46), `#6E5AA0` violet (L\* ≈ 45). Converted to greyscale the
  two arms separate by ~45 L\* units, so the figures survive a monochrome print.
* **Chart junk.** No top/right spines, no titles inside the image, grids only where
  they carry reading value and always behind the data, direct labelling in preference
  to legends wherever the label fits.
* **Output.** `pdf.fonttype = 42` (TrueType embedded, text selectable and editable),
  vector PDF plus a 600 dpi PNG of the same name. **No seaborn is imported anywhere.**

---

## 1. Figure-by-figure provenance

### `fig1_concept.pdf` — conventional vs controlled evaluation (6.00 × 2.62 in, `0.85\textwidth`)
Schematic; script `a6_fig1_concept.py`. The only numerals are the two headline pairs:

| Number in the figure | File | Row / column |
|---|---|---|
| validity 0.330, 69 distinct formulas | `a2_random_baseline_summary.csv` | `arm=cvae`, `validity_rate.mean`, `unique_valid_count.mean` |
| validity 0.727, 11,821 distinct formulas | `a2_random_baseline_summary.csv` | `arm=rand_uniform41`, same columns |

Filter description ("integer stoichiometry · charge balance · 41-element whitelist")
is the manuscript's Algorithm 1.

### `fig2_funnel.pdf` — measured attrition, both arms (6.00 × 3.55 in, `0.85\textwidth`)
Script `a6_fig2_funnel.py`. Panel (a) counts on a log axis; panel (b) the pass rate
relative to the preceding gate.

| Gate | CVAE | Random (uniform, 41) | File / column |
|---|---|---|---|
| Sampled | 20,000 | 20,000 | by construction, 5 seeds |
| Valid | 6,604 ± 37 | 14,547 ± 34 | `a2_random_baseline_summary.csv`, `validity_rate` mean/std × 20,000 |
| Distinct formulas | 69 ± 2 | 11,821 ± 51 | same file, `unique_valid_count` mean/std |
| Novel | 68 ± 2 | 11,793 ± 50 | same file, `novel_unique_count` mean/std |
| uMLIP-assessed (CHGNet) | 53 | 50 | `a7_energetics_comparison.csv`, `calc=chgnet, mode=full`, `n_a` / `n_b` |
| Favourable, ΔE_dec ≤ 0 | 5 | 12 | `a7_energetics_proportions.csv`, `calc=chgnet, mode=full`, `k_a` / `k_b` |

Ratios printed in the right margin: 2.2× (validity), 171× (distinct), 173× (novel),
computed from those means.

**Honesty note carried inside the figure.** Gates 5–6 are drawn hatched, below a dashed
rule, with an explicit `n.a.` pass rate, and the figure's own footnote says why: they
are a *screening subset*, not a population gate. All 66 candidates of the deposited
CVAE archive were attempted and 53 had complete binary-oxide references, whereas the
random arm was sub-sampled to 50 of its 331 reference-eligible candidates
(`a7_REPORT.md` §2.1). The two arms are comparable there **only in rate**
(9.4 % vs 24.0 %), never in count. This was the one place where a funnel could easily
have been drawn misleadingly, so it is flagged three ways.

**One inconsistency the reader should know about.** Gates 1–4 are the five-seed means
that the manuscript's Table 6 reports (validity 0.330, 69 distinct). The uMLIP screen
was run on the *deposited archive* run, which gave 6,768 valid / 66 distinct / 64 novel
(`dgx/outputs_dgx/composition_cvae_generated_metrics.json`) — within the seed spread but
not the same run. Drawing the funnel entirely from the deposited archive instead gives
6,768 / 66 / 64 and ratios 2.1× / 178×. I chose the five-seed means so the figure
agrees with the manuscript's table; the alternative is a two-line change in
`a6_fig2_funnel.py`.

### `fig3_freebits.pdf` — posterior collapse, diagnosis and fix (3.30 × 4.75 in, `\columnwidth`)
Script `a6_fig3_freebits.py`.

| Element | File | Column |
|---|---|---|
| (a) deposited per-dimension KL, 32 values | `a8_baseline_pathology.json` | `latent.kl_per_dim_nats_sorted` |
| (a) "1 of 32 active, 0.17 nats total" | same | `latent.active_units_kl_gt_0.01`, `latent.total_kl_nats` |
| (a) corrected per-dimension KL, 32 values | `a8_sweep_all.csv` | `tag=fb0.5_b0.01`, `kl_per_dim_sorted` |
| (a) "32 of 32 active, 14.8 nats total" | same row | `active_units_kl_gt_0.01`, `total_kl_nats` |
| (b) λ sweep, 10 points | `a8_sweep_all.csv` | rows `loss=fixed & beta_max=0.01`; `free_bits_per_dim`, `test_composition_mse`, `gen_validity_rate_mean`, `gen_unique_valid_count_mean` |
| (b) deposited MSE reference 0.128 | `dgx/outputs_dgx/composition_cvae_test_metrics.json` | `composition_mse` = 0.12786 |
| (b) deposited 69-formula reference | `a8_headline_comparison.csv` | `arm=deposited_cvae`, `unique_valid_count_mean` |

The deposited model has no per-dimension λ, so it is drawn as two dotted reference
rules rather than as a sweep point at λ = 0 — putting it at λ = 0 would have been a
category error, since the corrected loss at λ = 0 is a *different* model
(`fb0.0_b0.01`, MSE 0.0793) that is also on the curve.

The nominal minimum of the deposited per-dimension KL is 1.55 × 10⁻⁶ nats, not zero,
so no value is clipped onto the log-axis floor; the grey band marks the
0.01 nats active-unit threshold and 31 of 32 dimensions lie in it.

### `fig4_energetics.pdf` — ΔE_dec distributions, CVAE vs random (3.30 × 4.45 in, `\columnwidth`)
Script `a6_fig4_energetics.py`. Empirical CDFs, not violins, because the claim is a
null and an ECDF shows the overlap without any smoothing choice to argue about.

| Element | File | Filter / column |
|---|---|---|
| distributions, 4 series | `a7_energetics_all.csv` | `mode=='full'`, non-null `decomposition_energy_eV_per_atom`; `calc` ∈ {`a7cv_full_chgnet`, `a7rn_full_chgnet`, `a7cv_full_mace`, `a7rn_full_mace`}; arm from `population` |
| medians, δ, CI, p | `a7_energetics_comparison.csv` | `mode=='full'`: `median_a`, `median_b`, `cliffs_delta`, `cliffs_delta_ci95`, `p_value` |
| % with ΔE_dec ≤ 0, Fisher p | `a7_energetics_proportions.csv` | `mode=='full'`: `p_a`, `p_b`, `fisher_p` |

Recomputing the medians and n from the raw column reproduces the comparison file
exactly: CHGNet 53 / 50 at +0.0861 / +0.08645; MACE 53 / 50 at +0.09958 / +0.095485;
fraction ≤ 0 = 0.0943 / 0.24 (CHGNet) and 0.0755 / 0.24 (MACE).

**Not implying a difference.** The two medians are marked on the same 0.5 rule with a
short stem each (offset only so both markers are visible where they coincide); no
shading, arrow or ordering device separates the curves; and the effect sizes appear as
a printed strip with their confidence intervals, with the footnote stating that all
intervals span zero. The x-axis is clipped at −0.30 eV atom⁻¹ so the bulk is legible,
and the single random candidate beyond it (−0.748 CHGNet, −0.726 MACE) is annotated in
place rather than dropped.

### `fig5_chemistry.pdf` — what the model does learn (3.30 × 4.35 in, `\columnwidth`)
Script `a6_fig5_chemistry.py`. Paired dot-and-interval plot, both arms on the same row.

| Element | File | Row / column |
|---|---|---|
| (a) four proportions, both arms, 95 % CIs, Fisher p | `a7_compare_proportions.csv` | `comparison=='cvae vs full random pool'`; axes `has_open_shell`, `has_stage1_good`, `passes_quality_gate`, `requires_reduced_cation`; columns `p_a`, `p_b`, `p_a_ci95`, `p_b_ci95`, `diff`, `fisher_p`, `n_a`=66, `n_b`=11,744 |
| (b) Zr / Al / Hf occurrence, CIs, odds ratios | `a7_stage1_enrichment.json` | keys `Zr`, `Al`, `Hf`: `p_a`, `p_b`, `p_a_ci95`, `p_b_ci95`, `odds_ratio`, `fisher_p`, `k_a`, `n_a`, `k_b`, `n_b` |

Values as drawn: open-shell 27.3 % vs 82.1 % (−54.8 pp, p = 4 × 10⁻²²); Hf/Zr/Al
53.0 % vs 18.7 % (+34.4 pp, p = 5 × 10⁻¹⁰); wide-gap gate 63.6 % vs 15.9 %
(+47.7 pp, p = 4 × 10⁻¹⁸); Ti³⁺/Sn²⁺ 36.4 % vs 8.3 % (+28.1 pp, p = 3 × 10⁻¹⁰);
Zr 8.4×, Al 4.2×, Hf 3.0×.

**Not overselling.** The model's own chemical failure — needing Ti³⁺/Sn²⁺ to charge
balance — is the fourth row of panel (a), drawn in exactly the same style and not set
apart, so the panel cannot be read as four wins. The intervals on the random arm are
invisibly narrow because n = 11,744; that is a real property of the data, and the
figure states both n values in the key.

### `fig6_stage1.pdf` — the corrected Stage-1 landscape (3.30 × 4.60 in, `\columnwidth`)
Script `a6_fig6_stage1.py`.

| Element | File | Column |
|---|---|---|
| (a) scatter, 200 points | `a1_corrected_top200.csv` | `gap_utility` vs `stability_utility`, coloured by `source_dataset`, open symbols where `stability_known == 0` |
| (a) Pareto front, 7 points | `a1_corrected_pareto.csv` | `stability_utility`, `gap_utility`, `formula` |
| (a) labelled compounds | `a1_corrected_top200.csv` | rows BaHfO₃, SrHfO₃, CaHfO₃, BaZrO₃, SrZrO₃, LaAlO₃, KTaO₃, CaSiYAlO₆ |
| (b) rank of CaSiYAlO₆ vs ρ | `a1_imputation_sensitivity.csv` | `rho`, `rank_CaSiYAlO6` (216 / 83 / 14 / 1 / 1) |
| (b) double perovskites in the top 20 | same | `double_perovskites_in_top20` (0 / 0 / 1 / 18 / 20) |
| (b) top-ranked formula per ρ | same | `top_formula` |

**Two honesty devices.** Double-perovskite records are drawn as *open* symbols sitting
on a dotted ρ = 0.5 rule, labelled as imputed, because their vertical coordinate is an
assumption rather than a measurement; and panel (b) shows exactly how much of the head
of the ranking that assumption controls.

**A stated limitation.** Panel (a) shows the 200 highest-scoring records, not the full
corrected pool of 1,890 (`a1_corrected_pool_summary.csv`, 1,048 + 444 + 398). The full
ranked pool is not among the deposited a1 result files — only `top200`, the ρ-sweep
`top50`s, the acquisition batch and the expert-review pack are — so the complete
landscape could not be plotted without re-running a1's pipeline, which I did not do.
The axis label and this report say "top 200"; if the authors want the full cloud, a1
would need to emit the ranked pool.

### `fig7_stage1_workflow.pdf` — Stage-1 screening workflow (3.30 × 4.85 in, `\columnwidth`)
Script `a6_fig7_stage1_workflow.py`. Matplotlib redraw of the former Visio export.

| Number | Source |
|---|---|
| 1,056 dielectric records, 757 after the Eg ≥ 1 eV gate | `a1_dielectric_legacy_ranked.csv` has 757 rows; `a1_REPORT.md` line 39 |
| Castelli 18,928 · Wolverton 4,914 · double perovskites 1,306 = 25,148 | `a1_REPORT.md` lines 40–42 and 167–169 (agent-computed) |
| 25,148 → 1,890 after the harmonised gate | `a1_corrected_pool_summary.csv`, Σ`candidates` = 1,048 + 444 + 398 |
| 7 Pareto records | `a1_corrected_pareto.csv`, row count |
| 20-record acquisition batch, greedy on 0.65 S + 0.35 d_min | `a1_corrected_acquisition_batch.csv`, row count; rule from the manuscript Methods |
| the four scalarisation rules | `a1_true_scoring_rules.csv` / manuscript Table 2 |

### `fig8_stage2_pipeline.pdf` — Stage-2 generation **plus the parallel random arm** (6.00 × 3.85 in, `0.85\textwidth`)
Script `a6_fig8_stage2_pipeline.py`. Matplotlib redraw of the former Visio export, with
the control arm now running beside the generator and both converging on a single filter
block that is drawn once, because there is only one code path.

| Number | Source |
|---|---|
| 25,148 records, 77 fractions + 2 conditions, 79→256→128, 32-dim latent, 32+2→128→256→77 | manuscript Methods §2.2 |
| filter steps (top-3 cations, max count 4, GCD, O 1–12, whitelist) | manuscript Algorithm 1 |
| reachable output set 129,205 | `a2_decoder_reachability.json`, `total_reachable_valid_formulas` |
| validity 0.330 ± 0.002, 69 formulas | `a2_random_baseline_summary.csv`, `arm=cvae` |
| validity 0.727 ± 0.002, 11,821 formulas | `a2_random_baseline_summary.csv`, `arm=rand_uniform41` |

---

## 2. What I could not do, and why

1. **The full Stage-1 landscape (Fig. 6a).** Only the top 200 ranked records are
   deposited; the complete 1,890-record corrected pool is not. I plotted and labelled
   the top 200 rather than reconstruct the pool myself. Re-running a1's scoring script
   with an "emit full ranked pool" flag would fix this.
2. **A genuine funnel for the random arm's last two gates (Fig. 2).** The random arm's
   uMLIP screen is a 50-candidate sub-sample of 331 reference-eligible candidates, not
   an exhaustive pass, so no honest count-based attrition exists there. I drew the bars
   hatched and below a rule, labelled the pass rate `n.a.`, and stated the restriction
   in the figure's own footnote. Extending the screen to all 331 would make those two
   gates real.
3. **Any per-candidate hull distance.** Not computable — every structure database was
   proxy-blocked in this environment (a4/a7 reports). Fig. 4 therefore plots
   decomposition energy into binary oxides and the axis says so; nothing in the figure
   claims stability.
4. **A λ-sweep figure with error bars (Fig. 3b).** `a8_sweep_all.csv` carries
   `n_seeds = 1` for every sweep row and `gen_validity_rate_std = 0` throughout, so the
   sweep is single-seed and I plotted it without intervals rather than draw zero-width
   ones that would imply precision the run does not have. Only the deposited and the
   two `a8_downstream_*` configurations were run over five seeds
   (`a8_headline_comparison.csv`). If the authors want intervals on Fig. 3b, the sweep
   needs re-running at more seeds.
5. **External-novelty content in any figure.** Absolute novelty is unestablished
   (every provider blocked, a3/AUDIT §E), so no figure plots it. Fig. 2's "novel" gate
   is labelled *absent from training table* in full, not "novel".

## 3. Cautions for the authors

* Include each figure at the width given in §1. These are built at final size; scaling
  Fig. 1 or Fig. 8 down to `\columnwidth` would put their type at ~4.5 pt.
* The manuscript's `\includegraphics[width=0.85\textwidth]{fig1_concept.pdf}` is
  already correct. The other seven are not yet referenced in
  `main_elsarticle.tex`; Figs. 2–6 want `width=\columnwidth` and Fig. 8
  `width=0.85\textwidth`, Fig. 7 `width=\columnwidth`.
* Fig. 2 and Fig. 8 both print validity 0.330 / 0.727 and 69 / 11,821. If Table 6 is
  ever re-run at a different seed count, both scripts read the same CSV and will follow
  automatically — but check them, because the numbers are also written into the
  manuscript prose in three places.
* Fig. 3's λ-sweep is single-seed (see §2.4). The manuscript's Table 8 is the same
  single-seed sweep, so figure and table agree, but neither should be described as
  carrying uncertainty.
* Nothing in this set was drawn from memory. Every script re-reads its result file at
  run time and prints the values it plotted to stdout, so `sh a6_make_all.sh` is also
  the audit trail.

## Restyle to the authors' Visio theme (12 Oct 2026)

All eight figures were restyled so the set matches the authors' original Visio schematics (`manuscript/figures/ceramic_*_rendered.pdf`); no plotted value or diagram content changed apart from the in-figure cross-references, which now use the manuscript's figure numbers (Fig. 4a, 7, 7b, 8).

* Font: Times New Roman metrics (Liberation Serif), STIX maths — as in the Visio exports.
* Palette: the Office theme sampled from those exports — outline `#4472C4`, pill fill `#DEE9F7`, teal highlight `#D9EDF3`, connector and CVAE colour `#1F497D`, random control `#4BACC6`, emphasis `#C0504D`, third series `#77933C`.
* Schematics (Figs. 1–3): rebuilt with `a6_visio.py` in the Visio grammar — white rounded containers with bold titles, light-blue pills, dashed teal groups with white boxes for outcomes, dashed frames, thick dark-blue connectors — and the original snake layouts.
* Fig. 6(b): two labels that collided with the validity curve were moved.

## Four-arm update (13 Oct 2026)

Five figures were moved from the two-arm comparison (CVAE vs random) to the a9 four-arm study: A random, B element statistics, C ion substitution, D corrected CVAE `fb0.5_b0.01`. The filenames are unchanged. Every script reads its numbers at run time and prints them to stdout. Figs. 1, 7 and 8 also assert the claims they print in words. Figs. 3, 6 and 7 were not changed.

**Palette (`a6_style.py`, new keys only).**
- A `#4BACC6` (aqua, unchanged), L\* 66.
- B `#7F7F7F` (grey), L\* 53.
- C `#4F6228` (Office olive green, darker 50 %), L\* 39. The suggested `#77933C` is L\* 57, only 4 units from the grey, and the two could not be told apart in greyscale.
- D `#1F497D` (dark blue), L\* 31.
- Deposited CVAE `#558ED5`. It is always dashed or hollow.
- Second cues that do not depend on colour: markers o / s / ^ / D, and line styles dotted / dashed / dash-dot / solid.
- `a6_visio.py` gains `pill_arm` (the arm's tint as fill, the arm colour as outline).

### `fig1_concept` [Fig. 1]
Panel (a) is unchanged. Panel (b) shows the four-arm ladder going into the identical filter. It contains no numbers. Its sentences are directions taken from the a9 results and asserted in the script:
- validity falls at every step and A has the most unique valid formulas (`a9_counts_summary.csv` `validity_rate_mean`, `unique_valid_count_mean`);
- open-shell content falls and the gate pass rate rises at every step, and D is highest on Ti³⁺/Sn²⁺ (`a9_compare_proportions.csv` `p_a`/`p_b`);
- C differs from D (p < 0.01) and A and B do not (p > 0.05) (`a9_energetics_pairwise_vs_D.csv`, `mode=full`, `cliffs_delta`, `mannwhitney_p`).

### `fig8_stage2_pipeline` [Fig. 3]

| Number | File / column |
|---|---|
| validity ± s.d. and distinct valid ± s.d., arms A–D | `a9_counts_summary.csv`: `validity_rate_mean/_std`, `unique_valid_count_mean/_std` |
| 2,200 parents, threshold 10⁻³ | `analysis/a9_fourarm/a9_arm_c_parents.json`: `n_parents`, `threshold` |
| λ = 0.5, β_max = 0.01 | `a8_sweep_all.csv`, `tag=fb0.5_b0.01`: `free_bits_per_dim`, `beta_max` |
| reachable set 129,205 | `a2_decoder_reachability.json`: `total_reachable_valid_formulas` |
| architecture, 25,148 records | manuscript Methods (unchanged) |

### `fig2_funnel` [Fig. 5]

| Element | File / column |
|---|---|
| Gates 1–4, counts ± s.d. | `a9_counts_summary.csv`: `valid_count_mean/_std`, `unique_valid_count_mean/_std`, `novel_unique_count_mean/_std`. Sampled = 20,000. |
| Pass rates | ratios of those means |
| A : D ratios | same file |
| Gate 5: % ΔE_dec ≤ 0 under CHGNet, Wilson CI | `a9_energetics_summary.csv`, `calc=chgnet, mode=full`: `frac_le0`, `frac_le0_wilson_lo/_hi`, `k_le0`, `n_with_reference` |
| Gate 6: novel and ≤ 0 under both potentials | `a9_energetics_novel_yield.csv`: `yield_novel`, `yield_novel_wilson_lo/_hi`, `k_le0_both_novel` |
| known-compound counts | `a9_energetics_summary.csv`: `n_in_training` |

Gates 5–6 are shown only as rates: they are hatched, below the dashed rule, and have no bars in panel (a).

### `fig5_chemistry` [Fig. 7]
- **(a)** `a9_compare_proportions.csv`, rows `D vs A`, `D vs B` and `D vs C`, on the axes `has_open_shell`, `has_stage1_good`, `passes_quality_gate` and `requires_reduced_cation`.
  - Arm D comes from `p_a`/`p_a_ci95`.
  - Arms A/B/C come from `p_b`/`p_b_ci95`.
  - The right-hand column is `diff`.
  - The largest `fisher_p` is 1.5×10⁻⁹, printed as ≤ 10⁻⁸.
- **(b)** `a9_element_enrichment.csv`, elements Zr/Al/Hf, using the same columns plus `odds_ratio`.
- The Ti³⁺/Sn²⁺ row is drawn like every other row. On this row D is the worst arm.

### `fig4_energetics` [Fig. 8]
Spinel-corrected files only. No `*_asrun` file is used.

| Element | File / column |
|---|---|
| ECDFs | `a9_energetics_all.csv`, `mode=full`, non-null `decomposition_energy_eV_per_atom`, `pot` ∈ {chgnet, mace}, `arm` ∈ {A, B, C, D, CVAE_dep} |
| known-compound markers on C | `a9_energetics_all.csv`: `in_training_table` |
| median, IQR, ≤ 0 % | `a9_energetics_summary.csv`: `median_dE` (asserted equal to the plotted median), `q25`, `q75`, `frac_le0` |
| δ vs D, CI, p | `a9_energetics_pairwise_vs_D.csv`, `mode=full`: `cliffs_delta`, `cliffs_delta_ci_lo/_hi`, `mannwhitney_p` |
| Kruskal–Wallis | `a9_energetics_kruskal.csv`: `kruskal_p`, `kruskal_ABD_p` |
| C novel median +0.008 (n = 26) | `a9_energetics_summary.csv`, C, `mode=full_novel` |

- **Known compounds in C.** The 23 training-table compounds of arm C are marked on C's curve. The script asserts that C's five lowest values are known compounds under both potentials.
- **Off-scale value.** One D value under CHGNet (+0.553) lies beyond the axis and is annotated where it would fall.

### Not plotted, and why
- **Gates 5–6 as counts (Fig. 5).** The 50-per-arm subsamples were drawn from different-sized eligible pools: A and B from the seed-123 runs, C and D from the pooled seeds 0–4. The figure therefore shows rates only. The eligible-fraction denominators for A and B are not stored in an a9 file.
- **The deposited CVAE in Figs. 5 and 7.** It is not an a9 arm, and its a7 chemistry numbers were computed against a different random pool. It appears only in Fig. 8, as a dashed reference, because a9 re-analysed it on the same footing as the four arms.
