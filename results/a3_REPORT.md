# A3 — Novelty re-assessment of Stage-2 CVAE candidates (MTCOMM-D-26-04679)

Agent: a3_novelty. All numbers below come from scripts in
`analysis/a3_novelty/` (`a3_analysis.py`, `a3_families.py`,
`a3_finalize.py`) run against the deposited artifacts in
`dgx/outputs_dgx/` — nothing here is looked up or estimated by hand.

**Starting point (confirmed, matches the audit brief):** of the archived
20,000-sample run, 6,768 rows are `valid == True`. Deduplicating on
`canonical_formula` gives **66** unique valid formulas, of which **64** are
flagged `novel == True` by the paper's own test — exact string match against
`composition_cvae_training_records.csv` (96.97%, not the manuscript's 71/97.2%,
because the manuscript's number comes from a different, unreproduced run —
see agent a2's report). This report treats 66/64 as ground truth and asks
what "novel" should actually mean.

---

## Task 1 — External database cross-check

**Result: this could not be completed, and I am reporting that rather than
guessing.** Full connectivity log:

| Provider | Path tried | Outcome |
|---|---|---|
| Materials Project (OPTIMADE gateway) | `optimade.materialsproject.org` | Direct HTTPS: blocked by the sandbox egress proxy (`403` at CONNECT, host not on the allow-list). Via the `WebFetch` tool (a separate network path): `/v1/info` itself returned HTTP 403 — the OPTIMADE gateway appears to require authentication/API key, consistent with the audit brief's note that MP needs a key we don't have. |
| OQMD | `oqmd.org/optimade` | Direct HTTPS: blocked (403 at CONNECT). Via `WebFetch`: blocked by the site's own `robots.txt`. |
| AFLOW | `aflow.org/API/optimade` | Direct HTTPS: blocked (403 at CONNECT). Via `WebFetch`: `/v1/info` succeeded (valid OPTIMADE metadata, versions 1.0.0/1.1.0), but every call to `/v1/structures` — with or without a filter, with or without `page_limit` — returned HTTP 500 from the server. Reproduced repeatedly; this is a provider-side fault in that endpoint reachable from this tool, not a query-syntax issue. |
| COD (Crystallography Open Database) | `www.crystallography.net/cod/optimade` | Direct HTTPS: blocked (403 at CONNECT). Via `WebFetch`: blocked by `robots.txt`. |
| Alexandria | `alexandria.icams.rub.de/pbe/v1` | Direct HTTPS: blocked (403 at CONNECT). Via `WebFetch`: `/v1/info` and an **unfiltered** `/v1/structures?page_limit=2` both succeeded (confirms 5,779,351 structures reachable, e.g. entries `agm000999956`/`agm000999957`). But every attempt to add *any* query parameter — `filter=chemical_formula_reduced="..."`, `filter=nsites=2`, `sort=nsites`, `page_offset=0`, even syntactically trivial ones — returned HTTP 500, reproducibly, across many independent tries with different parameter names and formula strings. No targeted per-composition lookup could be executed. |
| `providers.optimade.org` (the provider registry itself) | — | Blocked at CONNECT (403), so I could not even enumerate the full current provider list programmatically; the list above is the standard set named in the task plus what I could independently confirm as live. |

So: **zero of the five target providers could be queried for a
candidate-specific match.** Two (AFLOW, Alexandria) are demonstrably live and
reachable for *unfiltered* requests through the `WebFetch` path, which rules
out "the whole internet is unreachable" — the failure is specifically in
issuing a filtered/parameterised OPTIMADE query from this sandbox, either
because the egress proxy strips/mangles query strings passed through that
tool, or because of a provider-side bug triggered by any filter. I did not
fabricate a workaround (e.g., scraping the unfiltered structure stream and
filtering client-side over 5.7M records is infeasible in the time available
and was not attempted).

As a **non-systematic, clearly-labelled supplement**, I ran two general
web searches (not a database query) for two of the highest-scoring
candidates, `Al2HfO7Zr`/`Al2ZrHfO7` and `Al2BaHfO6`/`BaAl2HfO6`: neither
returned any indexed literature or database hit for those exact formulas
(only unrelated Al₂O₃ and BaO–Al₂O₃ pages came up). This is weak evidence
of absence, not confirmation — a negative web search does not rule out an
entry existing in a structured database we could not query.

**Consequence for the three novelty levels the task asked for:**
- (i) absent from the training set: **computable, done** — see Task 2/the
  per-candidate CSV (`novelty_level_i_absent_from_training_exact_match`);
  64/66 = 97.0%.
- (ii) absent from every external database reachable: **not verifiable**
  from this environment. Reported as `not_verifiable` in the per-candidate
  CSV and the summary, not as "yes."
- (iii) no entry in the same chemical system at all: **computable only
  against the training set** (no external check available). Against
  training only, 53/66 (80.3%) of candidates use an element combination
  that never appears anywhere in the 12,150 unique training formulas; the
  other 13/66 sit in a chemical system training data already covers (e.g.
  `CaO4SrZr`'s element set {Ca,O,Sr,Zr} also appears, at a different
  stoichiometry, in the training table).

**Bottom line for Task 1: the paper's novelty claim cannot be upgraded to
"externally verified" with the tools available here, and the revision
should not claim external-database verification it does not have.**

---

## Task 2 — Nearest-neighbour analysis

Two metrics were computed, both candidate vs. the 12,150 unique training
formulas:

1. **Element Mover's Distance (ElMD)**, magpie element-feature weights,
   installed via `pip install --break-system-packages ElMD` (succeeded,
   version 0.5.15). Note for reproducibility: the package's numba-JIT
   network-simplex solver fails to compile under the installed
   numpy/numba pair (`AssertionError` inside `numba.core.typeinfer`); all
   ElMD distances here were computed with `NUMBA_DISABLE_JIT=1` (pure
   interpreted mode — correct, just slower, ~13 minutes for all 66×12,150
   pairs). Also needed `setuptools<81` reinstalled because ElMD imports
   the now-removed `pkg_resources`.
2. **Euclidean distance in element-fraction space** over the model's own
   77-element vocabulary (`composition_cvae_metadata.json`) — this is
   exactly the metric `generate_composition_candidates.py` already uses
   for its `nearest_known_distance` column, recomputed here against the
   full training set as a reproducibility cross-check.

Distribution (n=66 unique valid candidates):

| Metric | min | p25 | median | p75 | max | mean |
|---|---|---|---|---|---|---|
| ElMD (magpie) | 0.00 | 65.06 | 83.03 | 104.36 | 179.66 | 86.53 |
| Euclidean (element-fraction, 77-dim) | 0.00 | 0.128 | 0.145 | 0.166 | 0.200 | 0.142 |

Both metrics agree on the two exact duplicates: `CaO6SrZr2` and `CaO6SnZr2`
have distance 0 by both metrics and are the two candidates the paper's own
flag already marks `novel == False` — a useful sanity check that the
pipeline is internally consistent.

The two metrics broadly agree on *ranking* (Spearman correlation not
separately re-verified here, but the same handful of candidates — `Al2LaO6Sc`,
`Al2BaO6Zr`, `Al2BaHfO6` — sit closest to training analogues under both).
For the top-10-by-score candidates, nearest training-set neighbour under
ElMD:

| Formula | Score | ElMD NN distance | Nearest training formula |
|---|---|---|---|
| Al2BaHfO6 | 0.884 | 30.41 | AlBaHfLaO6 |
| Al2LaO6Sc | 0.881 | 5.16 | Al2LaO6Y |
| Al2HfO7Zr | 0.877 | 75.53 | HfMgO6SnZr |
| Al2BaO6Zr | 0.847 | 24.00 | AlLaMgO6Zr |
| AlLaO5Zr | 0.775 | 60.96 | AlLaO6SrZr |
| AlBa2LaO5 | 0.759 | 64.96 | BaLaO3 |
| AlBaLaO4 | 0.738 | 49.94 | AlLaO3 |
| HfO7SrZr2 | 0.722 | 116.53 | CaHfO6SrZr |
| CaO4SrZr | 0.721 | 77.68 | CaGeO6SrZr |
| AlHfO5Sc | 0.696 | 89.06 | AlCaHfO6Y |

`Al2LaO6Sc` is a striking case: it is one atom-substitution away
(Sc↔Y, same periodic group) from a training-set compound `Al2LaO6Y`, at
ElMD distance 5.16 — the smallest non-zero distance among the top 10 — while
still passing the paper's exact-string novelty test simply because Sc≠Y.
This is exactly the kind of case a same-group substitution check catches
and exact-match novelty misses.

Full per-candidate distances (and nearest training formula) are in
`a3_per_candidate_novelty.csv`.

---

## Task 3 — Substitution families

Method: each of the 66 unique valid formulas was reduced to a
**family key** = (sorted atom-count pattern including oxygen, multiset of
(count, periodic-table group) pairs for the cation sites, oxygen count).
Two formulas land in the same family exactly when one can be produced from
the other by swapping cations that fill the same structural role and sit
in the same periodic-table group (e.g. Zr↔Hf, both group 4; Sc↔Y, both
group 3) at the same stoichiometric position — this operationalises
Reviewer 1's "single same-group substitution" observation.

**Result: 66 unique formulas fall into 46 distinct substitution families.**
Family-size distribution: 32 families are singletons (one formula each,
i.e. genuinely un-paired within this candidate set); 12 families have 2
members; 1 family has 4 members; 1 family has 6 members. The two largest
families:

- **F01** (shape 5-1-1-1 + O5, cation groups {3,4,13}): `AlO5ScZr,
  AlHfO5Sc, AlLaO5Zr, GaO5ScTi, AlO5ScTi, GaO5TiY` — 6 members, all an
  M(III)-M(IV)-M(III/13) + O5 template with Al/Ga (group 13) and
  Sc/Ti/Zr/La/Y (groups 3–4) permuted across the same two slots.
  These are the formulas Reviewer 1's Al2-{Zr,Hf,Sc,La,Ba} observation was
  pointing at.
- **F02** (shape 4-1-1-1 + O4, cation groups {2,3,13}): `AlBaO4Sc,
  BaGaLaO4, AlBaLaO4, BaGaO4Sc` — 4 members, Al/Ga in the group-13 slot,
  Sc/La in the group-3 slot, Ba fixed.

So the honest count of "how many chemically distinct things the generator
actually explored" is **46, not 69 (and not 66)**: roughly 30% of the
apparently-unique output is same-group cation substitution on a shared
stoichiometric skeleton, concentrated in two families (F01, F02) that
together account for 10 of the 66 formulas. The remaining 32 formulas each
sit alone in their own family under this definition — real, if narrow,
diversity. Full family membership is in `a3_families.csv` /
`a3_per_candidate_novelty.csv` (`family_id` column).

---

## Deliverables

- `a3_per_candidate_novelty.csv` — one row per unique valid candidate:
  paper's `novel` flag, level-(i)/(ii)/(iii) novelty flags, ElMD and
  Euclidean nearest-neighbour distance and nearest training formula,
  `family_id`, and per-provider external-DB columns (all `not_verifiable`,
  with the reason repeated in `external_db_check_note`).
- `a3_summary.json` / `a3_summary_flat.csv` — headline numbers.
- `a3_families.csv` / `a3_families_summary.json` — family grouping detail.
- `a3_stage_nn.csv` — intermediate (nearest-neighbour stage only).
- `a3_table_top10.tex`, `a3_table_families.tex`,
  `a3_table_novelty_levels.tex` — booktabs bodies for `elsarticle`.
- Scripts (kept, re-runnable): `a3_analysis.py`, `a3_families.py`,
  `a3_finalize.py` in `analysis/a3_novelty/`.

## Suggested abstract sentence

> Of the 66 unique, charge-balanced compositions the model generated,
> 64 (97.0%) do not exactly match any training-set formula, but they
> collapse into only 46 chemically distinct substitution families once
> same-group cation swaps on shared stoichiometric templates are
> accounted for, and 53 of the 66 (80.3%) involve an element combination
> absent from the training data entirely; whether any of these
> compositions are novel relative to the broader literature could not be
> established because no external crystallographic database (Materials
> Project, OQMD, AFLOW, COD, Alexandria) could be queried from our
> compute environment, so "novelty" here should be read strictly as
> novelty with respect to our own training corpus, not as novelty with
> respect to the field.

## Cautions for the authors

1. Do not publish "97.2% novel" (or even our reproduced 96.97%) as a
   field-level novelty claim — it is a training-set-relative number only;
   the manuscript text should say so explicitly rather than implying
   database verification.
2. If a real OPTIMADE/Materials-Project cross-check is wanted before
   resubmission, it needs to be run from a network environment that (a)
   has an allow-listed path to at least one of these five hosts, and
   (b) — for Alexandria and AFLOW specifically — either finds a working
   filtered-query path (the servers 500 on any filter through the tool
   used here) or downloads/queries a local snapshot instead of the live
   API.
3. Reviewer 1's substitution-family critique is correct and, if anything,
   understated for the top of the ranking: the two top-scoring families
   (F01, F02) alone cover 10 of the 66 unique candidates and are built
   from swapping Al/Ga, Sc/Ti/Zr/La/Y and Ba/nothing on a fixed
   stoichiometric skeleton.
4. The ElMD nearest-neighbour numbers (magpie feature space, raw units,
   not normalised to [0,1]) are not directly comparable in magnitude to
   the paper's existing Euclidean `nearest_known_distance` column; report
   them as a separate, complementary metric rather than replacing it,
   since they agree in ranking on the cases checked but are not on the
   same scale.
