"""
a3_families.py -- Substitution-family grouping (task 3).

Groups the 66 unique valid candidates into "substitution families": same
anonymised stoichiometry (sorted atom-count pattern, oxygen included) AND
the same multiset of periodic-table groups for the cation sites. Two
formulas land in the same family iff one can be obtained from the other by
swapping cations that occupy chemically equivalent (same-group) roles,
e.g. Al2ZrHfO7 and Al2ZrScO6-type patterns with matching shapes.

This directly tests Reviewer 1's observation that the top-10 list is
mostly permutations of ~8 common cations across a handful of templates.
"""
from __future__ import annotations

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from pymatgen.core import Element

sys.path.insert(0, _CCID_ROOT + "/dgx/code")
from composition_cvae_common import parse_formula  # noqa: E402

RESULTS = Path(_CCID_ROOT + "/results")

df = pd.read_csv(RESULTS / "a3_stage_nn.csv")


def group_of(el: str) -> int:
    return Element(el).group


def family_key(formula: str):
    counts = parse_formula(formula)
    cation_counts = {el: c for el, c in counts.items() if el != "O"}
    o_count = counts.get("O", 0)
    # cation-group multiset: (count, periodic group) pairs, sorted
    cation_group_pairs = tuple(sorted((int(round(c)), group_of(el)) for el, c in cation_counts.items()))
    stoich_shape = tuple(sorted([int(round(v)) for v in counts.values()], reverse=True))
    return (stoich_shape, cation_group_pairs, int(round(o_count)))


df["family_key"] = df["canonical_formula"].apply(lambda f: repr(family_key(f)))
df["family_key_raw"] = df["canonical_formula"].apply(family_key)

families = defaultdict(list)
for _, row in df.iterrows():
    families[row["family_key"]].append(row["canonical_formula"])

family_ids = {k: f"F{idx+1:02d}" for idx, k in enumerate(sorted(families, key=lambda k: -len(families[k])))}
df["family_id"] = df["family_key"].map(family_ids)

n_formulas = len(df)
n_families = len(families)

print(f"unique valid formulas: {n_formulas}")
print(f"distinct substitution families: {n_families}")
print()
sizes = Counter(len(v) for v in families.values())
print("family size distribution (family_size: n_families):", dict(sorted(sizes.items())))
print()
for k, members in sorted(families.items(), key=lambda kv: -len(kv[1])):
    if len(members) > 1:
        print(family_ids[k], "shape/groups:", k, "->", members)

df.to_csv(RESULTS / "a3_families.csv", index=False)

summary = {
    "n_unique_valid_formulas": n_formulas,
    "n_substitution_families": n_families,
    "family_size_distribution": {str(k): v for k, v in sorted(sizes.items())},
}
import json
json.dump(summary, open(RESULTS / "a3_families_summary.json", "w"), indent=2)
print("\nwrote a3_families.csv and a3_families_summary.json")
