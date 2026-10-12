# E2 credit-g Apriori data preparation — complete under A5 v1.0 and A's clarification

## Source and specification

- Input: `DM-Group-Project1/data/raw/credit-g.csv`, SHA-256 `d0baf9fddd41e5a6af0ba84e6037a415f11384dd40c494186563a2d3b68f5c25`.
- A5: `a5_source_spec_creditg.md`, copied from GitHub commit `64a17e4ce68a892ec7a9256afb3e337f722258cf`; SHA-256 `561d54508918b09973867d487b878fe57bc5f0bb56cf49f011d6422bd8b48ccd`.
- Four-field clarification: `a5_four_fields_clarification.md`, recording the answer forwarded by the user on 2026-10-12; SHA-256 `39d7a00130046f577cc298ee1813200e9c7709019ddb4084d7ffba21718f583a`.
- A5 specifies `KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')` for `duration`, `credit_amount`, and `age`; nominal columns remain unchanged. This is for E's Apriori table only, not the supervised classification pipelines.
- Implementation uses scikit-learn 1.9.1, matching the group repository's pinned version. The script checks both input and A5 specification hashes before transforming.

## What the script does

1. Read and validate 1,000 records, 21 original columns, zero missing entries, and the 700 `good` / 300 `bad` class balance.
2. Fit A5's `KBinsDiscretizer` to the three named continuous fields and label the resulting ordinal codes `q1_of_5` through `q5_of_5`. Save every fitted edge and bin count in `output/conversion_record.json`.
3. Keep nominal values unchanged. A confirmed that the four other numeric-coded fields (`installment_commitment`, `residence_since`, `existing_credits`, `num_dependents`) retain their original 1–4 or 1–2 levels as separate `column=value` Apriori items. They are neither binned again nor excluded.
4. Save the 21-column categorical table, then turn each `column=value` into a Boolean item. The `class=bad` item is included to discover association rules with bad credit as the consequent; it must not be used as a predictor in supervised modeling.

## Verified output

- `output/credit_g_discretized.csv`: 1,000 × 21; SHA-256 `c4b88faef799110700ce6155c5d6dbd9c06c5af5ad9c8d5dbc8cd169ace9b2a5`.
- `output/credit_g_transactions.csv`: 1,000 × 85; SHA-256 `8a067ddcba7c3885aba2883368ff69d1b04a27af83a473b67a510ab23fb38747`.
- Every row has exactly one true item per original field, hence 21 true items. `class=bad` occurs in exactly 300 rows.
- A separate rerun with the group's pinned NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1, and scikit-learn 1.9.1 produced byte-identical hashes for both CSV outputs.
- Fitted edges: duration `[4, 12, 15, 24, 30, 72]`; credit_amount `[250, 1262, 1906.5, 2853.5, 4726, 18424]`; age `[19, 26, 30, 36, 45, 75]`. See the JSON record for counts.
- Interpret scikit-learn's interval labels using its boundary rule: for example, duration `q1_of_5` means `[4, 12)`, `q2_of_5` means `[12, 15)`, and the last bin includes the maximum value 72.

## Difference from the provisional version and group document follow-up

The provisional script used `pandas.qcut(q=5)`. A5 specifically names scikit-learn's `KBinsDiscretizer`, so the code and both output files were regenerated. The provisional transaction matrix SHA-256 was `d0645432307cb874fd3fc20a5c7c432a79dd42c73826b967d1925c1415856a95`; it must not be used for E3. Even where an edge is identical, values on an edge can enter a different bin under the two APIs; for duration, the first bin has 180 records with A5's method rather than 359 in the provisional version.

The four fields have only 2–4 levels, so another five-bin transform could merge existing levels without adding useful detail. Rare values remain in the transaction table and are removed by E3's minimum-support threshold when appropriate: `existing_credits=4` occurs in 6/1000 rows (0.6%), below 5%. The source CSV has **921** distinct `credit_amount` values; the user-forwarded explanation said 923, so use the verified 921 figure in the report and presentation.

E2's local transformation and audit are complete. A confirmed the four-field wording, and [PR #2](https://github.com/chenyanwu83-blip/DM-Group-Project1/pull/2) was merged into `main` on 2026-10-12. The PR added five lines to the GitHub A5 v1.0 file without changing any original parameters. The `status` value in the local JSON and conversion record describes the state when these outputs were generated; the data transformation itself has not changed since the merge. The E2 script, outputs, and this note are published under `src/e_apriori/` in the group repository.
