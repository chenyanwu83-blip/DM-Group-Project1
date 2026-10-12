# A5 Apriori four-field clarification (2026-10-12)

Source: the user forwarded the response to the question sent to A. The GitHub `configs/spec_creditg.md` file at commit `64a17e4` still lacks this explicit sentence; this local record does not change the group's published specification.

For the Apriori transaction table, retain `installment_commitment`, `residence_since`, `existing_credits`, and `num_dependents` at their original discrete levels. Encode each as `column=value` (for example, `installment_commitment=4`). Do not apply additional bins or exclude these fields. The three continuous fields named by A5 are the only ones passed to `KBinsDiscretizer`.

Checked against the input CSV (1,000 rows):

| Field | Exact level counts |
|---|---|
| installment_commitment | 1:136; 2:231; 3:157; 4:476 |
| residence_since | 1:130; 2:308; 3:149; 4:413 |
| existing_credits | 1:633; 2:333; 3:28; 4:6 |
| num_dependents | 1:845; 2:155 |

These fields already have only 2–4 distinct levels, so each level is directly usable as an Apriori item. A rare level such as `existing_credits=4` has support 6/1000 = 0.6% and will be removed by E3 when minimum support is at least 5%, without deleting the field in advance. The user-forwarded explanation said credit_amount had 923 unique values; the project input CSV and A's A2 field dictionary both show **921**. Use 921 in E2 notes and presentation.

Group document follow-up: A/team may add the clarification sentence to `configs/spec_creditg.md` under the M2 change process. No GitHub file was edited or uploaded by E.
