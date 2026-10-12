# E2 credit-g Apriori preparation

This folder contains E2 code, the confirmed A5 inputs, the discretized credit-g table, the Boolean transaction table, and the conversion record. A approved the four-field clarification in [PR #2](https://github.com/chenyanwu83-blip/DM-Group-Project1/pull/2).

From the repository root, with the pinned `requirements.txt` packages installed:

```bash
python src/e_apriori/prepare_credit_g.py --input data/raw/credit-g.csv --spec src/e_apriori/discretization_spec.json --output-dir src/e_apriori
```

The outputs are 1,000×21 and 1,000×85. Every transaction row has 21 true items. See `conversion_notes.md` and `conversion_record.json` for edges, class counts, and SHA-256 hashes. The A5 source and clarification files are historical snapshots used for reproducibility; their pending-language predates the PR merge. The published conversion record normalizes four local paths to repository-relative paths without changing data or numerical results.
