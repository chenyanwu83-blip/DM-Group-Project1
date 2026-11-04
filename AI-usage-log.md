# AI Usage Disclosure

This project used AI-assisted development tools (ChatGPT, Claude, GitHub Copilot and similar) as supplementary aids for code scaffolding, debugging, and English language polishing. All research decisions — experimental design, method selection, data interpretation, and conclusions — were made by team members. AI tools did not fabricate or modify any experimental data, generate unverified conclusions, or substitute for our understanding of the methods. Every reported number traces to an actual script in this repository and can be reproduced.

## What AI was NOT used for

- **Fabricating or modifying experimental data.** All results come from scripts in `src/` that were executed on the real datasets. The SHA-256 hash chain from raw data to result files is intact and archived in `results/split_manifest.json`.
- **Generating unverified conclusions.** Every claim in the report was checked against the archived outputs in `results/`. The D8 audit cross-referenced all 84 result units (1,024 check items) with zero failures.
- **Replacing our understanding of the methods.** Every team member can explain the full pipeline, interpret all metrics, and answer methodology questions during the presentation.

## Per-member usage log

### A — HU Zhongyuan (credit-g data owner)

| Phase | Tool | Used for | My verification |
|---|---|---|---|
| Data audit (A1–A3) | ChatGPT | Pandas code scaffolding for schema and quality checks | All code executed; shape, class distribution, hash independently confirmed against raw file |
| EDA figures (A4) | ChatGPT | Matplotlib boilerplate for histograms and bar charts | Visual output checked against value counts; five-number summaries hand-verified |
| Encoding ablation (A6) | ChatGPT | Cross-validation loop structure; paired t-test syntax | p-values recomputed manually for F1 (0.021) and AUC (0.008); attribution control (ordinal + scaler) run separately to isolate encoding effect |
| Experiment runs (A7/A7x) | Claude | Script debugging when ColumnTransformer threw errors | 16 credit-g configs + 4 ensemble configs re-run; outputs match archived CSVs |
| Report sec2/sec3a | Claude | English prose drafting; grammar polishing | Every statistic traced to results/ CSV; provenance details checked against UCI page |

### B — LING Tong (sick data owner)

| Phase | Tool | Used for | My verification |
|---|---|---|---|
| Data pipeline (B1) | Claude | CSV conversion script from Weka ARFF; SHA-256 hash verification | Hash confirmed against split_manifest.json; row count (3,772) and column count (30) independently checked |
| Label and quality audit (B2/B3) | Claude | Pandas code for missing-value audit; measured-pair consistency check | 6,064 missing cells hand-counted in a spreadsheet sample; 6 lab pairs zero-mismatch result independently reproduced |
| Imputation ablation (B4) | Claude | Cross-validation loop; metric calculation debugging | Three treatments re-run; F1/AUC values match archived b4_missing_comparison.csv to 4 decimal places |
| Ensemble study (B7x) | Claude | Bagging/AdaBoost pipeline setup; imblearn syntax | 4 configs × 2 protocols re-run; 24 result files + 8 pred files match D4 audit |
| Report sec3b | Claude | English prose drafting; medical terminology cross-referencing | sick-euthyroid definition corrected after A9 peer review, verified against UCI thyroid.theory; all numbers cross-checked with D8 audit (0 FAIL) |

### C — CHEN Yanwu (modelling lead)

| Phase | Tool | Used for | My verification |
|---|---|---|---|
| Pipeline architecture (C1–C2) | ChatGPT | imblearn Pipeline boilerplate; ColumnTransformer configuration | Pipeline structure verified via print(pipe); 16/16 AUC equivalence check (threshold ≡ plain) as machine-level leak detector |
| Statistical tests (C5/C11) | ChatGPT | Nadeau–Bengio correction formula reference; scipy syntax | Formula checked against original paper (Nadeau & Bengio 2003); t and p values independently recomputed, matching table to 1e-12 |
| Run log and README (C6/C8/C10) | Claude | Documentation organisation; wording | Content sourced from actual run outputs; D co-signed the leak-check |
| Report sec4/sec5 | Claude | LaTeX debugging; English polishing | ~290 numeric assertions traced back to frozen results/ assets |

### D — CHEN Yinan (evaluation lead)

| Phase | Tool | Used for | My verification |
|---|---|---|---|
| Evaluation protocol (D1–D2) | ChatGPT | evaluate.py function structure; metric formula verification | Each metric (accuracy, precision, recall, F1, AUC, kappa, balanced accuracy, macro-F1) hand-checked against sklearn documentation |
| Full result audit (D3–D4) | Claude | Pivot table code for 84-unit summary; audit script structure | 1,024 check items independently recomputed from fold-level data; 0 failures |
| Core figures (D5) | ChatGPT | Matplotlib styling for ROC/PR curves and confusion matrices | Figure values verified against source CSVs; CVD-safe colours validated |
| Report sec6/sec7 | Claude | English phrasing for results interpretation | Every table cross-checked against results/; Holm correction counts verified by C |

### E — ZHANG Yijie (unsupervised and integration lead)

| Phase | Tool | Used for | My verification |
|---|---|---|---|
| Apriori analysis (E2–E4) | ChatGPT | mlxtend apriori API usage; rule filtering logic | Rules re-run with fixed parameters (min_support, min_confidence); top rules hand-checked |
| Report integration (E6) | Claude | LaTeX compilation troubleshooting; reference formatting | Final PDF reviewed by all five members before submission |
| Presentation materials | ChatGPT | Slide structure suggestions | Content verified against report and archived results |

---

**Summary.** AI tools served as development assistants (code hints, grammar checking, documentation lookup). The intellectual work — choosing what to test, interpreting why results differ across datasets, identifying the winner-flip phenomenon, and drawing conclusions — was done by the team. All five members can explain any part of the pipeline and answer methodology questions during the defense.

*Last updated: 2026-11-04*
