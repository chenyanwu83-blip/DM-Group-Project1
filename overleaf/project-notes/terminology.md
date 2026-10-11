# Shared terminology for the report

| Use consistently | Meaning / reporting rule |
| --- | --- |
| `credit-g` | UCI Statlog German Credit Data, dataset A |
| `sick` | The selected `sick` subset of UCI Thyroid Disease, dataset B; do not use repository-wide counts as counts for this file |
| Positive class: `bad` | credit-g minority label; verify probability-column order in each model |
| Positive class: `sick` | sick minority label; `negative` does not mean universally healthy |
| Plain | Classifier without an imbalance intervention |
| SMOTE | Synthetic minority oversampling, applied only to training folds |
| Cost-sensitive | Model trained with an explicit error-cost or class-weight setting; give the exact setting |
| Threshold adjustment | Decision threshold fixed using development data before holdout evaluation |
| Precision / Recall / F1 | Report for the named positive class; include confusion matrix counts |
| ROC-AUC | Area under the receiver operating characteristic curve; identify the positive class |
| 10-fold CV | Stratified 10-fold cross-validation, with seed 42 in the group plan |
| Holdout | Separate stratified 70:30 split, seed 42; evaluate only after settings are fixed |
| Rule support | Fraction of all records containing antecedent and consequent |
| Rule confidence | Fraction of antecedent records also containing the consequent |
| Rule lift | Confidence divided by the consequent's overall frequency; values above 1 indicate positive association |

Keep experimental findings out of this glossary until measured. Use `association` rather than `causes` when describing Apriori rules.
