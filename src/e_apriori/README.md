# E2 credit-g Apriori preparation

This folder contains E2 code, the confirmed A5 inputs, the discretized credit-g table, the Boolean transaction table, and the conversion record. A approved the four-field clarification in [PR #2](https://github.com/chenyanwu83-blip/DM-Group-Project1/pull/2).

From the repository root, with the pinned `requirements.txt` packages installed:

```bash
python src/e_apriori/prepare_credit_g.py --input data/raw/credit-g.csv --spec src/e_apriori/discretization_spec.json --output-dir src/e_apriori
```

The outputs are 1,000×21 and 1,000×85. Every transaction row has 21 true items. See `conversion_notes.md` and `conversion_record.json` for edges, class counts, and SHA-256 hashes. The A5 source and clarification files are historical snapshots used for reproducibility; their pending-language predates the PR merge. The published conversion record normalizes four local paths to repository-relative paths without changing data or numerical results.

# E4 可解释性对照

`e4_interpretability_comparison.md` 是 E3 六条规则与 C3 credit-g plain 决策树路径、随机森林重要性的并排对照及报告素材。`e4_extract_tree_rf.py` 按 C 的模型配置和固定 700/300 行索引重建两个 plain 模型，核对保存的留出预测后生成 `e4_decision_tree_paths.txt` 与 `e4_rf_feature_importance.csv`。从仓库根目录执行 `PYTHONPATH=. python src/e_apriori/e4_extract_tree_rf.py`。
