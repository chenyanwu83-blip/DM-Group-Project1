from pathlib import Path
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from hashlib import sha256

script_dir = Path(__file__).parent
data_path = script_dir / "credit_g_transactions.csv"
if not data_path.exists():
    data_path = script_dir / "output" / "credit_g_transactions.csv"
output_dir = script_dir
expected_sha256 = "8a067ddcba7c3885aba2883368ff69d1b04a27af83a473b67a510ab23fb38747"
actual_sha256 = sha256(data_path.read_bytes()).hexdigest()

if actual_sha256 != expected_sha256:
    raise ValueError("E2 交易表版本不符，停止 E3")

print("E2 输入版本核对通过")
transactions = pd.read_csv(data_path)

print("表格大小：", transactions.shape)
print("各列的数据类型：")
print(transactions.dtypes.value_counts())
print("bad 的笔数：", transactions["class=bad"].sum())

itemsets_030 = apriori(
    transactions,
    min_support=0.30,
    use_colnames=True,
    max_len=3,
)

print("支持度至少 30% 的频繁项集数：", len(itemsets_030))
print(itemsets_030.head(10).to_string(index=False))

scan_rows = []

for threshold in [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]:
    itemsets = apriori(
        transactions,
        min_support=threshold,
        use_colnames=True,
        max_len=3,
    )
    trial_rules = association_rules(
        itemsets, metric="confidence", min_threshold=0.0
    )
    trial_bad = trial_rules[
        trial_rules["consequents"] == frozenset({"class=bad"})
    ]
    scan_rows.append({
        "min_support": threshold,
        "frequent_itemsets": len(itemsets),
        "bad_rules": len(trial_bad),
        "bad_rules_conf_050": int((trial_bad["confidence"] >= 0.50).sum()),
    })

scan_table = pd.DataFrame(scan_rows)
scan_path = output_dir / "e3_support_scan.csv"
scan_table.to_csv(scan_path, index=False)
print(scan_table.to_string(index=False))

itemsets_005 = apriori(
    transactions,
    min_support=0.05,
    use_colnames=True,
    max_len=3,
)

rules = association_rules(
    itemsets_005,
    metric="confidence",
    min_threshold=0.0,
)

bad_rules = rules[
    rules["consequents"] == frozenset({"class=bad"})
].copy()

print("生成的全部规则：", len(rules))
print("结论为 bad 的规则：", len(bad_rules))

bad_rules["condition_count"] = (
    bad_rules["antecedent support"] * len(transactions)
).round().astype(int)

bad_rules["bad_count"] = (
    bad_rules["support"] * len(transactions)
).round().astype(int)

ranked_rules = bad_rules.sort_values(
    ["confidence", "support"],
    ascending=[False, False],
)

print(
    ranked_rules[
        ["antecedents", "condition_count", "bad_count",
         "support", "confidence", "lift"]
    ].head(10).to_string(index=False)
)

bad_rules["num_conditions"] = bad_rules["antecedents"].apply(len)

single_rules = bad_rules[bad_rules["num_conditions"] == 1]

print("单条件规则数：", len(single_rules))
print("双条件规则数：", len(bad_rules) - len(single_rules))

print(
    single_rules.sort_values("confidence", ascending=False)[
        ["antecedents", "condition_count", "bad_count",
         "support", "confidence", "lift"]
    ].head(10).to_string(index=False)
)

print("confidence 至少 50%：",
      (bad_rules["confidence"] >= 0.50).sum())
print("confidence 至少 60%：",
      (bad_rules["confidence"] >= 0.60).sum())

shortlist = ranked_rules[ranked_rules["confidence"] >= 0.50]

print(shortlist[
    ["antecedents", "condition_count", "bad_count",
     "support", "confidence", "lift"]
].to_string(index=False))

shortlist = shortlist.copy()

def strongest_single_conf(items):
    rates = []
    for item in items:
        one_item_rows = transactions[transactions[item]]
        bad_rate = one_item_rows["class=bad"].sum() / len(one_item_rows)
        rates.append(bad_rate)
    return max(rates)

shortlist["best_single_conf"] = shortlist["antecedents"].apply(
    strongest_single_conf
)
shortlist["gain_pp"] = 100 * (
    shortlist["confidence"] - shortlist["best_single_conf"]
)

print(shortlist.sort_values("gain_pp", ascending=False)[
    ["antecedents", "condition_count", "bad_count",
     "support", "confidence", "lift", "best_single_conf", "gain_pp"]
].to_string(index=False))

six_candidates = shortlist[shortlist["gain_pp"] >= 5.0].copy()
six_candidates = six_candidates.sort_values("gain_pp", ascending=False)

print("候选代表规则数：", len(six_candidates))
print(six_candidates[
    ["antecedents", "condition_count", "bad_count",
     "support", "confidence", "lift", "gain_pp"]
].round(3).to_string(index=False))

for _, rule in six_candidates.iterrows():
    columns = list(rule["antecedents"])
    matching = transactions[columns].all(axis=1)
    matching_bad = matching & transactions["class=bad"]

    assert int(matching.sum()) == int(rule["condition_count"])
    assert int(matching_bad.sum()) == int(rule["bad_count"])

print("六条候选规则的人数核对通过")

export_table = six_candidates.copy()

export_table["antecedent"] = export_table["antecedents"].apply(
    lambda items: " AND ".join(sorted(items))
)
export_table["consequent"] = "class=bad"

output_path = output_dir / "e3_selected_rules.csv"

export_columns = [
    "antecedent", "consequent",
    "condition_count", "bad_count",
    "support", "confidence", "lift",
    "best_single_conf", "gain_pp",
]

export_table[export_columns].to_csv(output_path, index=False)
print("规则表已保存：", output_path)
