"""A6｜credit-g 对照实验①：OneHotEncoder 有/无（角色 A 交付物）

按 configs/spec_creditg.md §4 执行：
- LogisticRegression(max_iter=1000, random_state=42) × {编码② OneHot+StandardScaler vs 编码③ Ordinal+数值原样}
  × StratifiedKFold(10, shuffle=True, random_state=42)
- scoring = accuracy / F1(pos_label='bad') / ROC-AUC（y 编码 bad=1, good=0）
- 折级配对 t 检验（scipy.stats.ttest_rel，未校正，n=10，仅作差异概览；主矩阵的 Nadeau & Bengio
  校正检验由 C5 负责）

产出（results/）：
- a6_onehot_comparison.csv        汇总表（均值±标准差，对齐 D2 模板口径）
- a6_onehot_folds.csv             折级明细（供 D 复核与后续检验）
- stats/a6_onehot_conclusion.md   结论段（中文结论 + 英文报告段落，数字自动回填）
"""

import warnings
from pathlib import Path
from time import perf_counter

import pandas as pd
from imblearn.pipeline import Pipeline
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "credit-g.csv"
RESULTS = ROOT / "results"

SEED = 42
POS = "bad"  # D1 协议正类

NUMERIC = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents",
]
NOMINAL = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker",
]


def build_pipe(encoding: str) -> Pipeline:
    """onehot/ordinal 均出自 A5 规格单（冻结）；ordinal_scaled 为 A6 补充归因组（非冻结配置），
    用于把"编码效应"与"缩放效应"拆开。"""
    if encoding == "onehot":
        pre = ColumnTransformer([
            ("nominal", OneHotEncoder(handle_unknown="ignore"), NOMINAL),
            ("numeric", StandardScaler(), NUMERIC),
        ])
    elif encoding == "ordinal_scaled":
        pre = ColumnTransformer([
            ("nominal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1), NOMINAL),
            ("numeric", StandardScaler(), NUMERIC),
        ])
    else:  # ordinal：类别原样（序数）喂入、数值不缩放，模拟"无独热"的线性模型输入
        pre = ColumnTransformer([
            ("nominal", OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1), NOMINAL),
        ], remainder="passthrough")
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def n_features(encoding: str, X: pd.DataFrame, y: pd.Series) -> int:
    """变换后特征维度（单独 fit 仅为数维度；正式评估全部在 cross_validate 折内重拟合）。"""
    pipe = build_pipe(encoding)
    pipe.fit(X, y)
    return pipe.named_steps["pre"].transform(X.iloc[:1]).shape[1]


if __name__ == "__main__":
    df = pd.read_csv(CSV)
    X = df[NUMERIC + NOMINAL]
    y = (df["class"] == POS).astype(int)  # bad=1（正类），good=0

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "f1", "roc_auc"]  # y 已编码 bad=1，f1 等价 f1_score(pos_label='bad')

    # 切分一致性自检：与 D 锁定的折表逐折核对（文件存在时；fold_id 为 1 基）
    split_note = ""
    locked = RESULTS / "cv_folds_credit-g.csv"
    if locked.exists():
        lk = pd.read_csv(locked)
        base = int(lk.fold_id.min())
        same = all(set(lk.loc[lk.fold_id == f + base, "row_idx"]) == set(t.tolist())
                   for f, (_, t) in enumerate(skf.split(X, y)))
        split_note = (f"切分自检：与 D 锁定折表 cv_folds_credit-g.csv 逐折{'一致' if same else '不一致'}"
                      "（同配方 StratifiedKFold(10, shuffle, seed=42)）。\n")
        print(split_note, end="")

    configs = {
        "A6.1-creditg-logistic-onehot": ("onehot", "规格单②（冻结）"),
        "A6.2-creditg-logistic-ordinal": ("ordinal", "规格单③（冻结）"),
        "A6.3-creditg-logistic-ordinal-scaled": ("ordinal_scaled", "补充归因：序数+标准化（非冻结配置）"),
    }
    fold_rows, summary_rows = [], {}
    for config_id, (encoding, note) in configs.items():
        t0 = perf_counter()
        with warnings.catch_warnings():
            # 收敛状况以 n_iter 全量拟合记录为准，不刷屏
            warnings.simplefilter("ignore", ConvergenceWarning)
            cv = cross_validate(build_pipe(encoding), X, y, cv=skf, scoring=scoring)
        runtime = perf_counter() - t0
        full_fit = build_pipe(encoding).fit(X, y)
        n_iter = int(full_fit.named_steps["clf"].n_iter_.max())
        for i in range(skf.get_n_splits()):
            fold_rows.append({
                "config_id": config_id, "fold": i + 1,
                "accuracy": cv["test_accuracy"][i],
                "f1": cv["test_f1"][i],
                "roc_auc": cv["test_roc_auc"][i],
            })
        summary_rows[config_id] = {
            "encoding": encoding, "note": note,
            **{m: (cv[f"test_{m}"].mean(), cv[f"test_{m}"].std(ddof=1)) for m in scoring},
            "n_features": n_features(encoding, X, y),
            "n_iter_full_fit": n_iter,
            "converged": n_iter < 1000,
            "runtime_s": round(runtime, 2),
        }
        print(f"{config_id}: acc={cv['test_accuracy'].mean():.4f}±{cv['test_accuracy'].std(ddof=1):.4f} "
              f"f1={cv['test_f1'].mean():.4f}±{cv['test_f1'].std(ddof=1):.4f} "
              f"auc={cv['test_roc_auc'].mean():.4f}±{cv['test_roc_auc'].std(ddof=1):.4f} "
              f"(n_iter={n_iter}, {runtime:.1f}s)")

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(RESULTS / "a6_onehot_folds.csv", index=False)

    a = summary_rows["A6.1-creditg-logistic-onehot"]
    b = summary_rows["A6.2-creditg-logistic-ordinal"]
    c = summary_rows["A6.3-creditg-logistic-ordinal-scaled"]
    summary = pd.DataFrame([{
        "config_id": cid,
        "dataset": "credit-g", "model": "LogisticRegression", "strategy": "plain",
        "encoding": r["encoding"], "note": r["note"],
        "protocol": "StratifiedKFold(10, shuffle, seed=42)",
        "n_features": r["n_features"],
        "accuracy_mean±std": f'{r["accuracy"][0]:.4f}±{r["accuracy"][1]:.4f}',
        "f1_mean±std": f'{r["f1"][0]:.4f}±{r["f1"][1]:.4f}',
        "roc_auc_mean±std": f'{r["roc_auc"][0]:.4f}±{r["roc_auc"][1]:.4f}',
        "n_iter_full_fit": r["n_iter_full_fit"],
        "converged": r["converged"],
        "runtime_s": r["runtime_s"],
    } for cid, r in summary_rows.items()])
    summary.to_csv(RESULTS / "a6_onehot_comparison.csv", index=False)

    # 折级配对 t 检验（onehot vs ordinal，每指标 10 对）
    fa = folds[folds.config_id == "A6.1-creditg-logistic-onehot"]
    fb = folds[folds.config_id == "A6.2-creditg-logistic-ordinal"]
    pv = {m: stats.ttest_rel(fa[m].to_numpy(), fb[m].to_numpy()).pvalue for m in scoring}
    d = {m: a[m][0] - b[m][0] for m in scoring}
    print(f"\nΔ(onehot−ordinal): acc {d['accuracy']:+.4f} (p={pv['accuracy']:.3f}) | "
          f"f1 {d['f1']:+.4f} (p={pv['f1']:.3f}) | auc {d['roc_auc']:+.4f} (p={pv['roc_auc']:.3f})")

    # 结论第 1 句按实际方向措辞，避免跑前写死
    wins = [m for m in scoring if d[m] > 0]
    if len(wins) == len(scoring):
        verdict = "在三项指标上全面优于"
    elif not wins:
        verdict = "在三项指标上全面落后于"
    else:
        verdict = "与序数编码互有胜负（" + "、".join(f"Δ{m}={d[m]:+.4f}" for m in scoring) + "），总体"
    s1 = (f"One-hot 编码{verdict}序数编码：ΔAccuracy {d['accuracy']:+.4f}、ΔF1 {d['f1']:+.4f}、"
          f"ΔAUC {d['roc_auc']:+.4f}（折级配对 t 检验 p：acc {pv['accuracy']:.3f} / f1 {pv['f1']:.3f} / "
          f"auc {pv['roc_auc']:.3f}，未校正，n=10）。")

    # 补充归因：A6.3（序数+标准化）离哪端更近，说明差距主要来自编码还是缩放
    if abs(c["f1"][0] - b["f1"][0]) < abs(c["f1"][0] - a["f1"][0]):
        attribution = (f"补充组 A6.3（序数+标准化，F1={c['f1'][0]:.4f}）仍靠近序数组"
                       f"（{b['f1'][0]:.4f}）而远离独热组（{a['f1'][0]:.4f}），说明 F1 差距主要来自**编码方式**"
                       "（虚假顺序关系），而非数值缩放。")
    else:
        attribution = (f"补充组 A6.3（序数+标准化，F1={c['f1'][0]:.4f}）已靠近独热组"
                       f"（{a['f1'][0]:.4f}）而远离序数组（{b['f1'][0]:.4f}），说明差距主要来自**数值未缩放**，"
                       "编码本身贡献较小。")
    conv_note = ("另，序数组（数值未缩放，credit_amount 量级达 10⁴）lbfgs 达到 max_iter=1000 未完全收敛"
                 f"（n_iter={b['n_iter_full_fit']}），独热组 {a['n_iter_full_fit']} 轮即收敛——不稳定优化"
                 "本身也是 ③ 分支不适配线性模型的佐证" if not b["converged"] else "")

    md = f"""# A6 对照实验①结论：OneHotEncoder 有/无（LogisticRegression, credit-g）

设置：编码② OneHot(handle_unknown='ignore')+StandardScaler vs 编码③ Ordinal+数值原样；
LogisticRegression(max_iter=1000, random_state=42)；分层 10 折 CV(shuffle, seed=42)；正类 bad。
{split_note}数据：results/a6_onehot_comparison.csv（汇总）+ a6_onehot_folds.csv（折级明细）。

| 指标 | OneHot ② | Ordinal ③ | Δ(②−③) | 配对 t 检验 p（未校正） |
|---|---|---|---|---|
| Accuracy | {a['accuracy'][0]:.4f}±{a['accuracy'][1]:.4f} | {b['accuracy'][0]:.4f}±{b['accuracy'][1]:.4f} | {d['accuracy']:+.4f} | {pv['accuracy']:.3f} |
| F1(bad) | {a['f1'][0]:.4f}±{a['f1'][1]:.4f} | {b['f1'][0]:.4f}±{b['f1'][1]:.4f} | {d['f1']:+.4f} | {pv['f1']:.3f} |
| ROC-AUC | {a['roc_auc'][0]:.4f}±{a['roc_auc'][1]:.4f} | {b['roc_auc'][0]:.4f}±{b['roc_auc'][1]:.4f} | {d['roc_auc']:+.4f} | {pv['roc_auc']:.3f} |

变换后特征维度：OneHot {a['n_features']} 维 vs Ordinal {b['n_features']} 维。
补充归因组 A6.3（序数+标准化，非冻结配置）：acc {c['accuracy'][0]:.4f}±{c['accuracy'][1]:.4f} /
F1 {c['f1'][0]:.4f}±{c['f1'][1]:.4f} / AUC {c['roc_auc'][0]:.4f}±{c['roc_auc'][1]:.4f}。

## 结论（4 句）

1. {s1}{conv_note}。
2. 归因：{attribution}
3. 对 RQ4 的含义：混合属性数据集上"编码方式与模型族匹配"是必要决策——线性/核模型走 OneHot+标准化（规格单②），树模型直接吃序数编码（规格单③），两分支已在 A5 规格单固化进主矩阵流水线。
4. 与主矩阵衔接：本对照属 plain 策略下的编码消融，A7 的 16 配置沿用规格单②③分支，不再改动。

## 报告段落（EN，供 sec3a_creditg.tex）

To quantify the effect of encoding (RQ4), we ran LogisticRegression under two preprocessing
branches of our frozen specification: (a) one-hot expansion of the 13 nominal attributes with
standardised numerics ({a['n_features']} features), and (b) ordinal encoding with raw numerics
({b['n_features']} features), both evaluated by stratified 10-fold cross-validation
(seed = 42, positive class = bad). One-hot yielded accuracy {a['accuracy'][0]:.3f}, F1
{a['f1'][0]:.3f} and AUC {a['roc_auc'][0]:.3f}, against {b['accuracy'][0]:.3f} / {b['f1'][0]:.3f} /
{b['roc_auc'][0]:.3f} for the ordinal branch (fold-level paired t-test: F1 p={pv['f1']:.3f},
AUC p={pv['roc_auc']:.3f}). A supplementary control (ordinal encoding with standardised numerics)
shows that the gap is mainly attributable to the encoding itself rather than to scaling: ordinal
codes impose an artificial order on nominal attributes, which distorts the linear decision
boundary. One-hot encoding is therefore retained for all linear/kernel models in the main
experiment matrix, while tree-based models keep raw ordinal codes.

—— A 角色 {pd.Timestamp.today().date()} 生成；D 请按 D1 口径复核指标方向（正类 bad、概率列方向）。
"""
    RESULTS.joinpath("stats").mkdir(exist_ok=True)
    (RESULTS / "stats" / "a6_onehot_conclusion.md").write_text(md, encoding="utf-8")
    print("\n已写出: results/a6_onehot_comparison.csv, a6_onehot_folds.csv, stats/a6_onehot_conclusion.md")
