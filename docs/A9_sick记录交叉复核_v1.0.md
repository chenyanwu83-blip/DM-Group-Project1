# A9 交叉复核意见：B 的 sick 来源记录与字段说明（1 页）

> 复核人：A（credit-g 负责人）｜日期：2026-10-27｜方式：对照 UCI 官方页面逐项打勾 + 独立重算数据本体（复用 A1 核验方法：文件级哈希 / 分布比对 / 缺失机制）
> 复核对象：`data/metadata/sick_source.md`、`data/metadata/sick_label_semantics.md`、`result/stat/sick_missing_audit.csv`、`configs/spec_sick.md`（B4 对照数字一并抽查）

## 一、来源四件套（对照 UCI dataset/102 页面，逐项打勾）

| 项 | B 记录 | UCI 页面 | 结果 |
|---|---|---|---|
| UCI ID / DOI | 102 / 10.24432/C5D010 | 102 / 10.24432/C5D010 | ✅ |
| 整理者 / 捐赠者 | Ross Quinlan（Weka 打包）/ Garavan Institute（悉尼） | creator: Ross Quinlan；6 databases from the Garavan Institute, Sydney | ✅ |
| 年份 | 1986 | donated 12/31/1986（ARFF 头注释写 1987，属供数年份，不影响引用） | ✅ |
| 许可 | CC BY 4.0 | CC BY 4.0 | ✅ |
| 获取方式 | Weka 3.8.7 自带 data 目录 → b1_make_sick_csv.py → C 的 load_data.py 统一重生成 | 本地 sick.arff 头注释（Garavan + Quinlan）与 Weka 发行版一致 | ✅ |

**备注（不阻塞）**：UCI 页面注明 Quinlan 认为个别同源文件（Hypothyroid/sick-euthyroid 的部分版本）"have been corrupted"；我们用的是 Weka 打包的 sick.arff 版本，B 已在来源记录中写明取数链路，建议 sec3b 引用时保留"via Weka package"措辞。

## 二、数据本体独立重算（SHA-256 → 逐项全过）

| 检查项 | 独立实算 | B 声称 | 结果 |
|---|---|---|---|
| SHA-256（文件字节级） | `3cc14348…854ab` | 同左（并与 results/split_manifest.json 锚定） | ✅ 三方一致 |
| 规模 | 3,772 × 30 | 3,772 × 30 | ✅ |
| 类别分布 | negative 3,541 : sick 231 | 同左（manifest 2640/1132 切分一致） | ✅ |
| 标签取值 | ['negative','sick']，0 缺失 | 同左 | ✅ |
| 完全重复行 | 61 | 61（保留并披露） | ✅ |
| TBG 机制 | TBG 100% 缺失；'TBG measured' 恒为 'f'（3,772 行） | 同左 | ✅ 删列决策成立 |

缺失口径（B4 一并核过）：删 TBG 两列后**格子级缺失 2.2%**；**29.9% 的行至少缺一项**（1,129 行，其中正类 19 → 231 剩 212）——与 spec §4「3,772→2,643、正类 231→212」及 D 的 B4 复核数字三重吻合。B4 结论数字与 `results/b4_missing_comparison.csv` 逐项一致（插补 F1 0.830±0.049 / AUC 0.974 vs 删行 0.843±0.069 / 0.964，差异在 1σ 内 → 插补）✅

## 三、字段说明语义核查 —— 🔴 发现 1 项必改

B 的两条红线（sick ≠「是否甲减」；negative ≠「完全健康」）**方向正确，予以保留**。但正类定义本身与标准医学定义**相反**：

- **B 现写**：「sick-euthyroid：甲状腺激素水平正常，但确诊患有甲状腺疾病（如甲状腺炎、结节等）」
- **问题**：sick-euthyroid = euthyroid sick syndrome（正常甲状腺病态综合征/非甲状腺疾病病态综合征）：患者患有严重的**非甲状腺**疾病（重症/感染/术后等），导致甲状腺**化验指标异常**（T3↓ 等），但**甲状腺本身并无病变**（euthyroid 即"无甲状腺病"）。B 的表述把两点都写反了——既把化验说成"正常"（恰因化验异常才被筛出），又把"无甲状腺病"写成"确诊甲状腺病"。
- **证据边界**：UCI 页面本身不含标签定义（已核实），Weka sick.arff 头也只写 `sick, negative | classes`；故语义必须引医学来源支撑，不能凭印象。
- **建议改法**（供 B 采用）：「sick = sick-euthyroid（正常甲状腺病态综合征）：严重非甲状腺疾病导致甲状腺化验指标异常、但甲状腺本身无病变的患者；negative = 本次筛查未检出甲状腺异常」，并在 sec3b 附一条医学来源引用（如内分泌学教材/综述）。
- **影响面**：此定义将进 sec3b 与答辩临床叙事，语义反一层整段全错——请 B 于 sec3b 定稿前修正。

## 四、低风险备注（2 项，随下次文档更新顺手改）

1. `spec_sick.md §0` 写留出索引存 `results/holdout_idx.csv`，实际产物为 `holdout_idx_sick.csv`（D 的 evaluate.py 命名）；`§1①` 写删列 `'TBG_measured'`，实际列名为 `'TBG measured'`（空格版）——后者 C2 回执第 5 节已登记闭环，前者建议同样补注。
2. 更正一则**非 B 的**错误：「sick 缺失 ≈16.1%」的说法出自 A 侧学习资料（本次复核前的 HTML），无数据支撑；实测为格子级 2.2%（不含 TBG 两列）/ 行级 29.9%。A 侧学习资料已同步修正。

## 五、结论

**通过（带 1 项必改）**。来源四件套、哈希指纹、规模/分布/重复、TBG 机制、B4 数字全部独立复算一致；标签语义一处需按第三节改写后再入报告。

签名：A（credit-g 负责人）｜2026-10-27
