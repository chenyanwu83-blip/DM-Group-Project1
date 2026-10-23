# DM-Group-Project1 · CDS535 数据挖掘小组项目

credit-g（UCI ID 144）+ sick（UCI ID 102）双数据集**类别不平衡**研究：4 模型 × 4 策略 × 2 数据集 = 32 配置主矩阵（cv10 + 70:30 holdout 双协议），附加集成学习 exploratory 研究（Bagging/AdaBoost）与 NB 校正折级配对 t 检验。

- 研究问题：RQ1 不平衡度（70:30 → 94:6）下四类策略的相对效果；RQ2 accuracy 何时失效与指标选择；RQ3 官方 5:1 代价矩阵下最优模型是否改变；RQ4 混合属性对预处理与模型选择的影响。
- 分工与时间线见组内 V5 合并版方案；代码角色：A=credit-g 数据、B=sick 数据、C=建模与平台、D=评估协议、E=无监督与整合。

## 1. 仓库结构

```text
data/raw/                  两数据集 CSV（OpenML 变体，语义列名；sha256 见 results/split_manifest.json）
data/metadata/             A1/B1 来源记录、标签语义
configs/                   A5/B5 预处理规格单（2026-10-12 冻结）
src/
  c_config.py              C2：冻结规格 → 常量（模型超参/数据集元信息/SEED=42）
  c_pipelines.py           C2：统一 imblearn 流水线（双通道预处理 + SMOTE + 模型）
  run_experiment.py        C3/C4 实验发动机：--baseline | --matrix --strategy {plain,smote,cost,threshold} | --zero
  c4_threshold.py          C4：ThresholdClassifier 包装器（内层 5 折 OOF 选 τ → 锁定 → 重拟合）
  b7x_ensemble_study.py    集成研究入口（B 原生 sick 侧，A 镜像 credit-g 侧）：bag/ada × plain/smote
  stats_test.py            C5：NB(2003) 校正折级配对 t 检验（全配对 120×2 数据集×4 指标 + Holm）
  c11_ensemble_ttest.py    C11：集成 2 对预指定检验（bag_plain vs dt_plain；ada_plain vs dt_smote）
  c6_run_log.py            C6：运行日志总表生成器
  evaluate.py              D 库：指标计算/汇总（compute_metrics / aggregate_folds），本项目不自写指标公式
  其余 a*/b*/d*/e*         各角色任务脚本
results/                   主矩阵 32 配置 folds/summary/preds + 锁定切分（cv_folds_*/idx_*/split_manifest）
results/ensemble/          集成研究独立命名空间（16 配置，不与主矩阵混排）
results/ttest/             C5 全量检验表 + C11 预指定检验表
results/run_log_v1.csv     全部 84 次运行的日志总表（含集成）
docs/                      各任务运行记录（C2–C6、A/B/D/E）与 D1 评估协议 / D2 调用说明
overleaf/                  Overleaf 报告工程镜像
figures/                   图与流水线结构图
RUN_GUIDE.md               快速上手（uv venv + 逐脚本用法，A 维护）
requirements.txt           锁定版本（sklearn 1.9.1 / pandas 3.0.6 / scipy 1.17.1 / imblearn 0.14.2 …）
```

## 2. 环境安装

```bash
# 方式一：conda（C 的 cds535 环境即此口径）
conda create -n cds535 python=3.11 -y && conda activate cds535
pip install -r requirements.txt

# 方式二：uv / venv（见 RUN_GUIDE.md，含 macOS pip 兼容问题说明）
```

## 3. 数据获取

数据已随仓库提供（`data/raw/credit-g.csv`、`data/raw/sick.csv`，sick 读取需 `na_values="?"`）。
重新获取：UCI ID 144（DOI 10.24432/C5NC77）/ ID 102（DOI 10.24432/C5D010），或 `ucimlrepo.fetch_ucirepo`。
**数据指纹以 `results/split_manifest.json` 的 SHA-256 为准**；列名变体说明见 `data/metadata/credit-g_source.md`。

## 4. 执行顺序与一键复现（0–9 共 10 条命令，第 0 条为环境预检）

任何机器、同版本环境、按序执行，结果应**逐位一致**（锁定切分 × seed=42 × 超参冻结 × 版本锁定，四件事同时成立）：

```bash
python baseline_check.py                                # 0. 版本与双机基线核对
python -m src.run_experiment --baseline                 # 1. C3: plain 8 配置×2协议 + ZeroR 参照
python -m src.run_experiment --matrix --strategy threshold   # 2. C4: threshold 8 配置
python -m src.run_experiment --matrix --strategy smote  # 3. A7/B7: smote 8 配置
python -m src.run_experiment --matrix --strategy cost   # 4. A7/B7: cost 8 配置（主矩阵 32 齐单侧16×2）
python src/b7x_ensemble_study.py --dataset sick --all   # 5. 集成 sick 侧 4 配置×2协议
python src/b7x_ensemble_study.py --dataset creditg --all  # 6. A 镜像 credit-g 侧
python -m src.stats_test                                # 7. C5: NB 校正全配对检验（锚点不过拒绝出表）
python -m src.c11_ensemble_ttest                        # 8. C11: 集成 2 对预指定检验
python -m src.c6_run_log                                # 9. C6: 重建运行日志总表
```

单配置冒烟：`python -m src.run_experiment --dataset creditg --model lr --strategy plain --protocol cv10`。

## 5. 集成研究（Ensemble Study · V4 增补）复现说明

- **入口**：`src/b7x_ensemble_study.py`（`--dataset {sick,creditg} --all`；单点冒烟 `--model bag --strategy plain --protocol cv10`）。
- **设定**：BaggingClassifier(DT) / AdaBoostClassifier(DT)，基学习器=主矩阵同款 `DecisionTree(entropy, ccp_alpha=0.01, rs=42)`，`n_estimators=100`（对齐 RF 口径）；策略 plain/smote（SMOTE 在 imblearn Pipeline 内，仅训练折）。
- **独立命名空间**：全部产出在 `results/ensemble/`（16 配置×2 协议 folds/summary + preds 16 份），**不与主矩阵 32 配置混排、不修改任何冻结资产**（C10 复核记录：git 层 50 文件全为新增、零修改删除）。
- **与主矩阵的关系**：复用同一锁定切分与 `src/evaluate.py` 口径，但统计上定位为 **exploratory extension**——只做 2 对预指定检验（`python -m src.c11_ensemble_ttest` → `results/ttest/ensemble_2pairs_prespecified_v1.csv`），其余只报均值±标准差，防多重比较质疑。
- 依据：V4 方案 §1.3/§1.4；运行记录 `docs/A7x_ensemble_creditg运行记录_v1.0.md`、C10/C11 记录。

## 6. 统计检验（C5/C11）

```bash
python -m src.stats_test --self-test-only   # 只跑 8 组核验锚点（锁定切分/summary 对账/C4 auc 锚点等）
python -m src.stats_test                    # 全量: 2 数据集×4 指标×120 配对(NB 校正 + Holm)
```

产出：`results/ttest/`（pairs 长表 / matrix 速览 / key_comparisons / 集成预指定 2 对）。方法与边界（identical→p=1、零方差→NaN、星号=raw p α=0.05）见 `docs/C5_统计检验运行记录_v1.0.md`。

## 7. 运行日志总表（C6）

`results/run_log_v1.csv`：84 行 = 主矩阵 64 + ZeroR 4 + 集成 16（每行含 config_id / dataset / 参数 / seed / data_sha256 / runtime_sec / folds·summary·preds 路径 / 文件齐全性）。重建：`python -m src.c6_run_log`（行数与 files_complete 断言不过即拒绝交付）。

## 8. 规范与协约（新人必读）

- 评估协议：`docs/D1_评估协议_v1.0.md`（标签方向/两协议口径/指标边界/代价公式）；接口与命名：`docs/D2_给ABC的调用说明.md`。
- 铁律：只消费锁定切分（绝不重新划分）；动样本集=配置变更须全组同意；模板列零偏差；提交按"代码/结果/文档"分段、备注逐文件写明用途。
- 各任务自检与锚点：见 `docs/C*` 运行记录系列。
- 冻结：M3（2026-10-23）后禁止新增实验与改配置；代码随冻结打 tag，课程封面填本仓库链接。
