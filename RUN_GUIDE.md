# 项目运行指南

## 项目概览
- 项目类型：Python 数据挖掘实验（CDS535 小组作业）
- 技术栈：pandas 3.0.6 / scikit-learn 1.9.1 / imbalanced-learn 0.14.2 / mlxtend 0.25.0（版本以 requirements.txt 锁定为准）
- Python 版本：3.11（与组内方案一致）

## 快速开始

### 1. 建虚拟环境（macOS 上 pip 可能因系统版本解析 bug 启动失败，推荐 uv）

```bash
# 装了 uv（推荐）
uv venv ~/venvs/dm-group --python 3.11
uv pip install -r requirements.txt -p ~/venvs/dm-group/bin/python

# 没有 uv 的组员用常规方式
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### 2. 运行

```bash
# 激活（uv 方式）
source ~/venvs/dm-group/bin/activate
# 或直接用绝对路径跑，免激活
~/venvs/dm-group/bin/python baseline_check.py     # 版本核对
~/venvs/dm-group/bin/python load_data.py          # ARFF → CSV（数据已转好，一般不用重跑）
~/venvs/dm-group/bin/python PreExperiment.py      # 决策树 + 10 折 CV 基线
~/venvs/dm-group/bin/python src/a1_creditg_source.py  # credit-g 来源记录 + SHA-256（需联网核验）
~/venvs/dm-group/bin/python src/a2a3_creditg_schema_stats.py  # A2 字段字典 + A3 质量统计
~/venvs/dm-group/bin/python src/a4_creditg_eda.py      # A4 三张探索图 + 图注（figures/）
```

## 基线数字（2026-10-11 本机复跑，供双机一致性核对）

| 数据集 | 决策树 10 折 CV Accuracy |
|---|---|
| credit-g | 0.6660 ± 0.0233 |
| sick | 0.9867 ± 0.0049 |

## 交付物位置
- data/metadata/credit-g_source.md — A1 来源记录（SHA-256 + BibTeX）
- results/credit-g_schema.xlsx — A2 字段字典（21 列）+ 数值五数概括
- results/stats/credit-g_stats.md — A3 质量统计（shape/类别分布/缺失/重复，供 D 核对）
- figures/fig1~fig3 + credit-g_fig_captions.md — A4 探索图与图注（数字自动生成）
- configs/spec_creditg.md — A5 预处理规格单 v1.0（2026-10-12 冻结）

## 常见问题
- **pip/ensurepip 报 `invalid literal for int()` 或 mac_ver 空**：本机 Homebrew Python 在 macOS 26 上解析版本号失败，改用 uv（它会自动下载可用的托管 Python）。
- **PyCharm 配置**：Settings → Project → Python Interpreter → Add Local → 选 `~/venvs/dm-group/bin/python`。
- **数据变体说明**：仓库 CSV 为 OpenML credit-g 变体（语义列名 + good/bad），UCI 官方为 Attribute1-20 编码 + 1/2 标签，同一数据，详见 data/metadata/credit-g_source.md。
