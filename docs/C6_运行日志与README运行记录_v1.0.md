# C6 运行日志总表 + README + 冻结 tag · 运行记录 v1.0

- **负责人**: C（建模与平台）
- **日期**: 2026-10-23（M3 冻结日；V5 合并版 C6 项）
- **状态**: 本机完成 + 自检 PASS，**尚未 commit/push、尚未打 tag**（按组内流程：过目 → 分段提交 → push → 打 tag；命令清单见 §5）
- **运行环境**: `/opt/anaconda3/envs/cds535/bin/python`
- **入口**: `python -m src.c6_run_log`（仓库根）

## 1. 交付物清单

| 文件 | 变更 | 内容 |
|---|---|---|
| `src/c6_run_log.py` | 新增 | 运行日志总表生成器：扫描主矩阵/ZeroR/集成全部 `*_summary.csv`，行数与 files_complete 断言不过即拒绝交付 |
| `results/run_log_v1.csv` | 新增 | **84 行** = 主矩阵 64（32 配置×2 协议）+ ZeroR 4 + 集成 16（V4 增补口径）；列：config_id / dataset / model / strategy / protocol / run_id / seed / data_sha256 / **params**（从 c_config 与 V4 §1.3 冻结规格代出，单一事实来源）/ runtime_sec / folds·summary·preds 三类文件路径 / files_complete |
| `README.md` | 新增 | 根目录总览：仓库结构、环境安装、数据获取（sha 以 split_manifest 为准）、**执行顺序与一键复现（0–9 共 10 条命令，第 0 条为环境预检）**、**ensemble 复现小节**（入口/独立命名空间/与主矩阵关系/exploratory 定位）、统计检验、运行日志、组规铁律 |
| 本记录 | 新增 | 含 §5 提交与打 tag 命令清单（待执行） |

## 2. 总表核验

| 锚点 | 结果 |
|---|---|
| 行数 84 = 64+4+16（脚本断言） | ✅ |
| files_complete（folds/summary/preds 三件齐全）全部 True | ✅ |
| seed 全 42；数据集×协议分布 21×4（16 主+1 zero+4 集成/侧） | ✅ |
| params 单一来源：主矩阵代出 c_config 冻结超参；集成按 V4 §1.3 冻结规格代出（字符串手写于脚本常量 ENS_PARAMS，**内容已经对抗审计逐字核对与 b7x_ensemble_study.py 实现一致**）；ZeroR=most_frequent | ✅ |
| 不重算任何指标（全部字段读自 summary 文件本身） | ✅ |
| **口径备注**：runtime_sec 为 3 位小数舍入展示值（秒级日志用途足够；秒级以下精度以各 summary 文件原值为准） | ✅ |

## 3. README 要点（与 RUN_GUIDE 的分工）

`RUN_GUIDE.md`（A 维护）保留：uv venv 快速上手、逐脚本用法、基线数字。`README.md` 新承担：仓库全景、复现链（0–9 共 10 条命令，含环境预检）、**ensemble 复现小节**（V5 C6 指定）、统计检验与运行日志入口、组规铁律（锁定切分/冻结令/提交规范）。二者互链不重复。

## 4. 一键复现承诺的边界（README §4 原文口径）

逐位一致依赖四件事同时成立：锁定切分（只消费 `cv_folds_*`/`idx_*`）× seed=42（一切随机源）× 超参冻结（c_config/V4 §1.3）× 版本锁定（requirements.txt）。缺任一件退化为"同分布不同值"。

## 5. 提交与打 tag 命令清单（**待组内过目后执行，本次未执行**）

```bash
# 三段式提交（沿用 C4/C5 惯例，备注逐文件写明用途）
git add src/c6_run_log.py results/run_log_v1.csv
git commit -m "C6-1 代码+总表: 运行日志 84 行=主矩阵64+ZeroR4+集成16 — src/c6_run_log.py(断言不过拒绝交付)+results/run_log_v1.csv(config/参数/seed/sha/runtime/三类文件路径/files_complete, 参数自c_config与V4§1.3代出零手抄)"

git add README.md
git commit -m "C6-2 文档: 根目录README.md — 仓库结构/环境/数据获取(sha以split_manifest为准)/一键复现10条命令(0-9含预检)/ensemble复现小节(入口b7x、独立命名空间、与主矩阵exploratory关系)/统计检验/运行日志入口/组规铁律; 与RUN_GUIDE互链不重复"

git add src/c11_ensemble_ttest.py results/ttest/ensemble_2pairs_prespecified_v1.csv docs/C10_集成资产复核记录_v1.0.md docs/C11_集成预指定检验运行记录_v1.0.md docs/C6_运行日志与README运行记录_v1.0.md
git commit -m "C10+C11: 集成资产复核0FAIL(git层50文件全新增零修改, 命名/锁定切分/preds/对账全过, 记录入docs) + 2对预指定检验(仅bag_plain vs dt_plain与ada_plain vs dt_smote×两侧=4检验, f1/NB校正复用C5公式, sick ada vs dt_smote p=6.4e-4***唯一强显著, 余3对ns/边缘; 复用stats_test.nb_paired_t公式单一来源, 同折配对前置断言)"

# push 后随 M3 冻结打 tag（当前仓库无任何 tag）
git push origin main
git tag -a v1.0-m3-freeze -m "M3 experiment freeze 2026-10-23: main matrix 32 configs + zero refs + ensemble 16 configs (results/ensemble) + NB-corrected t-tests (results/ttest) + run log 84 rows + README repro guide; locked splits x seed42 x frozen hyperparams x pinned versions"
git push origin v1.0-m3-freeze
```

**执行记录（2026-10-23，组内确认后）**：实际按任务边界细化为 **4 段提交**（C10 复核记录 → C11 检验代码+结果+记录 → C6-1 代码+总表 → C6-2 README+本记录）→ `git push origin main` → 打注释 tag `v1.0-m3-freeze` 并推送；实际提交信息以 `git log` 为准（要点与本节计划一致，另含对抗复审结论）。

## 6. 备注

- tag 名 `v1.0-m3-freeze` 为建议值，组长/E 如有命名偏好可改；tag 信息概括"冻结时点的全部资产"。
- C6 三个待 D/组内确认项延续挂起（不阻塞本任务）：τ 记录位置（notes vs 专列）、`_holdout` 中缀命名、A6 std 的 ddof 引用口径。
- **对抗复审新登记（2026-10-23，不改冻结代码）**：`src/run_experiment.py:37` 行内注释写"46 列"，实际 template 为 49 列（模块 docstring 已正确写 29/49）——系 C3 时期笔误、冻结点之前已存在，仅注释与实际不符、不影响任何行为；按冻结令不改已入库代码，登记待 D/组内下次顺路修订。
