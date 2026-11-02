# DM-Group-Project1（CDS535 数据挖掘小组作业）

这个项目研究两个 UCI 数据集上的类别不平衡问题：德国信贷 credit-g（UCI 144）和甲状腺 sick（UCI 102）。我们把 4 个模型（决策树 / 随机森林 / 逻辑回归 / SVM）和 4 种处理不平衡的策略（plain / SMOTE / 代价敏感 / 调阈值）在两个数据集上全部交叉了一遍，一共 32 个组合，每个组合用十折交叉验证 + 一个锁定的 70:30 留出集评估。在此之外还加了一组探索性的集成实验（Bagging / AdaBoost）和带 Nadeau–Bengio 校正的配对 t 检验。

我们想回答四个问题（2026-10-29 全组一起改定的版本，旧版到新版的对照在 docs/RQ_定义与映射_v1.0.md）：

- RQ1 不平衡 + 代价不对称的时候，策略和模型该怎么搭配着选？
- RQ2 accuracy 什么时候会骗人，应该看哪些指标？
- RQ3 混合类型的属性（数值 + 名义）该怎么处理，编码方式有多重要？
- RQ4 训练好的模型拿到没见过的新样本上，决策还靠谱吗？

分工：A 管 credit-g 数据，B 管 sick 数据，C 管建模和平台（跑实验的代码基本都是 C 写的），D 管评估协议，E 管无监督和最终整合。

## 1. 仓库里有什么

```text
data/raw/                  两个数据集的 csv（用 "？" 表示缺失，读的时候要 na_values="？"）
data/metadata/             数据来源和标签含义的记录
configs/                   A 和 B 写的预处理规格单（10 月 12 日冻结，后面没再改过）
src/
  c_config.py              所有配置常量都在这（模型超参、数据集信息、seed=42）
  c_pipelines.py           统一的 imblearn 流水线（预处理 + SMOTE + 模型）
  run_experiment.py        跑实验的主入口
  c4_threshold.py          调阈值策略的包装器
  b7x_ensemble_study.py    集成实验入口（B 写的，sick 和 credit-g 两侧都能跑）
  stats_test.py            NB 校正的配对 t 检验（全部 120 对 + Holm 校正）
  c11_ensemble_ttest.py    集成的 2 对预指定检验
  c6_run_log.py            汇总运行日志用
  evaluate.py              D 的指标库，全项目算指标都走它
  其余 a*/b*/d*/e*         其他同学的任务脚本
results/                   主矩阵 32 配置的结果 + 锁定的切分文件
results/ensemble/          集成实验的结果（单独放，不和主矩阵混）
results/ttest/             检验结果表
results/run_log_v1.csv     所有 84 次运行的日志
docs/                      每个任务的运行记录，还有 D 的评估协议说明
overleaf/                  Overleaf 报告的镜像
figures/                   图（含两张流水线示意图）
RUN_GUIDE.md               A 维护的上手指南
requirements.txt           锁定的依赖版本
```

## 2. 配环境

```bash
# conda（我们组平时用的就是这套）
conda create -n cds535 python=3.11 -y && conda activate cds535
pip install -r requirements.txt

# 或者用 uv / venv，见 RUN_GUIDE.md（里面还写了 macOS 上 pip 会碰到的坑）
```

## 3. 数据

数据已经直接放在仓库里了（`data/raw/credit-g.csv` 和 `data/raw/sick.csv`）。想重新下载的话去 UCI（ID 144 和 ID 102），或者用 `ucimlrepo.fetch_ucirepo`。以 `results/split_manifest.json` 里记录的 SHA-256 为准核对数据没变；列名的来龙去脉见 `data/metadata/credit-g_source.md`。

## 4. 怎么复现（按顺序跑这 10 条命令）

我们为了让结果能一模一样地复现，做了四件事：切分文件生成一次就锁死、随机种子固定 42、超参提前冻结、依赖版本锁定。所以只要环境版本一样，按下面顺序跑，出来的数字应该和仓库里的一模一样（唯一的例外是每行的 runtime_sec，那本来就是计时）。**注意：代码只读锁定切分，绝不重新划分；M3（10 月 23 日）之后也不允许再加实验或改配置。**

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

只想先试一个配置的话：`python -m src.run_experiment --dataset creditg --model lr --strategy plain --protocol cv10`。

## 5. 集成实验

入口是 `src/b7x_ensemble_study.py`，跑 Bagging 和 AdaBoost 两种集成（基学习器都是主矩阵那棵剪过枝的决策树，n_estimators=100 对齐随机森林），策略只跑 plain 和 smote。结果全部放在 `results/ensemble/`，和主矩阵分开，不动任何冻结的东西。它在统计上定位是 exploratory——我们只事先定了两对比较（bag_plain vs dt_plain，ada_plain vs dt_smote），其余的只报均值±标准差，不然多重比较说不过去。

## 6. 统计检验

```bash
python -m src.stats_test --self-test-only   # 只跑 8 组核验锚点（锁定切分/summary 对账/C4 auc 锚点等）
python -m src.stats_test                    # 全量: 2 数据集×4 指标×120 配对(NB 校正 + Holm)
```

结果在 `results/ttest/`。两个边界情况的处理（逐折完全相同给 p=1、零方差给 NaN）写在 `docs/C5_统计检验运行记录_v1.0.md` 里。

## 7. 运行日志

`results/run_log_v1.csv` 记录了全部 84 次运行（主矩阵 64 + ZeroR 4 + 集成 16），每个配置的参数、seed、数据指纹、耗时、三个结果文件的路径都在。重新生成：`python -m src.c6_run_log`。
