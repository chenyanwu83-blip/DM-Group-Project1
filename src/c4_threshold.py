# -*- coding: utf-8 -*-
"""C4: threshold 策略包装器 —— 规格冻结做法的代码化。

规格(A5/B5 §3 threshold 行 + D1 协议"阈值内层选"条款):
  训练折内层 5 折 OOF 概率选 τ (0.05–0.95 步长 0.05, 最大化 F1, pos_label 显式)
  → 锁定 τ → 全训练折重拟合 → 按 τ 判。

防泄漏要点(D1: τ 的任何信息不得接触外层验证折):
  - 选 τ 只用外层训练折内部的内层 5 折(out-of-fold 概率);
  - 内层每折重新 make_base() 造全新子流水线, 插补/编码/缩放随内层折重拟合;
  - predict() 按 best_tau_ 判(D2: predict 必须体现锁定决策规则),
    predict_proba() 透传(指标库 classes=est.classes_ 方向自检原样可用)。

τ 选择用的 F1 即 sklearn.metrics.f1_score(与 src/evaluate.py 同一库函数,
显式映射正类=1/负类=0 后调用, 不自写公式); 报告指标仍一律走 compute_metrics。
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

TAU_GRID = (0.05, 0.95, 0.05)      # 起/止/步长, np.arange 端点含 0.95
INNER_FOLDS = 5                     # 规格: 内层 5 折


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """包住整条 pre→clf 子流水线的阈值移动估计器。

    make_base: 无参工厂, 每次调用返回一条全新 plain 子流水线
               (由 c_pipelines.build_pipe 的 threshold 分支注入)。
    """

    def __init__(self, make_base, pos_label, neg_label,
                 tau_grid=TAU_GRID, inner_folds=INNER_FOLDS, seed=42):
        self.make_base = make_base
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.tau_grid = tau_grid
        self.inner_folds = inner_folds
        self.seed = seed

    # ---- 内部工具 -------------------------------------------------
    def _taus(self):
        lo, hi, step = self.tau_grid
        return np.round(np.arange(lo, hi + step / 2, step), 2)   # 0.05..0.95, 19 档

    def _proba_pos(self, est, X):
        j = list(est.classes_).index(self.pos_label)
        return est.predict_proba(X)[:, j]

    # ---- sklearn 接口 --------------------------------------------
    def fit(self, X, y):
        taus = self._taus()
        y_bin = (y == self.pos_label).astype(int).to_numpy()

        # 1) 内层 5 折 OOF 正类概率(每折全新子流水线, 折内拟合)
        inner = StratifiedKFold(n_splits=self.inner_folds,
                                shuffle=True, random_state=self.seed)
        proba_oof = np.full(len(y), np.nan)
        for tr, va in inner.split(X, y):
            est = self.make_base()
            est.fit(X.iloc[tr], y.iloc[tr])
            proba_oof[va] = self._proba_pos(est, X.iloc[va])
        assert np.isfinite(proba_oof).all(), "OOF 概率存在缺口(内层折未覆盖)"

        # 2) 网格选 τ: OOF 概率按 τ 判, 正类 F1 最大者胜; 并列取最小 τ(确定性)
        best_tau, best_f1 = 0.5, -1.0
        for tau in taus:
            pred_bin = (proba_oof >= tau).astype(int)
            f1 = f1_score(y_bin, pred_bin, zero_division=0)
            if f1 > best_f1:
                best_tau, best_f1 = float(tau), float(f1)
        self.best_tau_, self.best_oof_f1_ = best_tau, best_f1

        # 3) 锁定 τ, 全训练折重拟合(吃满全部训练数据)
        self.est_ = self.make_base()
        self.est_.fit(X, y)
        self.classes_ = self.est_.classes_
        return self

    def predict(self, X):                       # 体现锁定决策规则(D2)
        prob = self._proba_pos(self.est_, X)
        return np.where(prob >= self.best_tau_,
                        self.pos_label, self.neg_label)

    def predict_proba(self, X):                 # 透传(auc/方向自检照常)
        return self.est_.predict_proba(X)
