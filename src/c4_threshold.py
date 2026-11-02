"""
threshold 策略: 模型照 plain 训练, 只是预测的时候不用默认的 0.5,
而是自己挑一个阈值 tau。

做法: 在训练折内部再分 5 折, 拿折外(out-of-fold)概率在 0.05~0.95 的网格上
挑 F1 最大的 tau(并列取小的, 保证确定性), 锁定之后用全部训练数据重新拟合,
预测时概率 >= tau 判正类。tau 的信息从头到尾不接触外层验证折。

选 tau 用的 F1 就是 sklearn 的 f1_score, 正类显式映射成 1;
最终报告的指标还是统一走 evaluate.py, 这里不自算。
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

TAU_GRID = (0.05, 0.95, 0.05)   # 起/止/步长, 用 np.arange 注意端点要含 0.95
INNER_FOLDS = 5                  # 内层折数定为 5


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """包住一整条 plain 子流水线的阈值移动估计器。

    make_base 是个无参工厂, 每次调用返回一条全新的子流水线
    (由 c_pipelines.build_pipe 的 threshold 分支传进来)。
    """

    def __init__(self, make_base, pos_label, neg_label,
                 tau_grid=TAU_GRID, inner_folds=INNER_FOLDS, seed=42):
        self.make_base = make_base
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.tau_grid = tau_grid
        self.inner_folds = inner_folds
        self.seed = seed

    # 小工具
    def _taus(self):
        lo, hi, step = self.tau_grid
        return np.round(np.arange(lo, hi + step / 2, step), 2)   # 0.05 到 0.95 共 19 档

    def _proba_pos(self, est, X):
        j = list(est.classes_).index(self.pos_label)
        return est.predict_proba(X)[:, j]

    # 下面是 sklearn 的标准接口
    def fit(self, X, y):
        taus = self._taus()
        y_bin = (y == self.pos_label).astype(int).to_numpy()

        # 第一步: 内层 5 折收集折外正类概率, 每折都造全新流水线
        inner = StratifiedKFold(n_splits=self.inner_folds,
                                shuffle=True, random_state=self.seed)
        proba_oof = np.full(len(y), np.nan)
        for tr, va in inner.split(X, y):
            est = self.make_base()
            est.fit(X.iloc[tr], y.iloc[tr])
            proba_oof[va] = self._proba_pos(est, X.iloc[va])
        assert np.isfinite(proba_oof).all(), "OOF 概率存在缺口(内层折未覆盖)"

        # 第二步: 网格扫 tau, 谁的 F1 大用谁, 并列取小的 tau
        best_tau, best_f1 = 0.5, -1.0
        for tau in taus:
            pred_bin = (proba_oof >= tau).astype(int)
            f1 = f1_score(y_bin, pred_bin, zero_division=0)
            if f1 > best_f1:
                best_tau, best_f1 = float(tau), float(f1)
        self.best_tau_, self.best_oof_f1_ = best_tau, best_f1

        # 第三步: tau 定死了, 用全部训练数据重新拟合一遍
        self.est_ = self.make_base()
        self.est_.fit(X, y)
        self.classes_ = self.est_.classes_
        return self

    def predict(self, X):            # 按锁定的 tau 出预测
        prob = self._proba_pos(self.est_, X)
        return np.where(prob >= self.best_tau_,
                        self.pos_label, self.neg_label)

    def predict_proba(self, X):      # 概率原样透传, AUC 照常能算
        return self.est_.predict_proba(X)
