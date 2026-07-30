# -*- coding: utf-8 -*-
"""v5 阶段 B-2:outlier 分类器增强(XGB / CatBoost 二分类 + 概率融合)。

动机:v5 融合层的核心新元特征是期望值解析融合
    ev = p·(-33.219) + (1-p)·clean
其收益上限直接由 p 的判别力与校准质量决定。当前 p 只来自单个 LGB(AUC 0.902),
是整条链路上最薄弱、也最容易改进的一环 —— 分类任务比回归容易涨点,
且 AUC 每提升 0.005 都会通过 ev 传导到最终 RMSE。

产出:outputs/base/clf_xgb.npz / clf_cat.npz,以及三者 rank 平均的 clf_ens.npz。
融合层会把 clf_ens 作为 p 的来源之一参与方案比较。
用法:ELO_SEED=777 python src/v5_clf_boost.py
"""
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

import elo_pipeline as ep

BASE_DIR = "outputs/base"
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main():
    base = pd.read_parquet("data/processed/features.parquet")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    ybin = (y < -30).astype(int)
    imp = pd.read_csv("outputs/feature_importance.csv")
    selected = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"]
                if c in train.columns]
    X, X_test = train[selected], test[selected]
    folds = ep.make_folds(y)
    log(f"X={X.shape} outlier={int(ybin.sum())}({ybin.mean():.4%})")

    spw = float((ybin == 0).sum() / (ybin == 1).sum())  # 正类权重:替代 is_unbalance

    # ---- XGBoost 二分类 ----
    oof_x = np.zeros(len(X))
    pred_x = np.zeros(len(X_test))
    for k, (tr, va) in enumerate(folds):
        m = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.02, max_depth=6,
                              min_child_weight=40, subsample=0.8, colsample_bytree=0.7,
                              reg_alpha=1.0, reg_lambda=10.0, scale_pos_weight=spw,
                              tree_method="hist", eval_metric="auc",
                              early_stopping_rounds=200, n_jobs=ep.CONFIG["N_THREADS"],
                              random_state=ep.CONFIG["SEED"], verbosity=0)
        m.fit(X.iloc[tr], ybin.iloc[tr], eval_set=[(X.iloc[va], ybin.iloc[va])], verbose=False)
        oof_x[va] = m.predict_proba(X.iloc[va])[:, 1]
        pred_x += m.predict_proba(X_test)[:, 1] / len(folds)
        log(f"  [clf_xgb] fold{k + 1} AUC={roc_auc_score(ybin.iloc[va], oof_x[va]):.5f}")
    auc_x = roc_auc_score(ybin, oof_x)
    np.savez(os.path.join(BASE_DIR, "clf_xgb.npz"), oof=oof_x, pred=pred_x)
    log(f"[clf_xgb] 总 AUC={auc_x:.5f}")

    # ---- CatBoost 二分类 ----
    oof_c = np.zeros(len(X))
    pred_c = np.zeros(len(X_test))
    for k, (tr, va) in enumerate(folds):
        m = CatBoostClassifier(iterations=6000, learning_rate=0.03, depth=6,
                               l2_leaf_reg=12.0, bootstrap_type="Bernoulli", subsample=0.8,
                               loss_function="Logloss", eval_metric="AUC",
                               scale_pos_weight=spw, random_seed=ep.CONFIG["SEED"],
                               thread_count=ep.CONFIG["N_THREADS"],
                               allow_writing_files=False, verbose=0)
        m.fit(X.iloc[tr], ybin.iloc[tr], eval_set=(X.iloc[va], ybin.iloc[va]),
              early_stopping_rounds=200, use_best_model=True)
        oof_c[va] = m.predict_proba(X.iloc[va])[:, 1]
        pred_c += m.predict_proba(X_test)[:, 1] / len(folds)
        log(f"  [clf_cat] fold{k + 1} AUC={roc_auc_score(ybin.iloc[va], oof_c[va]):.5f}")
    auc_c = roc_auc_score(ybin, oof_c)
    np.savez(os.path.join(BASE_DIR, "clf_cat.npz"), oof=oof_c, pred=pred_c)
    log(f"[clf_cat] 总 AUC={auc_c:.5f}")

    # ---- 三者 rank 平均(概率尺度不同,rank 平均比算术平均稳)----
    oof_l, pred_l = (lambda d: (d["oof"], d["pred"]))(np.load(f"{BASE_DIR}/clf.npz"))
    auc_l = roc_auc_score(ybin, oof_l)
    r = lambda v: rankdata(v) / len(v)
    oof_e = (r(oof_l) + r(oof_x) + r(oof_c)) / 3
    pred_e = (r(pred_l) + r(pred_x) + r(pred_c)) / 3
    auc_e = roc_auc_score(ybin, oof_e)
    np.savez(os.path.join(BASE_DIR, "clf_ens.npz"), oof=oof_e, pred=pred_e)
    log(f"[clf] AUC 汇总:lgb={auc_l:.5f} xgb={auc_x:.5f} cat={auc_c:.5f} "
        f"rank-ens={auc_e:.5f}(提升 {auc_e - auc_l:+.5f})")
    log("[clf] 注意:rank 平均输出是分位数而非概率,融合层的 isotonic 校准会把它映回概率尺度")


if __name__ == "__main__":
    main()
