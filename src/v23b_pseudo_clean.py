# -*- coding: utf-8 -*-
"""v23-B:伪标签去泄漏复验(leave-one-fold-out 自蒸馏;docs/202608/04-v23_伪标签终验.md)。

上界版 +0.00806 超泄漏警报线(0.003),须去泄漏:fold k 的伪标签只能来自
fold k 自己的模型(训练时未见 fold k)的测试预测 —— 唯一无泄漏通道的教师。
Pass1 复现 fm-LGB 十折并保存逐折测试预测;Pass2 用折内配对伪标签重训。
判据:OOF vs 3.63246,Δ>0.0005。
用法:ELO_SEED=777 python src/v23b_pseudo_clean.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import v11_formula as v11

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    base = pd.read_parquet("data/processed/features.parquet")
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    fm_tr, fm_te = v11.formula_block(train), v11.formula_block(test)
    imp = pd.read_csv("outputs/feature_importance.csv")
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    z = np.load("outputs/te_features.npz", allow_pickle=True)
    te_names = [str(x) for x in z["names"]]
    td = pd.read_parquet("outputs/td_features.parquet")

    def asm(side, zte, fm):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True)], axis=1)

    X, X_test = asm(train, z["tr"], fm_tr), asm(test, z["te"], fm_te)
    log(f"X={X.shape};Pass1 逐折教师")

    # Pass1:复现 fm-LGB 十折,保存逐折测试预测(fold k 模型未见 fold k)
    per_fold = np.zeros((len(folds), len(X_test)))
    oof1 = np.zeros(len(X))
    for k, (tr, va) in enumerate(folds):
        m = lgb.train(ep.LGB_PARAMS, lgb.Dataset(X.iloc[tr], y.iloc[tr]), 10000,
                      valid_sets=[lgb.Dataset(X.iloc[va], y.iloc[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof1[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        per_fold[k] = m.predict(X_test, num_iteration=m.best_iteration)
        log(f"  [t] fold{k + 1}: rmse={rmse(y.iloc[va], oof1[va]):.5f} iter={m.best_iteration}")
    log(f"Pass1 教师 OOF={rmse(y, oof1):.5f}(应≈3.63246)")

    # Pass2:fold k 用 per_fold[k] 做伪标签(教师未见该折,无泄漏)
    W_PL = 0.5
    oof, pred = np.zeros(len(X)), np.zeros(len(X_test))
    for k, (tr, va) in enumerate(folds):
        Xtr = pd.concat([X.iloc[tr], X_test], ignore_index=True)
        ytr = pd.concat([y.iloc[tr], pd.Series(per_fold[k])], ignore_index=True)
        w = np.r_[np.ones(len(tr)), np.full(len(X_test), W_PL)]
        m = lgb.train(ep.LGB_PARAMS, lgb.Dataset(Xtr, ytr, weight=w), 10000,
                      valid_sets=[lgb.Dataset(X.iloc[va], y.iloc[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        pred += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
        log(f"  [pl] fold{k + 1}: rmse={rmse(y.iloc[va], oof[va]):.5f} iter={m.best_iteration}")
    os.makedirs("outputs/base_plc", exist_ok=True)
    np.savez("outputs/base_plc/lgb.npz", oof=oof, pred=pred, per_fold_teacher=per_fold)
    ref = rmse(y, np.load("outputs/base_fm/lgb.npz")["oof"])
    s = rmse(y, oof)
    d = ref - s
    log(f"判据[v23-B pseudo-label 去泄漏版]:OOF={s:.5f} vs fm 基线 {ref:.5f} → 改善 {d:+.5f} "
        f"{'✅ 真信号' if d > 0.0005 else '❌ 上界版增益系泄漏假象'}")
    if d <= 0.0005:
        sys.exit(3)


if __name__ == "__main__":
    main()
