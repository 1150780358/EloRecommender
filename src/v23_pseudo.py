# -*- coding: utf-8 -*-
"""v23:伪标签终验(docs/202608/04-v23_伪标签终验.md)。

最后一个未试过的标准技术:v14 融合测试预测作软标签,test 12万卡并入训练(权重 0.5)。
故意用全量融合预测(含 CV 乐观偏置)做上界版本 —— 上界都不过 0.0005 即确定性死亡;
若过了,须再做 leave-one-fold-out 去泄漏版才可信(见文档)。
判据:单模 lgb(sel+TE+td+fm,+PL 行)vs outputs/base_fm/lgb.npz(3.63246)>0.0005。
用法:ELO_SEED=777 python src/v23_pseudo.py
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
    pl = pd.read_csv("outputs/submission_v14_nn3.csv").set_index("card_id")["target"]
    y_pl = test["card_id"].map(pl)
    assert y_pl.notna().all(), "v14 提交与 test 卡不齐"
    log(f"X={X.shape} + PL {len(X_test)} 行(软标签范围 [{y_pl.min():.2f},{y_pl.max():.2f}],"
        f"train 真标签 [{y.min():.2f},{y.max():.2f}])")

    W_PL = 0.5
    oof, pred = np.zeros(len(X)), np.zeros(len(X_test))
    for k, (tr, va) in enumerate(folds):
        Xtr = pd.concat([X.iloc[tr], X_test], ignore_index=True)
        ytr = pd.concat([y.iloc[tr], y_pl], ignore_index=True)
        w = np.r_[np.ones(len(tr)), np.full(len(X_test), W_PL)]
        m = lgb.train(ep.LGB_PARAMS, lgb.Dataset(Xtr, ytr, weight=w), 10000,
                      valid_sets=[lgb.Dataset(X.iloc[va], y.iloc[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        pred += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
        log(f"  [pl] fold{k + 1}: rmse={rmse(y.iloc[va], oof[va]):.5f} iter={m.best_iteration}")
    os.makedirs("outputs/base_pl", exist_ok=True)
    np.savez("outputs/base_pl/lgb.npz", oof=oof, pred=pred)
    ref = rmse(y, np.load("outputs/base_fm/lgb.npz")["oof"])
    s = rmse(y, oof)
    d = ref - s
    log(f"判据[v23 pseudo-label 上界版]:OOF={s:.5f} vs fm 基线 {ref:.5f} → 改善 {d:+.5f} "
        f"{'⚠️ 过门槛(须去泄漏复验)' if d > 0.0005 else '❌ 确定性死亡(上界即不足)'}")
    if d <= 0.0005:
        sys.exit(3)


if __name__ == "__main__":
    main()
