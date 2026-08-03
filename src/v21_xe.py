# -*- coding: utf-8 -*-
"""v21-A:xentropy 目标基模型(docs/202608/03-v21_榜单二次考古.md)。

1st/7th 配方:target MinMax→[0,1],LGB objective=xentropy(连续标签交叉熵),
对 outlier 端的梯度形状与 L2/Huber 完全不同 —— 目标函数轴的异构成员。
判据:同池对照(F31 全成员 ± e_xe,bayes),ΔOOF>0.0005。
用法:ELO_SEED=777 python src/v21_xe.py
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
import v5_fusion as vf

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
    lo, hi = float(y.min()), float(y.max())
    ys = ((y - lo) / (hi - lo)).clip(0, 1)
    log(f"X={X.shape} target minmax [{lo:.3f},{hi:.3f}]→[0,1]")

    params = {**ep.LGB_PARAMS, "objective": "xentropy", "metric": "xentropy"}
    oof01, pred01 = np.zeros(len(X)), np.zeros(len(X_test))
    for k, (tr, va) in enumerate(folds):
        m = lgb.train(params, lgb.Dataset(X.iloc[tr], ys.iloc[tr]), 10000,
                      valid_sets=[lgb.Dataset(X.iloc[va], ys.iloc[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof01[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        pred01 += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
        log(f"  [xe] fold{k + 1}: rmse={rmse(y.iloc[va], oof01[va] * (hi - lo) + lo):.5f} iter={m.best_iteration}")
    oof = oof01 * (hi - lo) + lo
    pred = pred01 * (hi - lo) + lo
    os.makedirs("outputs/base_xe", exist_ok=True)
    np.savez("outputs/base_xe/xe.npz", oof=oof, pred=pred)
    s = rmse(y, oof)
    f_lgb = np.load("outputs/base_fm/lgb.npz")["oof"]
    log(f"[xe] 单模 OOF={s:.5f}(f_lgb 3.63246,预测相关 {np.corrcoef(oof, f_lgb)[0, 1]:.4f})")

    # 同池对照判据:F31 全成员 ± e_xe
    bases = vf.load_bases()
    bases["e_xe"] = (oof, pred)
    ybin = (y < -30).astype(int).to_numpy()
    REG = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    T = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    D = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    F = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    N = sorted(k for k in bases if k.startswith("n_"))
    allf = (REG + T + D + F + ["t_clf", "t_clean", "d_clf", "d_clean",
            "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"] + N)
    r0, _, _ = vf.evaluate(allf, "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
    r1, _, pt = vf.evaluate(allf + ["e_xe"], "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
    d = r0 - r1
    log(f"判据[v21-A xentropy]:基线={r0:.5f} +xe={r1:.5f} → Δ={d:+.5f} "
        f"{'✅ 通过' if d > 0.0005 else '❌ 不足'}")
    if d > 0.0005:
        sub = pd.read_csv(os.path.join(ep.CONFIG["DATA_DIR"], "sample_submission.csv"))
        sub["target"] = pt
        sub.to_csv("outputs/submission_v21_xe.csv", index=False)
        log("已保存 outputs/submission_v21_xe.csv")
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
