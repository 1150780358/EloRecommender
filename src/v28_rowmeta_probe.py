# -*- coding: utf-8 -*-
"""v28:19th 行级元特征 OOF-盲探针(docs/202608/04-v28_19th行级探针.md)。

19th 赛后补记:行级回归 target → 按卡聚合(min/mean/median/max/sum/std),
**Private +0.003 / Public −0.003 / CV 看不到** —— 与 v8 离线证伪方向相反的第三方私榜实证。
本脚本 = 忠实复刻(vs v8 的三处偏离):仅回归版、原始预测聚合(无秩归一)、含 median。
探针协议(预注册):A/B 单模对照 —— f_lgb 单模(已提交)vs 本脚本 +rm 单模,
**无视 OOF 判据直接提交**,私榜差值隔离 rm 效应;|Δprivate|≥0.0015 判有效。
用法:ELO_SEED=777 python src/v28_rowmeta_probe.py
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import v11_formula as v11
from v8_rowmeta import build_rows, ROW_REG, CAT_COLS

RM_CACHE = "outputs/rm777_tm_raw.parquet"
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_rm(train, y, folds) -> pd.DataFrame:
    if os.path.exists(RM_CACHE):
        return pd.read_parquet(RM_CACHE)
    rows = build_rows()
    card_fold = {}
    cards = train["card_id"].astype(str).to_numpy()
    for k, (_, va) in enumerate(folds):
        for c in cards[va]:
            card_fold[c] = k
    fr = rows["card_id"].map(card_fold).fillna(-1).to_numpy(np.int8)
    ym = rows["card_id"].map(dict(zip(cards, y.to_numpy(np.float32)))).to_numpy(np.float32)
    feat_cols = [c for c in rows.columns if c != "card_id"]
    rowf = rows[feat_cols]
    test_m = fr == -1
    tm_v = np.zeros(len(rows), np.float32)
    log(f"行级训练(tm raw):{len(rows)} 行,特征 {len(feat_cols)}")
    for k in range(10):
        tr_m = (fr >= 0) & (fr != k)
        va_m = fr == k
        m = lgb.train(ROW_REG, lgb.Dataset(rowf[tr_m], ym[tr_m], categorical_feature=CAT_COLS), 2000,
                      valid_sets=[lgb.Dataset(rowf[va_m], ym[va_m], categorical_feature=CAT_COLS)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        tm_v[va_m] = m.predict(rowf[va_m], num_iteration=m.best_iteration)   # 原始预测,无秩归一
        tm_v[test_m] += m.predict(rowf[test_m], num_iteration=m.best_iteration) / 10
        log(f"  fold{k + 1}: iter={m.best_iteration} 行级 RMSE={rmse(ym[va_m], tm_v[va_m]):.5f}")
        del m
        gc.collect()
    d = pd.DataFrame({"card_id": rows["card_id"], "is_new": rows["is_new"], "tm": tm_v})
    g = d.groupby("card_id")["tm"]
    a = g.agg(["min", "mean", "median", "max", "sum", "std"])
    a.columns = [f"rm_tm_{s}" for s in a.columns]
    side = d.groupby(["card_id", "is_new"])["tm"].mean().unstack()
    side.columns = ["rm_tm_hist_mean", "rm_tm_new_mean"][:len(side.columns)]
    res = a.join(side).reset_index()
    res.to_parquet(RM_CACHE)
    log(f"rm 特征缓存 {RM_CACHE}: {res.shape}")
    del rows
    gc.collect()
    return res


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    base = pd.read_parquet("data/processed/features.parquet")
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    rm = build_rm(train, y, folds)
    base = base.merge(rm, on="card_id", how="left")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    fm_tr, fm_te = v11.formula_block(train), v11.formula_block(test)
    imp = pd.read_csv("outputs/feature_importance.csv")
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    z = np.load("outputs/te_features.npz", allow_pickle=True)
    te_names = [str(x) for x in z["names"]]
    td = pd.read_parquet("outputs/td_features.parquet")
    rm_cols = [c for c in rm.columns if c != "card_id"]

    def asm(side, zte, fm):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True),
                          side[rm_cols].reset_index(drop=True).astype(np.float32)], axis=1)

    X, X_test = asm(train, z["tr"], fm_tr), asm(test, z["te"], fm_te)
    log(f"X={X.shape}(含 rm {len(rm_cols)})")
    oof, pred, _, gain = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb+rm")
    os.makedirs("outputs/base_rm777", exist_ok=True)
    np.savez("outputs/base_rm777/lgb.npz", oof=oof, pred=pred)
    ref = rmse(y, np.load("outputs/base_fm/lgb.npz")["oof"])
    s = rmse(y, oof)
    g2 = pd.DataFrame({"feature": X.columns, "gain": gain}).sort_values("gain", ascending=False)
    log("rm 列 gain 排名前 8:\n" + g2[g2["feature"].str.startswith("rm_")].head(8).to_string(index=False))
    log(f"OOF={s:.5f} vs f_lgb {ref:.5f}(Δ={ref - s:+.5f})——预注册探针:无视此数,直接提交")
    sub = pd.read_csv(os.path.join(ep.CONFIG["DATA_DIR"], "sample_submission.csv"))
    sub["target"] = pred
    sub.to_csv("outputs/submission_v28_rm_single.csv", index=False)
    log("已保存 outputs/submission_v28_rm_single.csv")


if __name__ == "__main__":
    main()
