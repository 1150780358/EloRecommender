# -*- coding: utf-8 -*-
"""v10:10th 名月度序列族(bolkonsky #82093,一手补挖)。

来源:docs/202607/31-前十writeup覆盖审计.md —— #10 帖此前从未读过,本族全部折外无依赖
(纯行为统计/自监督预测),按转化率定律(v7 119%)属高先验。

特征(hist[auth=1] + new 合并的逐月金额/笔数序列,month_lag -13..+2):
  A 逐月比值链:amt[m]/amt[m-1](m=+2..-3)+ 间隔 2/4/6 比值(m=+2..0)—— 10th 原文 intervals 2,4,6
  B 滚动均值:roll3(0..+2) / roll3(-3..-1) 及其比值
  C 指数平滑:alpha 0.3/0.6 递推 level(SimpleExpSmoothing 平替)及 level/月均比
  D lag+3/+4 预测:近 6 月线性趋势外推 f3/f4,取 f3/amt[+2]、f3/月均、f4/f3(10th"预测未来月并取比值")
  E 笔数比值链(近月)
  F 交易间隔分布:相邻交易天数差 mean/min/max/std × hist/new(7th interval 族顺手并入)

判据:单模 lgb(sel+TE+td+ms)OOF vs outputs/base_td/lgb.npz(777 规范)改善>0.0005。
用法:ELO_SEED=777 python src/v10_monthly.py feat|lgb|all
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep

MS_CACHE = "outputs/monthly_features.parquet"
OUT_DIR = "outputs/base_ms"
REF_LGB = "outputs/base_td/lgb.npz"
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()
LAGS = list(range(-13, 3))


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_ms() -> pd.DataFrame:
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))
    new = ep.clean_transactions(ep.load_transactions("new_merchant_transactions.csv"))
    # F:交易间隔分布(全量 hist 与 new 分侧)
    parts = []
    for df, pre in ((hist, "hist"), (new, "new")):
        d = df.sort_values(["card_id", "purchase_date"])
        gap = d.groupby("card_id", observed=True)["purchase_date"].diff().dt.total_seconds() / 86400.0
        g = pd.DataFrame({"card_id": d["card_id"], "gap": gap}).dropna().groupby("card_id", observed=True)["gap"]
        a = g.agg(["mean", "min", "max", "std"])
        a.columns = [f"{pre}_ms_gap_{c}" for c in a.columns]
        parts.append(a)
        log(f"[F] {pre} 间隔分布 {a.shape}")
    # 月度序列:hist 取 auth=1(10th 原文口径)+ new 全量
    rows = pd.concat([hist[hist["authorized_flag"] == 1][["card_id", "month_lag", "purchase_amount"]],
                      new[["card_id", "month_lag", "purchase_amount"]]], ignore_index=True)
    del hist, new
    g = rows.groupby(["card_id", "month_lag"], observed=True)["purchase_amount"]
    amt = g.sum().unstack().reindex(columns=LAGS).fillna(0.0)
    cnt = g.count().unstack().reindex(columns=LAGS).fillna(0.0)
    log(f"月度矩阵 {amt.shape}")
    A = amt.to_numpy(np.float64)
    C = cnt.to_numpy(np.float64)
    idx = {m: i for i, m in enumerate(LAGS)}
    f = pd.DataFrame(index=amt.index)
    div = lambda a, b: np.where(np.abs(b) > 1e-9, a / b, np.nan)
    # A 比值链
    for m in (2, 1, 0, -1, -2, -3):
        f[f"ms_r1_{m}"] = div(A[:, idx[m]], A[:, idx[m - 1]])
    for k in (2, 4, 6):
        for m in (2, 1, 0):
            f[f"ms_r{k}_{m}"] = div(A[:, idx[m]], A[:, idx[m - k]])
    # B 滚动均值
    roll_new = A[:, idx[0]:idx[2] + 1].mean(1)
    roll_old = A[:, idx[-3]:idx[-1] + 1].mean(1)
    f["ms_roll3_new"], f["ms_roll3_old"] = roll_new, roll_old
    f["ms_roll3_ratio"] = div(roll_new, roll_old)
    # C 指数平滑
    mean_amt = A.mean(1)
    for al in (0.3, 0.6):
        lv = A[:, 0].copy()
        for j in range(1, len(LAGS)):
            lv = al * A[:, j] + (1 - al) * lv
        f[f"ms_es{int(al * 10)}"] = lv
        f[f"ms_es{int(al * 10)}_vs_mean"] = div(lv, mean_amt)
    # D 线性趋势外推 lag+3/+4(近 6 月 -3..+2)
    W = A[:, idx[-3]:idx[2] + 1]
    x = np.arange(6, dtype=np.float64)
    slope = ((x - x.mean()) * (W - W.mean(1, keepdims=True))).sum(1) / ((x - x.mean()) ** 2).sum()
    last = W[:, -1]
    f3, f4 = last + slope, last + 2 * slope
    f["ms_f3"], f["ms_f4"], f["ms_slope6"] = f3, f4, slope
    f["ms_f3_vs_last"] = div(f3, last)
    f["ms_f3_vs_mean"] = div(f3, mean_amt)
    f["ms_f4_vs_f3"] = div(f4, f3)
    # E 笔数比值链
    for m in (2, 1, 0):
        f[f"ms_c1_{m}"] = div(C[:, idx[m]], C[:, idx[m - 1]])
        f[f"ms_c2_{m}"] = div(C[:, idx[m]], C[:, idx[m - 2]])
    f.index = f.index.astype(str)
    for a in parts:
        a.index = a.index.astype(str)
        f = f.join(a, how="outer")
    res = f.reset_index().rename(columns={"index": "card_id"})
    res.to_parquet(MS_CACHE)
    log(f"月度序列族缓存 {MS_CACHE}: {res.shape}")
    return res


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("feat", "all") or not os.path.exists(MS_CACHE):
        build_ms()
    if mode == "feat":
        return
    base = pd.read_parquet("data/processed/features.parquet")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    imp = pd.read_csv("outputs/feature_importance.csv")
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    z = np.load("outputs/te_features.npz", allow_pickle=True)
    te_names = [str(x) for x in z["names"]]
    td = pd.read_parquet("outputs/td_features.parquet")
    ms = pd.read_parquet(MS_CACHE)

    def assemble(side, zte):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        m2 = side[["card_id"]].merge(ms, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), m2.astype(np.float32)], axis=1)

    X = assemble(train, z["tr"])
    X_test = assemble(test, z["te"])
    log(f"X={X.shape}(sel {len(sel)} + TE {len(te_names)} + td {td.shape[1] - 1} + ms {ms.shape[1] - 1})")
    os.makedirs(OUT_DIR, exist_ok=True)
    oof, pred, _, gain = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb+ms")
    np.savez(os.path.join(OUT_DIR, "lgb.npz"), oof=oof, pred=pred)
    ref = rmse(y, np.load(REF_LGB)["oof"])
    s = rmse(y, oof)
    d = ref - s
    alarm = "(⚠️ 超警报线,先审计)" if d > 0.003 else ""
    log(f"判据[v10 月度序列族]:OOF={s:.5f} vs 基线 {ref:.5f} → 改善 {d:+.5f} "
        f"{'✅ 通过' if d > 0.0005 else '❌ 不足'}{alarm}")
    g = pd.DataFrame({"feature": X.columns, "gain": gain}).sort_values("gain", ascending=False)
    log("ms 列 gain 前 10:\n" + g[g["feature"].str.contains("ms_")].head(10).to_string(index=False))
    log(f"ms 列进入 gain 前 50 的个数:{int(g.head(50)['feature'].str.contains('ms_').sum())}")


if __name__ == "__main__":
    main()
