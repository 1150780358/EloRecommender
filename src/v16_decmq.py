# -*- coding: utf-8 -*-
"""v16:被拒交易时间结构 + merchants 增量(docs/202608/03-v16_dec时间结构.md)。

dec 族现有 5 列粗统计全在 gain 头部但时间结构空白;v14 NN 被拒序列增益佐证时间形状有信息。
判据:lgb(sel+TE+td+fm+dq)OOF vs outputs/base_fm/lgb.npz(777,3.63246)改善>0.0005。
用法:ELO_SEED=777 python src/v16_decmq.py [lgb|rest|all]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import v11_formula as v11

OUT_DIR = "outputs/base_dq"
REF_LGB = "outputs/base_fm/lgb.npz"
DQ_CACHE = "outputs/dec_mq.parquet"
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def dq_cache() -> pd.DataFrame:
    """hist 全量(auth 0/1)→ dec 时间结构 12 列 + merchants 增量 4 列。"""
    if os.path.exists(DQ_CACHE):
        return pd.read_parquet(DQ_CACHE)
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))
    hist["card_id"] = hist["card_id"].astype(str)
    hist["wkd"] = (hist["purchase_date"].dt.dayofweek >= 5).astype(np.float32)
    ok = hist[hist["authorized_flag"] == 1]
    dec = hist[hist["authorized_flag"] == 0]
    log(f"hist {hist.shape} 其中被拒 {dec.shape}")

    g = dec.groupby("card_id", observed=True)
    f = pd.DataFrame({
        "dq_lag_max": g["month_lag"].max(),
        "dq_lag_min": g["month_lag"].min(),
        "dq_lag_std": g["month_lag"].std(),
        "dq_amt_max": g["purchase_amount"].max(),
        "dq_wkd": g["wkd"].mean(),
        "dq_last_dec_ts": g["purchase_date"].max(),
    })
    r3 = dec[dec["month_lag"] >= -2].groupby("card_id", observed=True)
    f["dq_r3_cnt"] = r3.size()
    f["dq_r3_amt"] = r3["purchase_amount"].sum()
    dm = dec.groupby(["card_id", "merchant_id"], observed=True).size()
    f["dq_rep_mer"] = dm[dm >= 2].groupby("card_id", observed=True).size()

    o = ok.groupby("card_id", observed=True)
    f["dq_last_ok_ts"] = o["purchase_date"].max()
    f["dq_ok_cnt"] = o.size()
    f["dq_ok_r3_cnt"] = ok[ok["month_lag"] >= -2].groupby("card_id", observed=True).size()
    # 被拒后再无成功消费的笔数:purchase_date > 最后授权日的被拒交易
    d2 = dec.merge(f["dq_last_ok_ts"].rename("lok"), left_on="card_id", right_index=True, how="left")
    f["dq_after_ok"] = (d2[d2["purchase_date"] > d2["lok"]]
                        .groupby("card_id", observed=True).size())

    # merchants 增量:金额加权销售档 / 档位 std / top1 商户份额 / active_months_lag3
    m = pd.read_csv(os.path.join(ep.CONFIG["DATA_DIR"], "merchants.csv"))
    m = m.drop_duplicates("merchant_id", keep="first")
    m["rng"] = m["most_recent_sales_range"].map({"A": 4, "B": 3, "C": 2, "D": 1, "E": 0})
    m = m[["merchant_id", "rng", "active_months_lag3"]]
    j = ok[["card_id", "merchant_id", "purchase_amount"]].merge(m, on="merchant_id", how="left")
    j["w"] = np.clip(j["purchase_amount"].to_numpy(np.float64), 0, None)
    j["rw"] = j["rng"] * j["w"]
    gj = j.groupby("card_id", observed=True)
    f["mq_rng_wmean"] = gj["rw"].sum() / np.clip(gj["w"].sum(), 1e-9, None)
    f["mq_rng_std"] = gj["rng"].std()
    f["mq_active3"] = gj["active_months_lag3"].mean()
    top = (ok.groupby(["card_id", "merchant_id"], observed=True)["purchase_amount"].sum()
           .groupby("card_id", observed=True).max())
    f["mq_top1_share"] = top / np.clip(o["purchase_amount"].sum(), 1e-9, None)

    f = f.reset_index()
    f["card_id"] = f["card_id"].astype(str)
    f.to_parquet(DQ_CACHE)
    log(f"dq 缓存 {DQ_CACHE}: {f.shape}")
    return f


def dq_block(df: pd.DataFrame) -> pd.DataFrame:
    """派生最终 16 列(哨兵:无被拒卡 lag_max=-14,gap/率类填中性)。"""
    g = lambda c: df[c].to_numpy(np.float64) if c in df else np.full(len(df), np.nan)
    f = pd.DataFrame(index=df.index)
    f["dq_lag_max"] = np.nan_to_num(g("dq_lag_max"), nan=-14)
    f["dq_lag_min"] = np.nan_to_num(g("dq_lag_min"), nan=-14)
    f["dq_lag_std"] = np.nan_to_num(g("dq_lag_std"), nan=0.0)
    f["dq_amt_max"] = np.nan_to_num(g("dq_amt_max"), nan=0.0)
    f["dq_wkd"] = g("dq_wkd")
    r3c = np.nan_to_num(g("dq_r3_cnt"), nan=0.0)
    okr3 = np.nan_to_num(g("dq_ok_r3_cnt"), nan=0.0)
    f["dq_r3_cnt"] = r3c
    f["dq_r3_amt"] = np.nan_to_num(g("dq_r3_amt"), nan=0.0)
    f["dq_r3_rate"] = r3c / np.clip(r3c + okr3, 1, None)
    dc = np.nan_to_num(df["dec_count"].to_numpy(np.float64) if "dec_count" in df else np.zeros(len(df)), nan=0.0)
    okc = np.nan_to_num(g("dq_ok_cnt"), nan=0.0)
    f["dq_rate_delta"] = f["dq_r3_rate"] - dc / np.clip(dc + okc, 1, None)
    last_ok = pd.to_datetime(df["dq_last_ok_ts"]) if "dq_last_ok_ts" in df else pd.Series(pd.NaT, index=df.index)
    last_dec = pd.to_datetime(df["dq_last_dec_ts"]) if "dq_last_dec_ts" in df else pd.Series(pd.NaT, index=df.index)
    f["dq_last_gap"] = (last_ok - last_dec).dt.total_seconds().to_numpy() / 86400.0
    f["dq_after_ok"] = np.nan_to_num(g("dq_after_ok"), nan=0.0)
    f["dq_rep_mer"] = np.nan_to_num(g("dq_rep_mer"), nan=0.0)
    f["mq_rng_wmean"] = g("mq_rng_wmean")
    f["mq_rng_std"] = g("mq_rng_std")
    f["mq_active3"] = g("mq_active3")
    f["mq_top1_share"] = g("mq_top1_share")
    return f.astype(np.float32)


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    mode = sys.argv[1] if len(sys.argv) > 1 else "lgb"
    base = pd.read_parquet("data/processed/features.parquet")
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    base = base.merge(dq_cache(), on="card_id", how="left")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    fm_tr, fm_te = v11.formula_block(train), v11.formula_block(test)
    dq_tr, dq_te = dq_block(train), dq_block(test)
    ok = (y > -30).to_numpy()
    log("dq 特征 spearman(全量 | 非 outlier):")
    for c in dq_tr.columns:
        log(f"  {c:14s} {spearmanr(dq_tr[c], y, nan_policy='omit').statistic:+.4f} | "
            f"{spearmanr(dq_tr[c][ok], y[ok], nan_policy='omit').statistic:+.4f}")
    imp = pd.read_csv("outputs/feature_importance.csv")
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    z = np.load("outputs/te_features.npz", allow_pickle=True)
    te_names = [str(x) for x in z["names"]]
    td = pd.read_parquet("outputs/td_features.parquet")

    def assemble(side, zte, fm, dq):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True), dq.reset_index(drop=True)], axis=1)

    X = assemble(train, z["tr"], fm_tr, dq_tr)
    X_test = assemble(test, z["te"], fm_te, dq_te)
    log(f"X={X.shape}(sel {len(sel)} + TE {len(te_names)} + td {td.shape[1] - 1}"
        f" + fm {fm_tr.shape[1]} + dq {dq_tr.shape[1]})")
    os.makedirs(OUT_DIR, exist_ok=True)

    def dump(name, oof, pred):
        np.savez(os.path.join(OUT_DIR, f"{name}.npz"), oof=oof, pred=pred)
        log(f"[dq] {name:6s} OOF={rmse(y, oof):.5f} -> {OUT_DIR}/{name}.npz")

    if mode in ("lgb", "all"):
        oof, pred, _, gain = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb+dq")
        dump("lgb", oof, pred)
        ref = rmse(y, np.load(REF_LGB)["oof"])
        s = rmse(y, oof)
        d = ref - s
        alarm = "(⚠️ 超警报线,先审计)" if d > 0.003 else ""
        log(f"判据[v16 dec 时间结构]:OOF={s:.5f} vs fm 基线 {ref:.5f} → 改善 {d:+.5f} "
            f"{'✅ 通过' if d > 0.0005 else '❌ 不足'}{alarm}")
        g2 = pd.DataFrame({"feature": X.columns, "gain": gain}).sort_values("gain", ascending=False)
        pre = g2["feature"].str.startswith(("dq_", "mq_"))
        log("dq/mq 列 gain:\n" + g2[pre].to_string(index=False))
        log(f"dq/mq 进入 gain 前 50:{int(g2.head(50)['feature'].str.startswith(('dq_', 'mq_')).sum())}")
        if d <= 0.0005:
            sys.exit(3)

    if mode in ("rest", "all"):
        oof, pred, _ = ep.cv_xgboost(X, y, X_test, folds);            dump("xgb", oof, pred)
        oof, pred, _ = ep.cv_catboost(X, y, X_test, folds);           dump("cat", oof, pred)
        oof, pred, _, _ = ep.cv_lightgbm(X, y, X_test, folds, ep.HUB_PARAMS, "hub+dq"); dump("hub", oof, pred)
        oof, pred, auc = ep.cv_outlier_clf(X, y, X_test, folds)
        np.savez(os.path.join(OUT_DIR, "clf.npz"), oof=oof, pred=pred)
        log(f"[dq] clf AUC={auc:.5f}(fm 版 0.90586)")
        mask = (y < -30).to_numpy()
        oc, pc = np.zeros(len(X)), np.zeros(len(X_test))
        import lightgbm as lgb_
        for k, (tr, va) in enumerate(folds):
            tr_c, va_c = tr[~mask[tr]], va[~mask[va]]
            m = lgb_.train(ep.LGB_PARAMS, lgb_.Dataset(X.iloc[tr_c], y.iloc[tr_c]), 10000,
                           valid_sets=[lgb_.Dataset(X.iloc[va_c], y.iloc[va_c])],
                           callbacks=[lgb_.early_stopping(200, verbose=False)])
            oc[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
            pc += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
            log(f"  [clean+dq] fold{k + 1} iter={m.best_iteration}")
        dump("clean", oc, pc)


if __name__ == "__main__":
    main()
