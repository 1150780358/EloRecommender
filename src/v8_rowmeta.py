# -*- coding: utf-8 -*-
"""v8:Top1 行级元特征(transaction-level meta features)。

来源(一手,见 docs/202607/31-Top1方案挖掘报告.md):
  冠军 #82036 亲口指认的"most important features"(Home Credit 64503#378162 做法,
  KALE 解读获冠军确认"kale is right")+ senkin 7th #82055 回归变体(CV/LB +0.005~0.006)。

做法:hist+new 合成一张交易行表,把**卡级标签**贴到每条交易行,行级训 LGB:
  or_:标签 = 该卡是否 outlier(0/1),二分类 —— "这条交易属于 outlier 卡的概率";
  tm_:标签 = 该卡 loyalty target,回归(senkin 变体);
行级预测按 card_id 聚合(mean/max/min/std/sum + hist/new 分侧 mean)= 每目标 7 列,共 14 列。

折纪律(与 v6 TE 同等收紧):行级模型的折 = 主 CV 十折按 card 划分,行只属于其卡的折;
  train 侧 = 折外预测;test 侧 = 十折模型平均。
⚠️ 本特征是折依赖的(train 单折模型 vs test 十折平均)—— v6-v2 的分布错配通道存在,
  但每卡由成百上千行聚合,方差被大幅平滑;判据后仍须对抗验证审计 + 线上单候选验证。

判据:单模 lgb(sel+TE+td+rm)OOF vs base_td/lgb 3.63520 改善>0.0005;
     clf AUC 对标:我们 0.90507 → 冠军 0.914(评论区对照:AUC<0.91 期望式融合不 work)。

用法:python src/v8_rowmeta.py feat|lgb|clf|all
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, roc_auc_score

import elo_pipeline as ep

RM_CACHE = "outputs/rowmeta_features.parquet"
OUT_DIR = "outputs/base_rm"
REF_LGB = "outputs/base_td/lgb.npz"        # v7 后最优单模基线 3.63520
TD_CACHE = "outputs/td_features.parquet"
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()

# 行级数据 2800 万行,与卡级"16 线程最优"的结论不同,大数据吃得下更多线程
ROW_BIN = dict(objective="binary", metric="auc", learning_rate=0.05,
               num_leaves=31, min_data_in_leaf=5000, feature_fraction=0.8,
               bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
               verbosity=-1, num_threads=32, seed=ep.CONFIG["SEED"])
ROW_REG = {**ROW_BIN, "objective": "regression", "metric": "rmse"}
CAT_COLS = ["cat2", "cat3", "city", "state", "subsector", "mcat", "hour", "dow", "month"]


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_rows() -> pd.DataFrame:
    """交易行表:原始列 + 时间派生列(与 v7 时间差同口径),hist/new 合并。"""
    fam = pd.concat([
        pd.read_csv(os.path.join(ep.CONFIG["DATA_DIR"], f), usecols=["card_id", "first_active_month"])
        for f in ("train.csv", "test.csv")])
    fam = pd.Series(pd.to_datetime(fam["first_active_month"].values, format="%Y-%m", errors="coerce"),
                    index=fam["card_id"].values)
    now = pd.Timestamp(ep.CONFIG["REF_DATE"])
    frames = []
    for fname, isnew in (("historical_transactions.csv", 0), ("new_merchant_transactions.csv", 1)):
        df = ep.clean_transactions(ep.load_transactions(fname))
        mid = df["purchase_date"].dt.year * 12 + (df["purchase_date"].dt.month - 1)
        ref = (mid - df["month_lag"] + 1).astype(np.int32)
        ref_end = pd.to_datetime(pd.DataFrame({"year": ref // 12, "month": ref % 12 + 1, "day": 1}))
        out = pd.DataFrame({
            "card_id": df["card_id"].astype(str),
            "amount": df["purchase_amount"].astype(np.float32),
            "month_lag": df["month_lag"].astype(np.int8),
            "installments": df["installments"].astype(np.float32),
            "auth": df["authorized_flag"].astype(np.int8),
            "cat1": df["category_1"].astype(np.int8),
            "cat2": df["category_2"].astype(np.int8),
            "cat3": df["category_3"].astype(np.int8),
            "city": df["city_id"].astype(np.int16),
            "state": df["state_id"].astype(np.int8),
            "subsector": df["subsector_id"].astype(np.int8),
            "mcat": df["merchant_category_id"].astype(np.int16),
            "hour": df["purchase_date"].dt.hour.astype(np.int8),
            "dow": df["purchase_date"].dt.dayofweek.astype(np.int8),
            "day": df["purchase_date"].dt.day.astype(np.int8),
            "month": df["purchase_date"].dt.month.astype(np.int8),
            "a2p": (df["purchase_date"] - df["card_id"].astype(str).map(fam)).dt.days.astype(np.float32),
            "p2r": (ref_end - df["purchase_date"]).dt.days.astype(np.float32),
            "p2now": (now - df["purchase_date"]).dt.days.astype(np.float32),
            "is_new": np.int8(isnew),
        })
        frames.append(out)
        del df
        gc.collect()
    rows = pd.concat(frames, ignore_index=True)
    log(f"行表 {rows.shape}")
    return rows


def row_meta(rows, train, y, ybin, folds):
    """行级折外训练两个目标,返回每行的 (or 概率, tm 预测)。test 行 = 十折平均。"""
    card_fold = {}
    cards = train["card_id"].to_numpy()
    for k, (_, va) in enumerate(folds):
        for c in cards[va]:
            card_fold[c] = k
    fr = rows["card_id"].map(card_fold).fillna(-1).to_numpy(np.int8)   # -1 = test 卡
    yb_card = dict(zip(cards, ybin))
    ym_card = dict(zip(cards, y.to_numpy(np.float32)))
    yb = rows["card_id"].map(yb_card).to_numpy(np.float32)
    ym = rows["card_id"].map(ym_card).to_numpy(np.float32)
    feat_cols = [c for c in rows.columns if c != "card_id"]
    rowf = rows[feat_cols]
    test_m = fr == -1
    n = len(rows)
    or_v = np.zeros(n, np.float32)
    tm_v = np.zeros(n, np.float32)
    log(f"行级训练:{n} 行(test 行 {int(test_m.sum())}),特征 {len(feat_cols)}")
    for k in range(len(folds)):
        tr_m = (fr >= 0) & (fr != k)
        va_m = fr == k
        for tag, params, lab, acc in (("or", ROW_BIN, yb, or_v), ("tm", ROW_REG, ym, tm_v)):
            m = lgb.train(params, lgb.Dataset(rowf[tr_m], lab[tr_m], categorical_feature=CAT_COLS), 2000,
                          valid_sets=[lgb.Dataset(rowf[va_m], lab[va_m], categorical_feature=CAT_COLS)],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            # v2 修正:折内百分位秩归一 —— v1 直接聚合原始预测,train 侧各折模型不稳定
            # (15-31 轮早停、分布互不相同)而 test 侧是十折平均,分布错配导致 OOF 变差
            # (v6-v2 同病根)。秩归一后每折 val 行都是均匀分布,尺度错配从机制上消除。
            pv = m.predict(rowf[va_m], num_iteration=m.best_iteration)
            acc[va_m] = pd.Series(pv).rank(pct=True).to_numpy(np.float32)
            acc[test_m] += m.predict(rowf[test_m], num_iteration=m.best_iteration) / len(folds)
            score = (roc_auc_score(yb[va_m], pv) if tag == "or"
                     else rmse(ym[va_m], pv))
            log(f"  fold{k + 1} [{tag}] iter={m.best_iteration} 行级{'AUC' if tag == 'or' else 'RMSE'}={score:.5f}")
            del m
        gc.collect()
    # test 行:十折平均后同样整体秩归一,与 train 侧口径一致
    for acc in (or_v, tm_v):
        acc[test_m] = pd.Series(acc[test_m]).rank(pct=True).to_numpy(np.float32)
    return or_v, tm_v


def aggregate(rows, or_v, tm_v) -> pd.DataFrame:
    d = pd.DataFrame({"card_id": rows["card_id"], "is_new": rows["is_new"],
                      "or_": or_v, "tm_": tm_v})
    parts = []
    for tag in ("or_", "tm_"):
        g = d.groupby("card_id")[tag]
        a = g.agg(["mean", "max", "min", "std", "sum"])
        a.columns = [f"rm_{tag}{s}" for s in a.columns]
        side = d.groupby(["card_id", "is_new"])[tag].mean().unstack()
        side.columns = [f"rm_{tag}hist_mean", f"rm_{tag}new_mean"][:len(side.columns)]
        parts.append(a.join(side))
    res = parts[0].join(parts[1]).reset_index()
    res.to_parquet(RM_CACHE)
    log(f"行级元特征缓存 {RM_CACHE}: {res.shape}")
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    base = pd.read_parquet("data/processed/features.parquet")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    ybin = (y < -30).astype(int).to_numpy()
    folds = ep.make_folds(y)

    if mode in ("feat", "all") or not os.path.exists(RM_CACHE):
        rows = build_rows()
        or_v, tm_v = row_meta(rows, train, y, ybin, folds)
        aggregate(rows, or_v, tm_v)
        del rows
        gc.collect()
    if mode == "feat":
        return

    imp = pd.read_csv("outputs/feature_importance.csv")
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    z = np.load("outputs/te_features.npz", allow_pickle=True)
    te_tr, te_te, te_names = z["tr"], z["te"], [str(x) for x in z["names"]]
    td = pd.read_parquet(TD_CACHE)
    rm = pd.read_parquet(RM_CACHE)

    def assemble(side):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        m2 = side[["card_id"]].merge(rm, on="card_id", how="left").drop(columns="card_id")
        return m1.astype(np.float32), m2.astype(np.float32)

    td_tr, rm_tr = assemble(train)
    td_te, rm_te = assemble(test)
    X = pd.concat([train[sel].reset_index(drop=True), pd.DataFrame(te_tr, columns=te_names),
                   td_tr, rm_tr], axis=1)
    X_test = pd.concat([test[sel].reset_index(drop=True), pd.DataFrame(te_te, columns=te_names),
                        td_te, rm_te], axis=1)
    log(f"X={X.shape}(sel {len(sel)} + TE {len(te_names)} + td {td_tr.shape[1]} + rm {rm_tr.shape[1]})")
    os.makedirs(OUT_DIR, exist_ok=True)

    if mode in ("lgb", "all"):
        oof, pred, _, gain = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb+rm")
        np.savez(os.path.join(OUT_DIR, "lgb.npz"), oof=oof, pred=pred)
        ref = rmse(y, np.load(REF_LGB)["oof"])
        s = rmse(y, oof)
        d = ref - s
        alarm = "(⚠️ 超泄漏警报线 0.003,须对抗验证审计后再上线)" if d > 0.003 else ""
        log(f"判据[回归]:OOF={s:.5f} vs 基线 {ref:.5f} → 改善 {d:+.5f} "
            f"{'✅ 通过' if d > 0.0005 else '❌ 不足'}{alarm}")
        g = pd.DataFrame({"feature": X.columns, "gain": gain}).sort_values("gain", ascending=False)
        log("rm 列 gain 前 8:\n" + g[g["feature"].str.startswith("rm_")].head(8).to_string(index=False))

    if mode in ("clf", "all"):
        oof, pred, auc = ep.cv_outlier_clf(X, y, X_test, folds)
        np.savez(os.path.join(OUT_DIR, "clf.npz"), oof=oof, pred=pred)
        log(f"判据[分类]:AUC={auc:.5f} vs td 版 0.90507,冠军对标 0.914 "
            f"{'✅ 超冠军' if auc > 0.914 else ('✅ 破0.91' if auc > 0.91 else '继续追')}")


if __name__ == "__main__":
    main()
