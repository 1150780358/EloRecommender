# -*- coding: utf-8 -*-
"""v6-P5:商户侧折外目标编码(merchant-level outlier rate → 回聚到 card)。

动机:v6 已证明 card 侧目标编码是本项目最大的信息增量来源(单模 OOF −0.0077),
而增益主要来自高基数众数列(mode_city / mode_mcat)。但那仍是"每张卡最常去的那一个商户
类目"的粗略代理 —— 一张卡通常光顾几十个商户,众数丢掉了绝大部分结构。

本脚本把编码下沉到**交易粒度**:
    1. 折内:用训练折的 card 及其 outlier 标签,统计每个 merchant / merchant_category /
       city / subsector 的"顾客 outlier 率"(贝叶斯平滑);
    2. 把该率 map 回全部交易行(含验证折与 test 的 card 的交易);
    3. 按 card 聚合 mean/max/min/std/最近端加权 → card 粒度特征。

这是 card 侧特征完全无法表达的信息:**商户本身的"流失客户倾向"**。某些商户(如
一次性大额消费、清仓类)的顾客天然更容易流失,这个信号只存在于跨卡的商户维度上。
21st 的 `get_data_ctr_fea` 只编码了 card 侧属性,没有下沉到商户维度。

泄漏控制:统计只用训练折 card 的标签;验证折 card 的交易虽参与 map,但其标签从未进入
统计,与 v6_te.py 同口径。test 侧取十折统计的平均。

用法:ELO_SEED=777 python src/v6_merchant_te.py [build|lgb]
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep

CACHE = "outputs/merchant_te.npz"
OUT_DIR = "outputs/base_mte"
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()

# 编码哪些商户侧维度:基数从几万(merchant_id)到几百(subsector),覆盖不同粒度
MER_KEYS = [("merchant_id", "mid"), ("merchant_category_id", "mcat"),
            ("city_id", "city"), ("subsector_id", "subsec")]


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def load_hist():
    """只保留编码所需的列,避免 29M 行全表驻留。"""
    hist = ep.load_transactions("historical_transactions.csv", None)
    hist["card_id"] = hist["card_id"].astype(str)
    keep = ["card_id", "month_lag"] + [k for k, _ in MER_KEYS]
    hist = hist[keep]
    for k, _ in MER_KEYS:
        if hist[k].dtype == object:
            hist[k] = hist[k].fillna("M_ID_nan")
    log(f"hist(精简列)={hist.shape}")
    return hist


def build():
    base = pd.read_parquet("data/processed/features.parquet")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    ybin = (y < -30).astype(int).to_numpy()
    folds = ep.make_folds(y)
    n_tr, n_te = len(train), len(test)

    hist = load_hist()
    # 给所有 card 一个连续整数编号(train 在前、test 在后),把 29M 行的 groupby 从
    # 字符串键降为 int32 键 —— 字符串 groupby 在这个规模上要慢一个数量级。
    all_cards = np.concatenate([train["card_id"].astype(str).values,
                                test["card_id"].astype(str).values])
    cidx = pd.Series(np.arange(len(all_cards), dtype=np.int32), index=all_cards)
    cid = hist["card_id"].map(cidx)
    hist = hist[cid.notna()].copy()
    hist["cid"] = cid[cid.notna()].astype(np.int32).values
    n_all = len(all_cards)
    log(f"交易行对齐 card 编号:{hist.shape},覆盖 {hist['cid'].nunique()} / {n_all} 张卡")

    stats = ["mean", "max", "min", "std"]
    names = [f"mte_{tag}_{s}" for _, tag in MER_KEYS for s in stats] + \
            [f"mte_{tag}_recent" for _, tag in MER_KEYS]
    out_tr = np.zeros((n_tr, len(names)), np.float32)
    out_te = np.zeros((n_te, len(names)), np.float32)

    cid_arr = hist["cid"].to_numpy()
    recent = (hist["month_lag"] >= -2).to_numpy()
    is_train_row = cid_arr < n_tr                      # 该交易属于 train 卡
    tr_row = np.where(is_train_row, cid_arr, 0)        # 安全索引(非 train 行不会被用到)

    def by_card(vals, mask=None):
        """按 card 编号聚合成长度 n_all 的数组组(bincount 实现,比 groupby 快数倍)。"""
        c = cid_arr if mask is None else cid_arr[mask]
        v = vals if mask is None else vals[mask]
        cnt = np.bincount(c, minlength=n_all).astype(np.float64)
        ssum = np.bincount(c, weights=v, minlength=n_all)
        mean = np.divide(ssum, cnt, out=np.full(n_all, np.nan), where=cnt > 0)
        sq = np.bincount(c, weights=v.astype(np.float64) ** 2, minlength=n_all)
        var = np.divide(sq, cnt, out=np.full(n_all, np.nan), where=cnt > 0) - mean ** 2
        return mean, np.sqrt(np.clip(var, 0, None)), cnt

    for k, (tr, va) in enumerate(folds):
        prior = float(ybin[tr].mean())
        in_tr = np.zeros(n_tr, bool)
        in_tr[tr] = True
        m_stat = is_train_row & in_tr[tr_row]          # 只用该折训练卡的交易做统计
        lab = np.zeros(len(hist), np.float32)
        lab[m_stat] = ybin[tr_row[m_stat]].astype(np.float32)

        col = 0
        for ki, (key, tag) in enumerate(MER_KEYS):
            kv = hist[key]
            g = pd.DataFrame({"k": kv[m_stat].values, "y": lab[m_stat]}).groupby("k")["y"]
            rate = ((g.sum() + prior * 20.0) / (g.count() + 20.0)).astype(np.float32)
            mapped = kv.map(rate).fillna(prior).to_numpy(np.float32)

            mean, std, cnt = by_card(mapped)
            # max / min 用 ufunc.at 累积(bincount 无法求极值)
            mx = np.full(n_all, -np.inf, np.float32); np.maximum.at(mx, cid_arr, mapped)
            mn = np.full(n_all, np.inf, np.float32);  np.minimum.at(mn, cid_arr, mapped)
            rmean, _, rcnt = by_card(mapped, recent)

            for s, arr in zip(stats, [mean, mx, mn, std]):
                a = np.where(np.isfinite(arr), arr, prior).astype(np.float32)
                out_tr[va, col] = a[:n_tr][va]
                out_te[:, col] += a[n_tr:] / len(folds)
                col += 1
            ridx = len(MER_KEYS) * len(stats) + ki
            a = np.where(np.isfinite(rmean), rmean, prior).astype(np.float32)
            out_tr[va, ridx] = a[:n_tr][va]
            out_te[:, ridx] += a[n_tr:] / len(folds)
            del mapped, rate, mean, std, mx, mn, rmean
            gc.collect()
        log(f"  fold{k + 1} 完成")

    np.savez(CACHE, tr=out_tr, te=out_te, names=np.array(names, dtype=object))
    log(f"保存 {CACHE}:{out_tr.shape}({len(names)} 列)")


def train_lgb():
    z = np.load(CACHE, allow_pickle=True)
    mte_tr, mte_te, names = z["tr"], z["te"], list(z["names"])
    base = pd.read_parquet("data/processed/features.parquet")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)

    imp = pd.read_csv("outputs/feature_importance.csv")
    sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
    # 原实验与 card 侧 TE-v2 叠加;v2 已定性泄漏并于 2026-07-31 清除,改叠加 TE-v1(te_features.npz)。
    # 注意:当年 P5 证伪结论(OOF 3.63533 vs 3.63525)的对照基线是泄漏版 v2,若重跑需以 v1 基线重新评估。
    te = np.load("outputs/te_features.npz", allow_pickle=True)
    te_names = list(te["names"])

    X = pd.concat([train[sel].reset_index(drop=True),
                   pd.DataFrame(te["tr"], columns=te_names),
                   pd.DataFrame(mte_tr, columns=names)], axis=1)
    X_test = pd.concat([test[sel].reset_index(drop=True),
                        pd.DataFrame(te["te"], columns=te_names),
                        pd.DataFrame(mte_te, columns=names)], axis=1)
    log(f"X={X.shape}(原 {len(sel)} + cardTE {len(te_names)} + merchantTE {len(names)})")

    oof, pred, _, gain = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb+mte")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "lgb.npz"), oof=oof, pred=pred)
    log(f"判据:lgb+cardTE+merchantTE OOF={rmse(y, oof):.5f}(仅 cardTE-v2 为 3.63525)")
    imp_df = pd.DataFrame({"feature": X.columns, "gain": gain}).sort_values("gain", ascending=False)
    log("商户侧 TE 列 gain 前 8:\n" + imp_df[imp_df["feature"].str.startswith("mte_")].head(8).to_string(index=False))
    log(f"商户侧 TE 进入 gain 前 50 的个数:{int(imp_df.head(50)['feature'].str.startswith('mte_').sum())}")


if __name__ == "__main__":
    (build if (sys.argv[1:] or ["build"])[0] == "build" else train_lgb)()
