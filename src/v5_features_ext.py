# -*- coding: utf-8 -*-
"""v5 阶段 C:增量特征族(被拒交易完整画像 + 观察期末端近窗聚合)。

为什么是这两族 —— v5 融合层实验已证明:模型层/融合层的 OOF 改善不再转化为
Private 改善(OOF -0.00285 → Private 仅 -0.00006,CV-LB 偏移从 0.0304 漂到
0.0272)。要真正提分只能引入**新信息**。盘点现有 251 维后未被覆盖的信息:

1. **被拒交易(authorized_flag==0)只有 5 个特征**(mean/sum 金额、month_lag 均值、
   商户数、笔数)。被拒交易是风控摩擦与资金紧张的直接信号,与流失/低忠诚强相关,
   却几乎没画像。注:不做 authorized==1 子集聚合 —— 授权交易占约 90%,与全量
   聚合高度冗余,真正的增量信息全在这 10% 的被拒侧。
2. **观察期末端(month_lag=0 / -1)没有独立画像**。month_lag_pivot 只给了逐月
   count/sum 与斜率,缺末端月的金额分布、商户集中度、授权率。target 锚定观察月,
   末端行为的权重理应最高。

产出:data/processed/features_ext.parquet(card_id + 新列),由 v5_train_ext.py 合表训练。
用法:python src/v5_features_ext.py
"""
import gc
import time

import numpy as np
import pandas as pd

import elo_pipeline as ep

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def compact_agg(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """子集通用聚合:金额分布 + 时序近期性 + 多样性 + 分期/授权画像。

    只走 cython 聚合路径(不用 lambda),保证 29M 行级别可承受。
    """
    g = df.groupby("card_id", observed=True)
    out = g.agg({
        "purchase_amount": ["mean", "sum", "std", "min", "max", "median"],
        "price": ["mean", "max"],
        "month_lag": ["mean", "min", "std"],
        "month_diff": ["mean", "std"],
        "installments": ["mean", "max"],
        "hour": ["mean", "std"], "weekend": ["mean"], "category_1": ["mean"],
        "merchant_id": ["nunique"], "merchant_category_id": ["nunique"],
        "city_id": ["nunique"], "subsector_id": ["nunique"], "month_id": ["nunique"],
        "pt": ["min", "max"],
    })
    out.columns = [f"{prefix}_{c}_{s}" for c, s in out.columns]
    out[f"{prefix}_count"] = g.size()
    # 时间跨度与近期性
    out[f"{prefix}_date_ptp_days"] = (out[f"{prefix}_pt_max"] - out[f"{prefix}_pt_min"]) / 86400.0
    out[f"{prefix}_last_to_ref_days"] = (ep.REF_TS.timestamp() - out[f"{prefix}_pt_max"]) / 86400.0
    out[f"{prefix}_count_per_month"] = out[f"{prefix}_count"] / (out[f"{prefix}_month_id_nunique"] + 1e-4)
    out[f"{prefix}_sum_per_month"] = out[f"{prefix}_purchase_amount_sum"] / (out[f"{prefix}_month_id_nunique"] + 1e-4)
    # 商户集中度:笔数 / 独特商户数(复购强度)
    out[f"{prefix}_repeat_per_merchant"] = out[f"{prefix}_count"] / (out[f"{prefix}_merchant_id_nunique"] + 1e-4)
    return ep.reduce_mem_usage(out.reset_index(), verbose=False)


def main():
    log("读取 historical_transactions ...")
    hist = ep.load_transactions("historical_transactions.csv", None)
    hist = ep.clean_transactions(hist)
    hist = ep.add_time_features(hist, with_holidays=False)
    log(f"hist={hist.shape}")

    feats = []

    # ---- 1) 被拒交易完整画像(占比约 10%,是唯一未挖的行为子集)----
    dec = hist[hist["authorized_flag"] == 0]
    log(f"declined 子集: {dec.shape} ({len(dec) / len(hist):.2%})")
    feats.append(compact_agg(dec, "decx"))
    # 被拒交易的相邻间隔(连续被拒的节奏)
    s = dec[["card_id", "pt"]].sort_values(["card_id", "pt"])
    gap = s.groupby("card_id", observed=True)["pt"].diff() / 86400.0
    d = s.assign(gap=gap.astype(np.float32)).groupby("card_id", observed=True)["gap"] \
         .agg(["mean", "std", "max"])
    d.columns = [f"decx_gap_{c}" for c in d.columns]
    feats.append(ep.reduce_mem_usage(d.reset_index(), verbose=False))
    del dec, s, d
    gc.collect()

    # ---- 2) 观察期末端近窗画像(target 锚定观察月,末端权重最高)----
    for lags, tag in [([0], "m0"), ([0, -1], "m01"), ([0, -1, -2], "m012")]:
        sub = hist[hist["month_lag"].isin(lags)]
        log(f"近窗 {tag}: {sub.shape}")
        a = compact_agg(sub, tag)
        # 末端授权率单列(compact_agg 里没放 authorized_flag,因为被拒子集恒为 0)
        auth = sub.groupby("card_id", observed=True)["authorized_flag"].mean().rename(f"{tag}_auth_rate")
        a = a.merge(auth.reset_index(), on="card_id", how="left")
        feats.append(a)
        del sub, a
        gc.collect()

    # ---- 合并并派生跨子集比值(被拒/全量、末端/全期)----
    log("合并 ...")
    ext = feats[0]
    for f in feats[1:]:
        f["card_id"] = f["card_id"].astype(str)
        ext["card_id"] = ext["card_id"].astype(str)
        ext = ext.merge(f, on="card_id", how="outer")

    # 全量参照量(与主特征表口径一致,便于做比值)
    g = hist.groupby("card_id", observed=True)
    ref = pd.DataFrame({
        "all_count": g.size(),
        "all_sum": g["purchase_amount"].sum(),
        "all_mid_nunique": g["merchant_id"].nunique(),
    }).reset_index()
    ref["card_id"] = ref["card_id"].astype(str)
    ext = ext.merge(ref, on="card_id", how="left")

    eps = 1e-4
    ext["decx_count_ratio"] = ext["decx_count"] / (ext["all_count"] + eps)      # 被拒率(笔数)
    ext["decx_sum_ratio"] = ext["decx_purchase_amount_sum"] / (ext["all_sum"] + eps)
    ext["decx_merchant_ratio"] = ext["decx_merchant_id_nunique"] / (ext["all_mid_nunique"] + eps)
    for tag in ("m0", "m01", "m012"):
        ext[f"{tag}_count_ratio"] = ext[f"{tag}_count"] / (ext["all_count"] + eps)
        ext[f"{tag}_sum_ratio"] = ext[f"{tag}_purchase_amount_sum"] / (ext["all_sum"] + eps)
        ext[f"{tag}_merchant_ratio"] = ext[f"{tag}_merchant_id_nunique"] / (ext["all_mid_nunique"] + eps)
    # 末端金额均值相对全期的漂移(消费升级/降级)
    all_mean = (ext["all_sum"] / (ext["all_count"] + eps))
    for tag in ("m0", "m01", "m012"):
        ext[f"{tag}_mean_vs_all"] = ext[f"{tag}_purchase_amount_mean"] / (all_mean + eps)
    ext = ext.drop(columns=["all_count", "all_sum", "all_mid_nunique"])

    ext = ep.reduce_mem_usage(ext, verbose=False)
    path = "data/processed/features_ext.parquet"
    ext.to_parquet(path)
    log(f"保存 {path}: {ext.shape}(新增 {ext.shape[1] - 1} 列)")


if __name__ == "__main__":
    main()
