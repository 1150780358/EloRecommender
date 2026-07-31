# -*- coding: utf-8 -*-
"""v8-P3:rm 行级元特征的最后一次测试 —— 只进二层融合,不回灌一层 GBDT。

v8-v1/v2 证明:rm(目标的 OOF 预测聚合)回灌一层 GBDT 使 OOF 变差(一层深树
过度信任目标相关元特征,携带折间标签串扰噪声)。本测试把 rm 14 列作为二层
BayesianRidge 的附加元特征(cond 机制),线性低容量 + 折内 fit,无该通道。
判据:vs F23 全池 OOF 3.62711,改善 >0.0005 才考虑上线。
用法:python src/v8_fusion_test.py
"""
import numpy as np
import pandas as pd

import elo_pipeline as ep
import v5_fusion as vf

rm = pd.read_parquet("outputs/rowmeta_features.parquet")
bases = vf.load_bases()
base_tbl = pd.read_parquet("data/processed/features.parquet")
train = base_tbl[base_tbl["is_train"] == 1].reset_index(drop=True)
test = base_tbl[base_tbl["is_train"] == 0].reset_index(drop=True)
y = train["target"]
ybin = (y < -30).astype(int).to_numpy()
folds = ep.make_folds(y)

REG = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
TREG = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
DREG = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
allf = REG + TREG + DREG + ["t_clf", "t_clean", "d_clf", "d_clean", "p_cal", "ev", "p_cal_x_clean"]

rm_cols = [c for c in rm.columns if c != "card_id"]
Ctr = train[["card_id"]].merge(rm, on="card_id", how="left")[rm_cols].to_numpy(np.float32)
Cte = test[["card_id"]].merge(rm, on="card_id", how="left")[rm_cols].to_numpy(np.float32)
Ctr = np.nan_to_num(Ctr, nan=-1.0)
Cte = np.nan_to_num(Cte, nan=-1.0)

r0, _, _ = vf.evaluate(allf, "bayes", bases, y, ybin, folds, p_src="d_clf", clean_src="d_clean")
print(f"[v8-P3] F23 复算(无 rm)      OOF={r0:.5f}", flush=True)
r1, _, pt = vf.evaluate(allf, "bayes", bases, y, ybin, folds, cond=(Ctr, Cte),
                        p_src="d_clf", clean_src="d_clean")
print(f"[v8-P3] F23 + rm14(二层cond) OOF={r1:.5f} → Δ {r0 - r1:+.5f} "
      f"{'✅' if r0 - r1 > 0.0005 else '❌ 不足'}", flush=True)
sub = [c for c in ("rm_or_mean", "rm_or_max", "rm_tm_min", "rm_tm_mean") if c in rm_cols]
idx = [rm_cols.index(c) for c in sub]
r2, _, _ = vf.evaluate(allf, "bayes", bases, y, ybin, folds, cond=(Ctr[:, idx], Cte[:, idx]),
                       p_src="d_clf", clean_src="d_clean")
print(f"[v8-P3] F23 + rm4 精选       OOF={r2:.5f} → Δ {r0 - r2:+.5f} "
      f"{'✅' if r0 - r2 > 0.0005 else '❌ 不足'}", flush=True)
