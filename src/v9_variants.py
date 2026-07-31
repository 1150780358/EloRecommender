# -*- coding: utf-8 -*-
"""v9:剩余候选批量验证(16th 稳健 clean / 5th 分段建模),777 规范折协议。

E1 clean2(16th nlgn #82166,一手):现役 clean 模型用**真实** outlier 标签剔除训练行,
    test 侧无法复刻该口径 → CV 高估 clean 质量。16th 的修法:用分类器 OOF 概率的
    q90 分位切出"预测正常卡"训练(train/test 同口径),让 CV 如实继承分类器误差。
E2 seg(5th,#82314 bangda 转述,二手):折内 isotonic 校准概率,阈值 0.015 把卡切成
    高/低风险两段分别建模,按段拼接预测 —— 引入"分段"自由度,概率本身已在池中。

判据:融合层 vs F23(777 复跑值,前置链产出)Δ>0.0005 才考虑上线。
用法:ELO_SEED=777 python src/v9_variants.py(须在 base_td 777 复跑完成后运行)
"""
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import v5_fusion as vf

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
base = pd.read_parquet("data/processed/features.parquet")
train = base[base["is_train"] == 1].reset_index(drop=True)
test = base[base["is_train"] == 0].reset_index(drop=True)
y = train["target"]
ybin = (y < -30).astype(int).to_numpy()
folds = ep.make_folds(y)
imp = pd.read_csv("outputs/feature_importance.csv")
sel = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
z = np.load("outputs/te_features.npz", allow_pickle=True)
te_names = [str(x) for x in z["names"]]
td = pd.read_parquet("outputs/td_features.parquet")
td_tr = train[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
td_te = test[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
X = pd.concat([train[sel].reset_index(drop=True), pd.DataFrame(z["tr"], columns=te_names),
               td_tr.astype(np.float32)], axis=1)
X_test = pd.concat([test[sel].reset_index(drop=True), pd.DataFrame(z["te"], columns=te_names),
                    td_te.astype(np.float32)], axis=1)
zc = np.load("outputs/base_td/clf.npz")
p_oof, p_test = zc["oof"], zc["pred"]
log(f"X={X.shape},clf OOF 概率就绪(777 折)")

# ---- E1:16th 稳健 clean(clean2)----
q = float(np.quantile(p_oof, 0.90))
keep = p_oof <= q
log(f"[E1] q90={q:.4f},预测正常卡 {keep.sum()}/{len(keep)}"
    f"(真实 outlier 在保留集中占 {ybin[keep].mean():.4%},全量先验 {ybin.mean():.4%})")
oof1, pred1 = np.zeros(len(X)), np.zeros(len(X_test))
for k, (tr, va) in enumerate(folds):
    tr_k = tr[keep[tr]]
    m = lgb.train(ep.LGB_PARAMS, lgb.Dataset(X.iloc[tr_k], y.iloc[tr_k]), 10000,
                  valid_sets=[lgb.Dataset(X.iloc[va[keep[va]]], y.iloc[va[keep[va]]])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    oof1[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    pred1 += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
    log(f"[E1] fold{k + 1} iter={m.best_iteration}")
np.savez("outputs/base_td/clean2.npz", oof=oof1, pred=pred1)
log(f"[E1] clean2 OOF={rmse(y, oof1):.5f}(对照 d_clean 口径不同,以融合层为准)")

# ---- E2:5th 分段建模(seg)----
THR = 0.015
oof2, pred2 = np.zeros(len(X)), np.zeros(len(X_test))
for k, (tr, va) in enumerate(folds):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_oof[tr], ybin[tr])
    p_tr, p_va, p_te = iso.predict(p_oof[tr]), iso.predict(p_oof[va]), iso.predict(p_test)
    hi_tr, lo_tr = tr[p_tr >= THR], tr[p_tr < THR]
    if k == 0:
        log(f"[E2] fold1 段规模:hi={len(hi_tr)} lo={len(lo_tr)}"
            f"(hi 内真实 outlier 率 {ybin[hi_tr].mean():.2%})")
    pv, pt = np.zeros(len(va)), np.zeros(len(X_test))
    for seg_tr, mask_va, mask_te in ((lo_tr, p_va < THR, p_te < THR),
                                     (hi_tr, p_va >= THR, p_te >= THR)):
        m = lgb.train(ep.LGB_PARAMS, lgb.Dataset(X.iloc[seg_tr], y.iloc[seg_tr]), 10000,
                      valid_sets=[lgb.Dataset(X.iloc[va[mask_va]], y.iloc[va[mask_va]])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        pv[mask_va] = m.predict(X.iloc[va[mask_va]], num_iteration=m.best_iteration)
        pt[mask_te] = m.predict(X_test[mask_te], num_iteration=m.best_iteration)
    oof2[va] = pv
    pred2 += pt / len(folds)
    log(f"[E2] fold{k + 1} 完成")
np.savez("outputs/base_td/seg.npz", oof=oof2, pred=pred2)
log(f"[E2] seg OOF={rmse(y, oof2):.5f}")

# ---- 融合层判据(777 全池)----
bases = vf.load_bases()
REG = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
TREG = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
DREG = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
allf = REG + TREG + DREG + ["t_clf", "t_clean", "d_clf", "d_clean", "p_cal", "ev", "p_cal_x_clean"]
r0, _, _ = vf.evaluate(allf, "bayes", bases, y, ybin, folds, p_src="d_clf", clean_src="d_clean")
log(f"[判据] F23(777 基线)          OOF={r0:.5f}")
plans = [
    ("F24 +clean2",        allf + ["d_clean2"], dict(p_src="d_clf", clean_src="d_clean")),
    ("F25 clean源换clean2", allf,                dict(p_src="d_clf", clean_src="d_clean2")),
    ("F26 +seg",           allf + ["d_seg"],    dict(p_src="d_clf", clean_src="d_clean")),
    ("F27 +clean2+seg",    allf + ["d_clean2", "d_seg"], dict(p_src="d_clf", clean_src="d_clean")),
]
best = (None, r0, None)
for tag, feats, kw in plans:
    r, _, pt = vf.evaluate(feats, "bayes", bases, y, ybin, folds, **kw)
    v = "✅" if r0 - r > 0.0005 else "❌ 不足"
    log(f"[判据] {tag:22s} OOF={r:.5f} → Δ {r0 - r:+.5f} {v}")
    if r < best[1]:
        best = (tag, r, pt)
if best[0]:
    pd.DataFrame({"card_id": test["card_id"], "target": best[2]}).to_csv(
        "outputs/submission_v9.csv", index=False)
    log(f"[判据] 最优 {best[0]} OOF={best[1]:.5f} → outputs/submission_v9.csv(超阈值方可上线)")
else:
    log("[判据] 无方案超过 F23 基线,v9 证伪")
