# -*- coding: utf-8 -*-
"""v24-B:ct 世代整代注入(docs/202608/04-v24_自有创新.md)。

v24-A 单模 +0.00108 为真但单成员入池被吸收(+0.00033)—— 按 v7/v11 先例,
特征族须整代注入:ct 特征矩阵重训 xgb/cat/hub,与 e_ct(lgb)组成 ct 世代。
判据(联合):数值 Δ>0.0005(追加或换代取优)+ 定性新信息✓(数字结构语义)。
用法:ELO_SEED=777 python src/v24b_ct_gen.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import v11_formula as v11
import v5_fusion as vf
from v24_cents import build_ct

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    ct = build_ct()
    base = pd.read_parquet("data/processed/features.parquet")
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    base = base.merge(ct, on="card_id", how="left")
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
    ct_cols = [c for c in ct.columns if c != "card_id"]

    def asm(side, zte, fm):
        m1 = side[["card_id"]].merge(td, on="card_id", how="left").drop(columns="card_id")
        return pd.concat([side[sel].reset_index(drop=True), pd.DataFrame(zte, columns=te_names),
                          m1.astype(np.float32), fm.reset_index(drop=True),
                          side[ct_cols].reset_index(drop=True).astype(np.float32)], axis=1)

    X, X_test = asm(train, z["tr"], fm_tr), asm(test, z["te"], fm_te)
    log(f"X={X.shape};训练 ct 世代 xgb/hub/cat")
    for name in ("xgb", "hub", "cat"):
        out = f"outputs/base_ct/{name}.npz"
        if os.path.exists(out):
            log(f"[{name}] 已存在,跳过")
            continue
        if name == "xgb":
            oof, pred, _ = ep.cv_xgboost(X, y, X_test, folds)
        elif name == "cat":
            oof, pred, _ = ep.cv_catboost(X, y, X_test, folds)
        else:
            oof, pred, _, _ = ep.cv_lightgbm(X, y, X_test, folds, ep.HUB_PARAMS, "ct_hub")
        np.savez(out, oof=oof, pred=pred)
        log(f"[ct_{name}] OOF={rmse(y, oof):.5f}")

    bases = vf.load_bases()
    CT = []
    for name, key in [("lgb", "ct_lgb"), ("xgb", "ct_xgb"), ("cat", "ct_cat"), ("hub", "ct_hub")]:
        d = np.load(f"outputs/base_ct/{name}.npz")
        bases[key] = (d["oof"], d["pred"])
        CT.append(key)
    ybin = (y < -30).astype(int).to_numpy()
    REG = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    T = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    D = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    F = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    N = sorted(k for k in bases if k.startswith("n_"))
    HEADS = ["t_clf", "t_clean", "d_clf", "d_clean", "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"]
    allf = REG + T + D + F + HEADS + N
    r0, _, _ = vf.evaluate(allf, "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
    log(f"基线(33 成员)={r0:.5f}")
    best = (-1.0, None, None)
    for tag, feats in [("追加 ct 世代", allf + CT),
                       ("f 世代换 ct 世代", REG + T + D + CT + HEADS + N)]:
        r1, _, pt = vf.evaluate(feats, "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
        d = r0 - r1
        log(f"  {tag}: {r1:.5f} → Δ={d:+.5f}")
        if d > best[0]:
            best = (d, tag, pt)
    d, tag, pt = best
    log(f"判据[v24-B ct 世代]:最优[{tag}] Δ={d:+.5f} "
        f"{'✅ 通过(定性:新信息✓)' if d > 0.0005 else '❌ 不足'}")
    if d > 0.0005:
        sub = pd.read_csv(os.path.join(ep.CONFIG["DATA_DIR"], "sample_submission.csv"))
        sub["target"] = pt
        sub.to_csv("outputs/submission_v24_ct.csv", index=False)
        log("已保存 outputs/submission_v24_ct.csv")
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
