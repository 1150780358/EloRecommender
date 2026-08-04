# -*- coding: utf-8 -*-
"""v22-B:32nd 小特征子集成员(docs/202608/04-v22_低先验备选出清.md)。

32nd diversity 配方:每模型仅 20-60 列的小子集成员。
4 个 40 列 LGB:s1/s2/s3 = sel gain 排名 1-40/41-80/81-120 段,s4 = 全列随机 40(seed 777)。
判据:同池对照(F31 33 成员 ± e_s*,bayes),整组及逐一,ΔOOF>0.0005。
用法:ELO_SEED=777 python src/v22_subsets.py
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
    log(f"X={X.shape}")
    rng = np.random.RandomState(777)
    subsets = {"e_s1": sel[:40], "e_s2": sel[40:80], "e_s3": sel[80:120],
               "e_s4": list(rng.choice(X.columns, 40, replace=False))}

    bases = vf.load_bases()
    os.makedirs("outputs/base_ss", exist_ok=True)
    for name, cols in subsets.items():
        oof, pred, _, _ = ep.cv_lightgbm(X[cols], y, X_test[cols], folds, ep.LGB_PARAMS, name)
        np.savez(f"outputs/base_ss/{name}.npz", oof=oof, pred=pred)
        f_lgb = np.load("outputs/base_fm/lgb.npz")["oof"]
        log(f"[{name}] 单模 OOF={rmse(y, oof):.5f}(与 f_lgb 相关 {np.corrcoef(oof, f_lgb)[0, 1]:.4f})")
        bases[name] = (oof, pred)

    ybin = (y < -30).astype(int).to_numpy()
    REG = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    T = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    D = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    F = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    N = sorted(k for k in bases if k.startswith("n_"))
    allf = (REG + T + D + F + ["t_clf", "t_clean", "d_clf", "d_clean",
            "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"] + N)
    r0, _, _ = vf.evaluate(allf, "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
    log(f"基线(33 成员)={r0:.5f}")
    best_d, best_pt = -1.0, None
    for add in [["e_s1"], ["e_s2"], ["e_s3"], ["e_s4"], list(subsets)]:
        r1, _, pt = vf.evaluate(allf + add, "bayes", bases, y, ybin, folds,
                                p_src="f_clf", clean_src="f_clean")
        d = r0 - r1
        log(f"  +{'+'.join(add)}: {r1:.5f} → Δ={d:+.5f}")
        if d > best_d:
            best_d, best_pt = d, pt
    log(f"判据[v22-B subsets]:最优 Δ={best_d:+.5f} {'✅ 通过' if best_d > 0.0005 else '❌ 不足'}")
    if best_d > 0.0005:
        sub = pd.read_csv(os.path.join(ep.CONFIG["DATA_DIR"], "sample_submission.csv"))
        sub["target"] = best_pt
        sub.to_csv("outputs/submission_v22_ss.csv", index=False)
        log("已保存 outputs/submission_v22_ss.csv")
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
