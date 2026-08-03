# -*- coding: utf-8 -*-
"""v20:树基模型多 seed 平均(思路扫荡 1 号,docs/202608/03-v18_思路扫荡.md)。

结构性遗漏:NN 族全部 5-seed 平均,而融合池 16 个树基模型全是单 seed 777。
本脚本对最强的 fm 套(lgb/xgb/cat/hub/clf/clean)补 seed 778/779,三 seed 平均
→ outputs/base_fma/。折划分严格保持 777(只动模型内部随机性,OOF 协议不变)。
判据:v5_fusion F38/F39(f_* 替换为 a_*)vs F31 3.620625,ΔOOF>0.0005。
用法:ELO_SEED=777 python src/v20_seedavg.py run <778|779>
     ELO_SEED=777 python src/v20_seedavg.py merge
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import v11_formula as v11

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def patch_seed(sd):
    ep.LGB_PARAMS = {**ep.LGB_PARAMS, "seed": sd, "bagging_seed": sd + 1,
                     "feature_fraction_seed": sd + 2}
    ep.HUB_PARAMS = {**ep.LGB_PARAMS, "objective": "huber", "alpha": 1.35}
    ep.XGB_PARAMS = {**ep.XGB_PARAMS, "random_state": sd}
    ep.CAT_PARAMS = {**ep.CAT_PARAMS, "random_seed": sd}
    ep.CLF_PARAMS = {**ep.CLF_PARAMS, "seed": sd, "bagging_seed": sd + 1,
                     "feature_fraction_seed": sd + 2}
    log(f"params seed → {sd}(折协议仍 777)")


def assemble_xy():
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

    return asm(train, z["tr"], fm_tr), asm(test, z["te"], fm_te), y, folds


def run(sd):
    patch_seed(sd)
    X, X_test, y, folds = assemble_xy()
    out = f"outputs/base_fm_s{sd}"
    os.makedirs(out, exist_ok=True)
    log(f"X={X.shape} → {out}")

    def dump(name, oof, pred):
        np.savez(os.path.join(out, f"{name}.npz"), oof=oof, pred=pred)
        log(f"[s{sd}] {name:6s} OOF={rmse(y, oof):.5f}")

    oof, pred, _, _ = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, f"lgb-s{sd}")
    dump("lgb", oof, pred)
    oof, pred, _ = ep.cv_xgboost(X, y, X_test, folds);            dump("xgb", oof, pred)
    oof, pred, _ = ep.cv_catboost(X, y, X_test, folds);           dump("cat", oof, pred)
    oof, pred, _, _ = ep.cv_lightgbm(X, y, X_test, folds, ep.HUB_PARAMS, f"hub-s{sd}")
    dump("hub", oof, pred)
    oof, pred, auc = ep.cv_outlier_clf(X, y, X_test, folds)
    np.savez(os.path.join(out, "clf.npz"), oof=oof, pred=pred)
    log(f"[s{sd}] clf AUC={auc:.5f}")
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
    dump("clean", oc, pc)


def merge():
    base = pd.read_parquet("data/processed/features.parquet")
    y = base[base["is_train"] == 1]["target"].to_numpy()
    os.makedirs("outputs/base_fma", exist_ok=True)
    dirs = ["outputs/base_fm", "outputs/base_fm_s778", "outputs/base_fm_s779"]
    for name in ("lgb", "xgb", "cat", "hub", "clf", "clean"):
        oofs, preds = [], []
        for d in dirs:
            p = os.path.join(d, f"{name}.npz")
            if os.path.exists(p):
                zz = np.load(p)
                oofs.append(zz["oof"])
                preds.append(zz["pred"])
        np.savez(f"outputs/base_fma/{name}.npz",
                 oof=np.mean(oofs, 0), pred=np.mean(preds, 0))
        if name not in ("clf",):
            log(f"[fma] {name:6s} {len(oofs)}seed 平均 OOF={rmse(y, np.mean(oofs, 0)):.5f}"
                f"(777 单 seed {rmse(y, oofs[0]):.5f})")


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    mode = sys.argv[1]
    if mode == "run":
        run(int(sys.argv[2]))
    elif mode == "merge":
        merge()


if __name__ == "__main__":
    main()
