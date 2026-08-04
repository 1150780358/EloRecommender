# -*- coding: utf-8 -*-
"""v29:tf-idf 稀疏 Ridge 成员(55th 配方;docs/202608/04-v29_知乎与55th出清.md)。

55th:"tf-idf 直接当特征无用(=我们 v21-B 证伪),但其上加 Ridge stacking 层,
公私榜均 +0.002"。池内 34 成员无一见过原始稀疏共现矩阵 —— 线性×高维稀疏是全新载体。
做法:hist(authorized=1)card×merchant_id / card×mcat 计数 → tf-idf → hstack →
Ridge(sparse_cg)十折 OOF → e_ti 成员入池 bayes 判据 Δ>0.0005。
用法:ELO_SEED=777 python src/v29_tfidf_ridge.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import v5_fusion as vf

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_tfidf(cards: pd.Index):
    hist = ep.clean_transactions(ep.load_transactions("historical_transactions.csv"))
    hist = hist[hist["authorized_flag"] == 1][["card_id", "merchant_id", "merchant_category_id"]]
    hist["card_id"] = hist["card_id"].astype(str)
    ci = pd.Series(np.arange(len(cards)), index=cards)
    hist = hist[hist["card_id"].isin(ci.index)]
    row = hist["card_id"].map(ci).to_numpy(np.int32)
    mats = []
    for col, min_cnt in (("merchant_id", 3), ("merchant_category_id", 1)):
        codes, vals = pd.factorize(hist[col])
        vc = np.bincount(codes[codes >= 0], minlength=len(vals))
        keep = vc >= min_cnt
        remap = np.full(len(vals), -1, np.int32)
        remap[keep] = np.arange(int(keep.sum()), dtype=np.int32)
        c2 = np.where(codes >= 0, remap[np.clip(codes, 0, None)], -1)
        ok = c2 >= 0
        m = sp.csr_matrix((np.ones(int(ok.sum()), np.float32), (row[ok], c2[ok])),
                          shape=(len(cards), int(keep.sum())))
        m = TfidfTransformer(sublinear_tf=True).fit_transform(m)
        mats.append(m)
        log(f"[{col}] vocab={int(keep.sum())} nnz={m.nnz:,}")
    return sp.hstack(mats).tocsr()


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    base = pd.read_parquet("data/processed/features.parquet")
    cards = pd.Index(base["card_id"].astype(str))
    M = build_tfidf(cards)
    is_tr = (base["is_train"] == 1).to_numpy()
    train = base[base["is_train"] == 1].reset_index(drop=True)
    y = train["target"]
    folds = ep.make_folds(y)
    Mtr, Mte = M[is_tr], M[~is_tr]
    log(f"tf-idf 矩阵 {M.shape}")

    yv = y.to_numpy()
    oof = np.zeros(len(train))
    pred = np.zeros(int((~is_tr).sum()))
    for k, (tr, va) in enumerate(folds):
        r = Ridge(alpha=10.0, solver="sparse_cg", tol=1e-4)
        r.fit(Mtr[tr], yv[tr])
        oof[va] = r.predict(Mtr[va])
        pred += r.predict(Mte) / len(folds)
        log(f"  [ti] fold{k + 1}: rmse={rmse(yv[va], oof[va]):.5f}")
    os.makedirs("outputs/base_ti", exist_ok=True)
    np.savez("outputs/base_ti/ridge.npz", oof=oof, pred=pred)
    f_lgb = np.load("outputs/base_fm/lgb.npz")["oof"]
    log(f"[ti] 单模 OOF={rmse(yv, oof):.5f} 与 f_lgb 相关 {np.corrcoef(oof, f_lgb)[0, 1]:.4f}")

    bases = vf.load_bases()
    bases["e_ti"] = (oof, pred)
    ybin = (y < -30).astype(int).to_numpy()
    REG = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    T = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    D = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    F = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    N = sorted(k for k in bases if k.startswith("n_"))
    allf = (REG + T + D + F + ["t_clf", "t_clean", "d_clf", "d_clean",
            "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"] + N)
    r0, _, _ = vf.evaluate(allf, "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
    r1, _, pt = vf.evaluate(allf + ["e_ti"], "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
    d = r0 - r1
    log(f"判据[v29 tfidf-ridge]:基线={r0:.5f} +ti={r1:.5f} → Δ={d:+.5f} "
        f"{'✅ 通过(定性:新载体✓)' if d > 0.0005 else '❌ 不足'}")
    if d > 0.0005:
        sub = pd.read_csv(os.path.join(ep.CONFIG["DATA_DIR"], "sample_submission.csv"))
        sub["target"] = pt
        sub.to_csv("outputs/submission_v29_ti.csv", index=False)
        log("已保存 outputs/submission_v29_ti.csv")
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
