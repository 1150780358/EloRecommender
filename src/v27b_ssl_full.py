# -*- coding: utf-8 -*-
"""v27-B:SSL 编码器 + 全量工程特征微调(docs/202608/04-v27_自监督预训练clf.md)。

v27 第一迭代 AUC 0.828:裸序列 vs 334 列特征的不公平对比。
正确问题:预训练编码器在全量特征之上的**增量**。静态分支 30 列 → 全量 fm 特征矩阵
(标准化,NaN→0),复用已存 ssl_encoder.pt,逐折微调,同一判据链。
用法:ELO_SEED=777 python src/v27b_ssl_full.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

import elo_pipeline as ep
import v11_formula as v11
import v5_fusion as vf

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()
L = 128


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def build_full_static():
    base = pd.read_parquet("data/processed/features.parquet")
    base = base.merge(v11.hist_lag_amt(), on="card_id", how="left")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
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

    Xtr, Xte = asm(train, z["tr"], fm_tr), asm(test, z["te"], fm_te)
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    Xtr = ((Xtr - mu) / sd).fillna(0).clip(-5, 5).to_numpy(np.float32)
    Xte = ((Xte - mu) / sd).fillna(0).clip(-5, 5).to_numpy(np.float32)
    return Xtr, Xte, y


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    import torch
    import torch.nn as nn
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    z = np.load("outputs/tx_tensor.npz")
    n_mcat, n_ssec = int(z["vocab"][0]), int(z["vocab"][1])
    num_all = np.concatenate([z["num_tr"], z["num_te"]])
    cat_all = np.concatenate([z["cat_tr"], z["cat_te"]])
    n_tr = len(z["num_tr"])
    Xtr, Xte, y = build_full_static()
    folds = ep.make_folds(y)
    yb = (y < -30).astype(int).to_numpy()
    log(f"全量静态 {Xtr.shape};张量 {num_all.shape} dev={dev}")

    NUM = torch.from_numpy(num_all).to(dev)
    CAT = torch.from_numpy(cat_all.astype(np.int32)).to(dev)
    ST = torch.from_numpy(np.concatenate([Xtr, Xte])).to(dev)
    D = 96

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.em = nn.Embedding(n_mcat + 1, 16, padding_idx=0)
            self.es = nn.Embedding(n_ssec + 1, 8, padding_idx=0)
            self.proj = nn.Linear(8 + 24, D)
            self.pos = nn.Embedding(L, D)
            layer = nn.TransformerEncoderLayer(D, 4, 192, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 4)

        def forward(self, xn, xc, msk_flag):
            e = torch.cat([xn, msk_flag.unsqueeze(2),
                           self.em(xc[:, :, 0].long()), self.es(xc[:, :, 1].long())], 2)
            h = self.proj(e) + self.pos.weight.unsqueeze(0)
            pad = xn[:, :, 6] > 0.5
            return self.enc(h, src_key_padding_mask=pad), pad

    class ClfNet(nn.Module):
        def __init__(self, enc, st_dim):
            super().__init__()
            self.encoder = enc
            self.stat = nn.Sequential(nn.Linear(st_dim, 128), nn.ReLU(), nn.Dropout(0.2))
            self.head = nn.Sequential(
                nn.Linear(D + 128, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xn, xc, st):
            h, pad = self.encoder(xn, xc, torch.zeros_like(xn[:, :, 0]))
            w = (~pad).float().unsqueeze(2)
            pool = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
            return self.head(torch.cat([pool, self.stat(st)], 1)).squeeze(1)

    pre_state = torch.load("outputs/nn_parts/ssl_encoder.pt", map_location=dev)
    YB = torch.from_numpy(yb.astype(np.float32)).to(dev)
    pw = torch.tensor((1 - yb.mean()) / yb.mean(), device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def infer(model, idx, bs=2048):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                outs.append(torch.sigmoid(model(NUM[b].float(), CAT[b], ST[b].float())))
        return torch.cat(outs).float().cpu().numpy()

    N = len(NUM)
    te_idx = torch.arange(n_tr, N, device=dev)
    oof = np.zeros(n_tr)
    pred = np.zeros(N - n_tr)
    FBS, MAX_EP, PAT = 512, 12, 3
    for k, (tr, va) in enumerate(folds):
        torch.manual_seed(777 + k)
        enc = Encoder().to(dev)
        enc.load_state_dict(pre_state)
        model = ClfNet(enc, Xtr.shape[1]).to(dev)
        opt = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": 1e-4},
            {"params": list(model.stat.parameters()) + list(model.head.parameters()), "lr": 1e-3},
        ], weight_decay=1e-5)
        tr_t = torch.from_numpy(tr).to(dev)
        va_t = torch.from_numpy(va).to(dev)
        best, wait, best_state = -1.0, 0, None
        for epn in range(MAX_EP):
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), device=dev)]
            for i in range(0, len(perm), FBS):
                b = perm[i:i + FBS]
                opt.zero_grad()
                loss = bce(model(NUM[b].float(), CAT[b], ST[b].float()), YB[b])
                loss.backward()
                opt.step()
            a = roc_auc_score(yb[va], infer(model, va_t))
            if a > best + 1e-5:
                best, wait = a, 0
                best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PAT:
                    break
        model.load_state_dict(best_state)
        oof[va] = infer(model, va_t)
        pred += infer(model, te_idx) / len(folds)
        log(f"  [ftF] fold{k + 1}: AUC={best:.5f}")
    np.savez("outputs/base_nn_clf/ssl_full_clf.npz", oof=oof, pred=pred)
    auc_nn = roc_auc_score(yb, oof)
    log(f"[sslF] OOF AUC={auc_nn:.5f}(f_clf 0.90586 / 冠军 0.914)")

    bases = vf.load_bases()
    lgb_oof, lgb_pred = bases["f_clf"]
    def rk(a):
        return pd.Series(a).rank(pct=True).to_numpy()
    best_w, best_auc = 0.0, roc_auc_score(yb, rk(lgb_oof))
    for w in np.arange(0.05, 1.0, 0.05):
        a = roc_auc_score(yb, (1 - w) * rk(lgb_oof) + w * rk(oof))
        if a > best_auc:
            best_w, best_auc = w, a
    log(f"rank 混合:w_nn={best_w:.2f} AUC={best_auc:.5f}")
    bl_oof = (1 - best_w) * rk(lgb_oof) + best_w * rk(oof)
    bl_pred = (1 - best_w) * rk(lgb_pred) + best_w * rk(pred)

    REG = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    T = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    D_ = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    F = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    NN = sorted(k for k in bases if k.startswith("n_"))
    allf = (REG + T + D_ + F + ["t_clf", "t_clean", "d_clf", "d_clean",
            "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"] + NN)
    r0, _, _ = vf.evaluate(allf, "bayes", bases, y, yb, folds, p_src="f_clf", clean_src="f_clean")
    results = {}
    bases["p_ssl"] = (oof, pred)
    r1, _, _ = vf.evaluate(allf + ["p_ssl"], "bayes", bases, y, yb, folds, p_src="f_clf", clean_src="f_clean")
    results["追加 p_sslF 成员"] = r0 - r1
    del bases["p_ssl"]
    bases["f_clf"] = (bl_oof, bl_pred)
    r2, _, _ = vf.evaluate(allf, "bayes", bases, y, yb, folds, p_src="f_clf", clean_src="f_clean")
    results["f_clf 替换为混合"] = r0 - r2
    for tag, d in results.items():
        log(f"  {tag}: Δ={d:+.5f}")
    best_d = max(results.values())
    log(f"判据[v27-B SSL+全量特征]:基线={r0:.5f} 最优 Δ={best_d:+.5f} "
        f"{'✅ 通过' if best_d > 0.0005 else '❌ 不足'}")
    if best_d <= 0.0005:
        sys.exit(3)


if __name__ == "__main__":
    main()
