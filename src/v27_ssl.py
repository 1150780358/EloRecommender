# -*- coding: utf-8 -*-
"""v27:自监督预训练 clf(docs/202608/04-v27_自监督预训练clf.md)。

冲第一的最后一发:2019 年不存在的方法。
Stage1 masked 预训练:325k 卡(train+test,无标签)近 128 笔交易,Transformer(4L,d96),
  随机遮 15% 真实交易 → 预测其 merchant_category(CE)+ 金额通道(MSE)。
Stage2 微调:同一权重逐折(777 十折)接二分类头,pos_weight 平衡,早停看 val AUC。
判据链:①OOF AUC vs f_clf 0.90586;②rank 混合 f_clf 的 AUC;③融合层替换/追加 Δ>0.0005。
用法:ELO_SEED=777 python src/v27_ssl.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

import elo_pipeline as ep
import v5_fusion as vf

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()
L = 128


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    import torch
    import torch.nn as nn
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    z = np.load("outputs/tx_tensor.npz")
    n_mcat, n_ssec = int(z["vocab"][0]), int(z["vocab"][1])
    MC_MASK, SS_MASK = n_mcat, n_ssec                     # 新增 mask token id
    num_all = np.concatenate([z["num_tr"], z["num_te"]])   # [N,L,7] fp16(ch6=is_pad)
    cat_all = np.concatenate([z["cat_tr"], z["cat_te"]])   # [N,L,2] int16
    st_tr, st_te = z["st_tr"], z["st_te"]
    n_tr = len(z["num_tr"])
    base = pd.read_parquet("data/processed/features.parquet")
    y = base[base["is_train"] == 1].reset_index(drop=True)["target"]
    folds = ep.make_folds(y)
    yb = (y < -30).astype(int).to_numpy()
    log(f"张量 {num_all.shape} vocab=({n_mcat},{n_ssec}) dev={dev}")

    NUM = torch.from_numpy(num_all).to(dev)                # fp16
    CAT = torch.from_numpy(cat_all.astype(np.int32)).to(dev)
    ST = torch.from_numpy(np.concatenate([st_tr, st_te])).to(dev)
    D = 96

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.em = nn.Embedding(n_mcat + 1, 16, padding_idx=0)
            self.es = nn.Embedding(n_ssec + 1, 8, padding_idx=0)
            self.proj = nn.Linear(8 + 24, D)               # 7 数值 + is_masked + 24 embed
            self.pos = nn.Embedding(L, D)
            layer = nn.TransformerEncoderLayer(D, 4, 192, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 4)

        def forward(self, xn, xc, msk_flag):
            e = torch.cat([xn, msk_flag.unsqueeze(2),
                           self.em(xc[:, :, 0].long()), self.es(xc[:, :, 1].long())], 2)
            h = self.proj(e) + self.pos.weight.unsqueeze(0)
            pad = xn[:, :, 6] > 0.5
            return self.enc(h, src_key_padding_mask=pad), pad

    torch.manual_seed(777)
    np.random.seed(777)
    encoder = Encoder().to(dev)
    head_mc = nn.Linear(D, n_mcat).to(dev)
    head_amt = nn.Linear(D, 1).to(dev)

    # ---------- Stage1 masked 预训练 ----------
    PRE_EP, BS = 3, 256
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head_mc.parameters())
                            + list(head_amt.parameters()), lr=1e-3, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()
    N = len(NUM)
    gen = torch.Generator(device=dev).manual_seed(777)
    for epn in range(PRE_EP):
        perm = torch.randperm(N, device=dev)
        tot, tot_acc, tot_n = 0.0, 0.0, 0
        for i in range(0, N, BS):
            b = perm[i:i + BS]
            xn = NUM[b].float()
            xc = CAT[b].clone()
            real = xn[:, :, 6] < 0.5
            mrand = torch.rand(real.shape, device=dev, generator=gen) < 0.15
            msk = real & mrand
            mc_true = xc[:, :, 0][msk].long()
            amt_true = xn[:, :, 0][msk]
            xn2 = xn.clone()
            xn2[:, :, :6][msk] = 0.0
            xc[:, :, 0][msk] = MC_MASK
            xc[:, :, 1][msk] = SS_MASK
            opt.zero_grad()
            h, _ = encoder(xn2, xc, msk.float())
            hm = h[msk]
            loss = ce(head_mc(hm), mc_true) + 5.0 * torch.mean((head_amt(hm).squeeze(1) - amt_true) ** 2)
            loss.backward()
            opt.step()
            with torch.no_grad():
                tot += float(loss) * len(hm)
                tot_acc += float((head_mc(hm).argmax(1) == mc_true).float().sum())
                tot_n += len(hm)
        log(f"[pre] epoch{epn + 1}: loss={tot / tot_n:.4f} masked-mcat-acc={tot_acc / tot_n:.4f}")
    torch.save(encoder.state_dict(), "outputs/nn_parts/ssl_encoder.pt")
    pre_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}

    # ---------- Stage2 逐折微调二分类 ----------
    class ClfNet(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.encoder = enc
            self.stat = nn.Linear(st_tr.shape[1], 32)
            self.head = nn.Sequential(
                nn.Linear(D + 32, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xn, xc, st):
            h, pad = self.encoder(xn, xc, torch.zeros_like(xn[:, :, 0]))
            w = (~pad).float().unsqueeze(2)
            pool = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
            return self.head(torch.cat([pool, torch.relu(self.stat(st))], 1)).squeeze(1)

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

    te_idx = torch.arange(n_tr, N, device=dev)
    oof = np.zeros(n_tr)
    pred = np.zeros(N - n_tr)
    FBS, MAX_EP, PAT = 512, 12, 3
    for k, (tr, va) in enumerate(folds):
        torch.manual_seed(777 + k)
        encoder.load_state_dict(pre_state)
        model = ClfNet(encoder).to(dev)
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
        log(f"  [ft] fold{k + 1}: AUC={best:.5f}")
    np.savez("outputs/base_nn_clf/ssl_clf.npz", oof=oof, pred=pred)
    auc_nn = roc_auc_score(yb, oof)
    log(f"[ssl] OOF AUC={auc_nn:.5f}(f_clf 0.90586 / v15 NN ens 0.9078 / 冠军 0.914)")

    # ---------- 判据链:rank 混合 + 融合 ----------
    bases = vf.load_bases()
    lgb_oof, lgb_pred = bases["f_clf"]
    def rk(a):
        return pd.Series(a).rank(pct=True).to_numpy()
    best_w, best_auc = 0.0, roc_auc_score(yb, rk(lgb_oof))
    for w in np.arange(0.05, 1.0, 0.05):
        a = roc_auc_score(yb, (1 - w) * rk(lgb_oof) + w * rk(oof))
        if a > best_auc:
            best_w, best_auc = w, a
    log(f"rank 混合:w_nn={best_w:.2f} AUC={best_auc:.5f}(纯 lgb rank {roc_auc_score(yb, rk(lgb_oof)):.5f})")
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
    results["追加 p_ssl 成员"] = r0 - r1
    del bases["p_ssl"]
    bases["f_clf"] = (bl_oof, bl_pred)
    r2, _, _ = vf.evaluate(allf, "bayes", bases, y, yb, folds, p_src="f_clf", clean_src="f_clean")
    results["f_clf 替换为混合"] = r0 - r2
    for tag, d in results.items():
        log(f"  {tag}: Δ={d:+.5f}")
    best_d = max(results.values())
    log(f"判据[v27 SSL-clf]:基线={r0:.5f} 最优 Δ={best_d:+.5f} "
        f"{'✅ 通过' if best_d > 0.0005 else '❌ 不足'}")
    if best_d <= 0.0005:
        sys.exit(3)


if __name__ == "__main__":
    main()
