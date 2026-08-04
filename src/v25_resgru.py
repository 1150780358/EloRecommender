# -*- coding: utf-8 -*-
"""v25:残差异构学习(GRU 学融合残差;docs/202608/04-v25_残差异构与差距解剖.md)。

自有创新清单最后一项:v13 GRU 骨架,目标改为 r = y − fusion_oof(bayes F33 池),
异构载体在残差空间找树系融合漏掉的序列结构。
先验对抗:v8 定律(OOF 派生量回灌一层高容量模型=结构性错误)预测失败,本实验兼作其对照。
判据:同池对照(allf ± e_rg,bayes),Δ>0.0005;若通过须警惕 OOF 串扰乐观偏置。
用法:ELO_SEED=777 python src/v25_resgru.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep
import v5_fusion as vf

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def main():
    assert os.environ.get("ELO_SEED") == "777", "必须 ELO_SEED=777 运行(折协议纪律)"
    import torch
    import torch.nn as nn
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    z = np.load("outputs/seq_tensor.npz")
    xs_tr, st_tr = z["xs_tr"], z["st_tr"]
    xs_te, st_te = z["xs_te"], z["st_te"]
    base = pd.read_parquet("data/processed/features.parquet")
    y = base[base["is_train"] == 1].reset_index(drop=True)["target"]
    folds = ep.make_folds(y)
    yv = y.to_numpy(np.float32)
    ybin = (y < -30).astype(int).to_numpy()

    bases = vf.load_bases()
    REG = [k for k in ("lgb", "xgb", "cat", "hub", "mlp", "mlp2", "et") if k in bases]
    T = [k for k in ("t_lgb", "t_xgb", "t_cat", "t_hub") if k in bases]
    D = [k for k in ("d_lgb", "d_xgb", "d_cat", "d_hub") if k in bases]
    F = [k for k in ("f_lgb", "f_xgb", "f_cat", "f_hub") if k in bases]
    N = sorted(k for k in bases if k.startswith("n_"))
    allf = (REG + T + D + F + ["t_clf", "t_clean", "d_clf", "d_clean",
            "f_clf", "f_clean", "p_cal", "ev", "p_cal_x_clean"] + N)
    r0, fo, _ = vf.evaluate(allf, "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
    fo = np.asarray(fo)
    res = (yv - fo).astype(np.float32)
    log(f"融合基线={r0:.5f};残差 std={res.std():.4f} train {xs_tr.shape} dev={dev}")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(xs_tr.shape[2], 96, batch_first=True)
            self.stat = nn.Linear(st_tr.shape[1], 32)
            self.head = nn.Sequential(
                nn.Linear(96 + 32, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xs, st):
            _, h = self.gru(xs)
            return self.head(torch.cat([h[-1], torch.relu(self.stat(st))], 1)).squeeze(1)

    XS = torch.from_numpy(xs_tr).to(dev)
    ST = torch.from_numpy(st_tr).to(dev)
    R = torch.from_numpy(res).to(dev)
    XSe = torch.from_numpy(xs_te).to(dev)
    STe = torch.from_numpy(st_te).to(dev)

    def infer(model, xs, st, bs=8192):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(xs), bs):
                outs.append(model(xs[i:i + bs], st[i:i + bs]))
        return torch.cat(outs).float().cpu().numpy()

    oof = np.zeros(len(yv))
    pred = np.zeros(len(xs_te))
    BS, MAX_EP, PAT = 1024, 40, 5
    for k, (tr, va) in enumerate(folds):
        torch.manual_seed(777 + k)
        np.random.seed(777 + k)
        model = Net().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        lossf = nn.MSELoss()
        tr_t = torch.from_numpy(tr).to(dev)
        best, best_ep, wait, best_state = 1e9, -1, 0, None
        for epn in range(MAX_EP):
            model.train()
            perm = tr_t[torch.randperm(len(tr_t), device=dev)]
            for i in range(0, len(perm), BS):
                b = perm[i:i + BS]
                opt.zero_grad()
                loss = lossf(model(XS[b], ST[b]), R[b])
                loss.backward()
                opt.step()
            vr = rmse(res[va], infer(model, XS[va], ST[va]))
            sch.step(vr)
            if vr < best - 1e-5:
                best, best_ep, wait = vr, epn, 0
                best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PAT:
                    break
        model.load_state_dict(best_state)
        oof[va] = infer(model, XS[va], ST[va])
        pred += infer(model, XSe, STe) / len(folds)
        log(f"  [rg] fold{k + 1}: res-RMSE={best:.5f} (ep={best_ep + 1})")
    os.makedirs("outputs/base_rg", exist_ok=True)
    np.savez("outputs/base_rg/resgru.npz", oof=oof, pred=pred)
    cor = np.corrcoef(oof, res)[0, 1]
    log(f"[rg] 残差 OOF 相关={cor:.4f}(残差可预测性);直接校正 rmse={rmse(yv, fo + oof):.5f}")

    bases["e_rg"] = (oof, pred)
    r1, _, _ = vf.evaluate(allf + ["e_rg"], "bayes", bases, y, ybin, folds, p_src="f_clf", clean_src="f_clean")
    d = r0 - r1
    log(f"判据[v25 res-GRU]:基线={r0:.5f} +rg={r1:.5f} → Δ={d:+.5f} "
        f"{'⚠️ 过门槛(须 OOF 串扰审计)' if d > 0.0005 else '❌ 不足'}")
    if d <= 0.0005:
        sys.exit(3)


if __name__ == "__main__":
    main()
