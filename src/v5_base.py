# -*- coding: utf-8 -*-
"""v5 阶段 A-1:把基模型产物统一成 outputs/base/{name}.npz(oof + test 预测)。

背景:融合层此前无法重构,直接原因是产物不完整 ——
  · huber-LGB 只存了 OOF,test 预测从未落盘(main() 的 save_submission 名单里没有它);
  · clean 模型(剔除 outlier 训练)每次后处理实验都要重训一遍,结果也不落盘。
本脚本把已有的 lgb/xgb/cat/clf 迁移为标准格式,并重算 hub / clean 一次性存下,
之后所有融合实验都是秒级的纯代数运算。

用法:ELO_SEED=777 python src/v5_base.py [all|migrate|hub|clean]
      seed 须与 outputs/ 顶层产物一致。

关于 hub 的可复现性:实测重算的 hub 只有部分折能与 v4 产物逐元素吻合
(fold3/4 完全一致,其余折不同)。原因是 LightGBM 多线程下 histogram 浮点累加
顺序不固定,而 huber 目标的验证曲线在 5000-10000 轮区间极平坦,微小数值差异会让
early stopping 的 best_iteration 大幅漂移(如 7616 vs 10000 轮)。因此这里
  · 给 hub 打开 deterministic + force_row_wise,保证 v5 之后的实验可复现;
  · 自检改为统计一致性(OOF RMSE 偏差 + Pearson 相关),而非逐元素相等。
"""
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep

BASE_DIR = "outputs/base"
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))


def save_base(name: str, oof: np.ndarray, pred: np.ndarray, y):
    os.makedirs(BASE_DIR, exist_ok=True)
    np.savez(os.path.join(BASE_DIR, f"{name}.npz"), oof=oof, pred=pred)
    print(f"[base] {name:6s} OOF={rmse(y, oof):.5f}  -> {BASE_DIR}/{name}.npz", flush=True)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    seed = ep.CONFIG["SEED"]
    print(f"[base] stage={stage} seed={seed}(须与 outputs/ 顶层产物一致)", flush=True)

    base = pd.read_parquet("data/processed/features.parquet")
    train = base[base["is_train"] == 1].reset_index(drop=True)
    test = base[base["is_train"] == 0].reset_index(drop=True)
    y = train["target"]

    imp = pd.read_csv("outputs/feature_importance.csv")
    selected = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"]
                if c in train.columns]
    X, X_test = train[selected], test[selected]
    print(f"[base] X={X.shape} X_test={X_test.shape}", flush=True)

    oof_df = pd.read_csv("outputs/oof_predictions.csv")
    assert (oof_df["card_id"].values == train["card_id"].values).all(), "OOF 与特征表行序不一致"
    folds = ep.make_folds(y)

    # ---- 1) 迁移已有产物:oof 来自 oof_predictions.csv,test 来自 submission_{name}.csv ----
    if stage in ("all", "migrate"):
        for name, oof_col in [("lgb", "oof_lgb"), ("xgb", "oof_xgb"), ("cat", "oof_cat")]:
            sub = pd.read_csv(f"outputs/submission_{name}.csv")
            assert (sub["card_id"].values == test["card_id"].values).all(), f"{name} test 行序不一致"
            save_base(name, oof_df[oof_col].values, sub["target"].values, y)
        clf_test = pd.read_csv("outputs/test_clf_prob.csv")
        assert (clf_test["card_id"].values == test["card_id"].values).all(), "clf test 行序不一致"
        np.savez(os.path.join(BASE_DIR, "clf.npz"),
                 oof=oof_df["oof_clf_prob"].values, pred=clf_test["clf_prob"].values)
        print(f"[base] clf    (二分类概率) -> {BASE_DIR}/clf.npz", flush=True)

    # ---- 2) 重算 huber-LGB(test 预测缺失;打开 deterministic 保证今后可复现)----
    if stage in ("all", "hub"):
        print("[base] 重算 huber-LGB 十折(deterministic)...", flush=True)
        hub_params = {**ep.HUB_PARAMS, "deterministic": True, "force_row_wise": True}
        oof_hub, pred_hub, _, _ = ep.cv_lightgbm(X, y, X_test, folds, hub_params, "hub")
        save_base("hub", oof_hub, pred_hub, y)   # 先落盘,校验失败也不白跑
        ref = oof_df["oof_hub"].values
        d_rmse = abs(rmse(y, oof_hub) - rmse(y, ref))
        corr = float(np.corrcoef(oof_hub, ref)[0, 1])
        print(f"[base] hub 统计自检:ΔOOF_RMSE={d_rmse:.5f} corr={corr:.6f}", flush=True)
        assert d_rmse < 0.005 and corr > 0.999, (
            f"hub 与顶层产物差异过大(ΔRMSE={d_rmse:.4f} corr={corr:.4f})—— "
            f"疑为 seed/参数/特征列不配套,请核对 ELO_SEED")

    # ---- 3) 重算 clean-LGB(训练侧剔除 outlier,预测完整折)----
    # fit 只用非 outlier 行、早停验证集也只用非 outlier 行,但预测整个 va(含 outlier),
    # 这样得到全行 OOF 才能与 y 直接模拟融合后的真实 RMSE。
    if stage in ("all", "clean"):
        print("[base] 重算 clean-LGB 十折 ...", flush=True)
        mask_out = (y < -30).to_numpy()
        oof_clean = np.zeros(len(X))
        pred_clean = np.zeros(len(X_test))
        for k, (tr, va) in enumerate(folds):
            tr_c, va_c = tr[~mask_out[tr]], va[~mask_out[va]]
            m = lgb.train(ep.LGB_PARAMS, lgb.Dataset(X.iloc[tr_c], y.iloc[tr_c]), 10000,
                          valid_sets=[lgb.Dataset(X.iloc[va_c], y.iloc[va_c])],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            oof_clean[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
            pred_clean += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
            print(f"  [clean] fold{k + 1}: iter={m.best_iteration}", flush=True)
        save_base("clean", oof_clean, pred_clean, y)
        # 干净子集上的 RMSE 才是 clean 模型的真实水平(全行 RMSE 被 outlier 支配)
        print(f"[base] clean 非 outlier 子集 OOF={rmse(y[~mask_out], oof_clean[~mask_out]):.5f}",
              flush=True)

    print("[base] 完成。融合实验可直接读 outputs/base/*.npz", flush=True)


if __name__ == "__main__":
    main()
