# -*- coding: utf-8 -*-
"""实验①:clean 回归(训练集剔除 outlier)+ outlier 概率后处理。

读取当前最新工件(特征缓存 / oof_predictions.csv / submission_stack.csv /
test_clf_prob.csv),在 OOF 上扫描两类后处理策略,择优生成 submission_clean_pp.csv:
  A. top-N 替换:clf 概率最高的 N 张卡改用 stack(受 outlier 拉动充分)预测
  B. 概率融合:pred = (1-p^a)*clean + p^a*stack
带 --tuned 参数时加载 outputs/best_params.json 的 LGB 终选参数(实验③之后复跑)。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep

rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))

if "--tuned" in sys.argv and os.path.exists("outputs/best_params.json"):
    bp = json.load(open("outputs/best_params.json"))
    ep.LGB_PARAMS.update(bp.get("lgb_best_trial", {}))
    print("[exp1] 使用 Optuna 终选 LGB 参数", flush=True)

base = pd.read_parquet("data/processed/features.parquet")
train = base[base["is_train"] == 1].reset_index(drop=True)
test = base[base["is_train"] == 0].reset_index(drop=True)
y = train["target"]
imp = pd.read_csv("outputs/feature_importance.csv")
selected = [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]
X, X_test = train[selected], test[selected]
folds = ep.make_folds(y)

oof_df = pd.read_csv("outputs/oof_predictions.csv")
assert (oof_df["card_id"].values == train["card_id"].values).all(), "OOF 与特征表行序不一致"
oof_stack = oof_df["oof_stack"].values
oof_clf = oof_df["oof_clf_prob"].values
stack_test = pd.read_csv("outputs/submission_stack.csv")["target"].values
clf_test = pd.read_csv("outputs/test_clf_prob.csv")["clf_prob"].values

# ---- clean LGB:fit 剔除 outlier,早停用干净验证子集,预测完整折(含 outlier 行)----
# 这样得到全行 clean OOF,才能与 y 直接模拟后处理的真实 RMSE
mask_out = (y < -30).to_numpy()
oof_clean = np.zeros(len(X))
pred_clean = np.zeros(len(X_test))
for k, (tr, va) in enumerate(folds):
    tr_c = tr[~mask_out[tr]]
    va_c = va[~mask_out[va]]
    m = lgb.train(ep.LGB_PARAMS, lgb.Dataset(X.iloc[tr_c], y.iloc[tr_c]), 10000,
                  valid_sets=[lgb.Dataset(X.iloc[va_c], y.iloc[va_c])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    oof_clean[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    pred_clean += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
    print(f"[exp1] clean fold{k + 1} iter={m.best_iteration}", flush=True)

base_stack = rmse(y, oof_stack)
print(f"[exp1] 基线 stack OOF={base_stack:.5f} | clean 全行 OOF={rmse(y, oof_clean):.5f}", flush=True)

results = []
for frac in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    n = int(frac * len(y))
    idx = np.argsort(-oof_clf)[:n]
    p = oof_clean.copy()
    p[idx] = oof_stack[idx]
    results.append(("topN", frac, rmse(y, p)))
for a in [0.5, 0.75, 1.0, 1.5, 2.0]:
    w = np.clip(oof_clf, 0, 1) ** a
    results.append(("blend", a, rmse(y, (1 - w) * oof_clean + w * oof_stack)))

res = pd.DataFrame(results, columns=["strategy", "param", "oof_rmse"]).sort_values("oof_rmse")
print(res.to_string(index=False), flush=True)
best = res.iloc[0]
print(f"[exp1] 最优: {best['strategy']} param={best['param']} OOF={best['oof_rmse']:.5f} "
      f"(vs stack {base_stack:.5f})", flush=True)

if best["strategy"] == "topN":
    n_test = int(best["param"] * len(clf_test))
    idx = np.argsort(-clf_test)[:n_test]
    final = pred_clean.copy()
    final[idx] = stack_test[idx]
else:
    w = np.clip(clf_test, 0, 1) ** best["param"]
    final = (1 - w) * pred_clean + w * stack_test

pd.DataFrame({"card_id": test["card_id"], "target": final}).to_csv(
    "outputs/submission_clean_pp.csv", index=False)
desc = f"clean+pp {best['strategy']}-{best['param']}, OOF {best['oof_rmse']:.5f}"
open("outputs/exp1_desc.txt", "w").write(desc)
json.dump({"best": [str(best["strategy"]), float(best["param"]), float(best["oof_rmse"])],
           "baseline_stack_oof": base_stack,
           "table": [[s, float(p), float(r)] for s, p, r in results]},
          open("outputs/exp1_report.json", "w"), indent=2)
print(f"[exp1] 保存 outputs/submission_clean_pp.csv ({desc})", flush=True)
