# -*- coding: utf-8 -*-
"""v5 阶段 C 训练:合并增量特征 → 重新筛选 → 训练基模型 → 落盘 outputs/base_ext/。

折划分与 outputs/base/ 完全一致(同 seed、同 make_folds),因此两套特征训练出的
模型可以直接混进同一个融合层 —— 特征集差异带来的多样性通常比算法差异更有效
(v5 已证明换算法只带来 OOF 上的虚假收益)。

止损设计:先只训 lgb,与 v4 的 lgb(OOF 3.64291)对比。
新特征若拿不到 >0.001 的单模改善,就没有必要付全套训练的时间成本。

用法:
    python src/v5_train_ext.py select      # 合表 + 特征筛选(产出 imp_ext.csv)
    python src/v5_train_ext.py lgb         # 快速判据:单模是否有增益
    python src/v5_train_ext.py rest        # 其余基模型(xgb/cat/hub/clf/clean)
"""
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import elo_pipeline as ep

# TOP_K 可由环境变量覆盖:加入 149 个新特征后,固定 TOP_K=300 会把原有特征从 249 个
# 挤到 185 个,单模变差无法区分是「新特征无用」还是「老特征被挤掉」。ELO_TOPK=400
# 等于全部保留(方差过滤后共 400 列),才是新特征的公平判据。
if os.environ.get("ELO_TOPK"):
    ep.CONFIG["TOP_K"] = int(os.environ["ELO_TOPK"])

BASE_DIR = "outputs/base_ext"
MERGED = "data/processed/features_all.parquet"
IMP = "outputs/feature_importance_ext.csv"
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))


def build_merged():
    """主特征表 + 增量特征表合并(card_id 左连接,行序保持主表)。"""
    base = pd.read_parquet("data/processed/features.parquet")
    ext = pd.read_parquet("data/processed/features_ext.parquet")
    base["card_id"] = base["card_id"].astype(str)
    ext["card_id"] = ext["card_id"].astype(str)
    n0 = base.shape[1]
    merged = base.merge(ext, on="card_id", how="left")
    print(f"[ext] 合表 {n0} -> {merged.shape[1]} 列(新增 {merged.shape[1] - n0})", flush=True)
    merged.to_parquet(MERGED)
    return merged


def load_merged():
    return pd.read_parquet(MERGED) if os.path.exists(MERGED) else build_merged()


def get_xy():
    m = load_merged()
    train = m[m["is_train"] == 1].reset_index(drop=True)
    test = m[m["is_train"] == 0].reset_index(drop=True)
    y = train["target"]
    return train, test, y


def stage_select():
    train, test, y = get_xy()
    drop = {"card_id", "target", "is_train"}
    feat_cols = [c for c in train.columns if c not in drop]
    selected, imp_df = ep.select_features(train, y, feat_cols)
    imp_df.to_csv(IMP, index=False)
    n_new = sum(1 for c in selected if c.startswith(("decx_", "m0_", "m01_", "m012_")))
    print(f"[ext] 筛选后 {len(selected)} 维,其中新特征 {n_new} 个"
          f"({n_new / len(selected):.1%})—— 占比高说明新特征确实带信息", flush=True)
    return selected


def selected_cols(train):
    imp = pd.read_csv(IMP)
    return [c for c in imp[imp["gain"] > 0].head(ep.CONFIG["TOP_K"])["feature"] if c in train.columns]


def save_base(name, oof, pred, y):
    os.makedirs(BASE_DIR, exist_ok=True)
    tag = f"{name}_k{ep.CONFIG['TOP_K']}" if ep.CONFIG["TOP_K"] != 300 else name
    np.savez(os.path.join(BASE_DIR, f"{tag}.npz"), oof=oof, pred=pred)
    print(f"[ext] {tag:10s} OOF={rmse(y, oof):.5f}  -> {BASE_DIR}/{tag}.npz", flush=True)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "select"
    if stage == "select":
        stage_select()
        return

    train, test, y = get_xy()
    sel = selected_cols(train)
    X, X_test = train[sel], test[sel]
    folds = ep.make_folds(y)
    print(f"[ext] stage={stage} X={X.shape}", flush=True)

    if stage in ("lgb", "all"):
        oof, pred, _, _ = ep.cv_lightgbm(X, y, X_test, folds, ep.LGB_PARAMS, "lgb")
        save_base("lgb", oof, pred, y)
        ref = np.load("outputs/base/lgb.npz")["oof"]
        print(f"[ext] 判据:lgb OOF {rmse(y, oof):.5f} vs v4 特征 {rmse(y, ref):.5f} "
              f"→ 改善 {rmse(y, ref) - rmse(y, oof):+.5f}", flush=True)

    if stage in ("rest", "all"):
        oof, pred, _ = ep.cv_xgboost(X, y, X_test, folds)
        save_base("xgb", oof, pred, y)
        oof, pred, _ = ep.cv_catboost(X, y, X_test, folds)
        save_base("cat", oof, pred, y)
        oof, pred, _, _ = ep.cv_lightgbm(X, y, X_test, folds, ep.HUB_PARAMS, "hub")
        save_base("hub", oof, pred, y)
        oof, pred, auc = ep.cv_outlier_clf(X, y, X_test, folds)
        os.makedirs(BASE_DIR, exist_ok=True)
        np.savez(os.path.join(BASE_DIR, "clf.npz"), oof=oof, pred=pred)
        print(f"[ext] clf AUC={auc:.5f}(v4 特征 0.90235)", flush=True)
        # clean 模型:训练侧剔除 outlier,预测完整折
        mask = (y < -30).to_numpy()
        oc = np.zeros(len(X))
        pc = np.zeros(len(X_test))
        for k, (tr, va) in enumerate(folds):
            tr_c, va_c = tr[~mask[tr]], va[~mask[va]]
            m = lgb.train(ep.LGB_PARAMS, lgb.Dataset(X.iloc[tr_c], y.iloc[tr_c]), 10000,
                          valid_sets=[lgb.Dataset(X.iloc[va_c], y.iloc[va_c])],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            oc[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
            pc += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
            print(f"  [clean] fold{k + 1}: iter={m.best_iteration}", flush=True)
        save_base("clean", oc, pc, y)


if __name__ == "__main__":
    main()
