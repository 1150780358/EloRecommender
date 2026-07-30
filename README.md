# Elo Merchant Category Recommendation — 完整建模管线与刷榜复盘

Kaggle [Elo Merchant Category Recommendation](https://www.kaggle.com/competitions/elo-merchant-category-recommendation) 的端到端解决方案:从 2900 万行交易数据的特征工程,到分层十折 + 多模型 Stacking,再到基于**真实私榜分数表**的成绩定级与逐轮归因。

## 成绩

| 指标 | 数值 |
|---|---|
| Private RMSE | **3.60753** |
| 私榜等效名次 | **第 69 名 / 4111(top 1.7%,银牌区)** |
| OOF RMSE(分层十折) | 3.63455 |
| 金牌线(18 名) | 3.60078(尚差 0.00675) |

> **成绩口径**:本赛于 2019-02-26 截止,当前提交为 **Late Submission** —— Kaggle 正常评分并返回 Public/Private 分数,但**不进官方榜、不排名、不授奖牌**。「第 69 名」为等效名次:用 Kaggle API 取回真实私榜 Top200 分数表(`outputs/private_lb_top200.json`)作标尺,将本方案 Private 分数插入定位所得。不宣称获奖。

迭代曲线(每一步均有离线 OOF 支撑,详见 `docs/submissions.md`):

| 版本 | 关键改动 | Private | 等效名次 |
|---|---|---|---|
| v1 | 三模型 Ridge Stacking | 3.61276 | ~210 |
| v2 | 观察月锚定特征 | 3.60974 | 114 |
| v3 | Optuna 终选参数 | 3.60965 | 112 |
| v4 | 开源方案移植特征 + huber 基模型 + 3-seed | 3.60759 | 69 |
| v5 | 融合层重构 + 异构基学习器 | 3.60753 | 69 |

## 方法概要

**特征工程(2900 万行,约 4 分钟)**:内存压缩(int64→int8/16)、业务化分层清洗(installments 哨兵值、金额脱敏线性逆变换 + winsorize)、双表 card 粒度聚合(分位数/极差/nunique/授权占比)、消费节奏间隔统计、月度波动、近三月趋势比、商户/城市集中度与熵、month_lag 透视与金额斜率、众数聚合、序列 TFIDF→SVD 嵌入、hist↔new 跨表算术交叉。

**验证**:target 连续无法直接分层,按 outlier(`< -30`)二值做 `StratifiedKFold(10)`,全模型共享折划分 —— 这是本赛 CV/LB 相关性的关键。

**模型**:LightGBM / XGBoost / CatBoost / huber-LGB 四回归 + outlier 二分类(AUC 0.902)+ clean 回归(训练侧剔除 outlier);二层 Ridge / BayesianRidge Stacking。

## 文件地图

```
src/                          全部 Python 源码(同目录互相 import,统一从项目根运行)
├─ elo_pipeline.py            主管线:特征→筛选→分层十折→四模型+分类器→Stacking→落盘
├─ exp1_clean_postprocess.py  clean 回归 + outlier 概率后处理(topN vs 概率软融合)
├─ exp3_tune.py               Optuna TPE 复搜三模型超参
├─ avg_seeds.py               多 seed 预测平均
├─ submit_and_log.py          提交 Kaggle 并自动回填 docs/submissions.md
├─ v5_base.py                 基模型产物统一为 outputs/base/{name}.npz(oof + test)
├─ v5_hetero.py               异构基学习器:torch MLP / Huber-MLP / ExtraTrees
├─ v5_clf_boost.py            outlier 分类器增强(XGB/CAT 二分类 + rank 融合)
├─ v5_fusion.py               融合层实验:元特征扩展 × 二层模型选型(16 个方案同折对比)
├─ v5_features_ext.py         增量特征族:被拒交易画像 + 观察期末端近窗
└─ v5_train_ext.py            扩展特征集训练与单模判据

scripts/                      无人值守实验战役(nohup + setsid,日志入 outputs/logs/)
├─ run_campaign.sh            v2/v3:观察月特征 → clean+pp → Optuna 复搜
└─ run_campaign2.sh           v4:移植特征 + huber 基模型 + 3-seed 平均

data/raw/ · data/processed/   官方数据 · 特征缓存 parquet(均不入库)
outputs/                      提交 csv、OOF、base/*.npz 基模型产物、logs/ 运行日志
docs/submissions.md           提交流水、私榜标尺、等效名次一览
docs/202607/                  任务日志(Analysis / Plan / Progress / Review)
docs/实习周报计划.md           四周计划
docs/周报/                    正式周报(第 1 周已定稿)
refs/                         第三方开源方案参考(不入库)
```

## 复现

```bash
pip install -r requirements.txt
# Kaggle 官方数据放到 data/raw/(train.csv / test.csv / historical_transactions.csv /
# new_merchant_transactions.csv / merchants.csv)

# 以下命令均从项目根目录运行(src 内脚本同目录互相 import)
python src/elo_pipeline.py              # 全量:特征构建 + 训练 + 融合(约 25 分钟)
python src/exp1_clean_postprocess.py    # clean 模型 + 后处理择优
ELO_SEED=777 python src/v5_base.py all  # 基模型产物统一落盘
ELO_SEED=777 python src/v5_fusion.py    # 融合层方案对比(秒级)
```

环境:64 核 CPU / 125G 内存 / CUDA(MLP 可选)。`CONFIG["DEBUG"]=True` 可抽样端到端快速验证。

## 方法论结论(本项目最有价值的产出)

1. **公榜排名在本赛完全无参考价值**。私榜大洗牌:公榜第一 3.61285 在私榜仅排 ~200;早期用公榜(1493/4111)判断进展造成严重误导。一切以 OOF + 私榜标尺为准。
2. **OOF 改善到后期不再可靠转化为 Private**。v5 在 OOF 上赚 0.00285,线上只剩 0.00006,CV-LB 偏移从 0.03043 漂到 0.02721 —— 偏移本身在变,说明多赚的是"OOF 专属收益"。判据应升级为:**降低 OOF 且不缩小 CV-LB 偏移**。
3. **Stacking 在基模型方差差异大时有系统性偏差**。OOF 是「9 折训练、预测 1 折」的单模型输出,test 是「10 折平均」输出;折平均对高方差模型(MLP/ExtraTrees)的降噪幅度远大于低方差的 GBDT,二层因此高估异构成员的边际价值。
4. **特征重要性占比 ≠ 信息增量**。149 个新特征在 gain 排序中占 38.3%,单模 OOF 反而变差 0.0018;冗余特征与已有强特征相关,同样能拿到高 gain。判据只能是 OOF。
5. **解析式后处理对线性二层可能冗余**。期望值融合 `ev = p·(-33.219) + clean − p·clean` 恰是 `[p, clean, p×clean]` 的线性组合,对 Ridge/Bayes 不增加信息;实验完全符合。
6. **huber-LGB 的不可复现根因**:多线程 histogram 浮点累加顺序 × huber 验证曲线在 5000-10000 轮极平坦 → `best_iteration` 大幅漂移(7616 vs 10000)。打开 `deterministic + force_row_wise` 可解(代价:慢 4 倍)。

失败尝试同样记录在案(`docs/202607/30-v5_fusion_heterogeneous.md`):融合层重构、异构基学习器、outlier 分类器增强、增量特征族 —— 四个方向全部证伪。

## 致谢

`refs/` 下参考了 [bestpredicts/ELO](https://github.com/bestpredicts/ELO)(21st)与 [takehiro177/Kaggle-Elo-63th-Solution](https://github.com/takehiro177/Kaggle-Elo-63th-Solution)(63rd)的公开方案,已在 `.gitignore` 中排除、不纳入本仓库。
