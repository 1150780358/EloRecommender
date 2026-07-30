# Elo 开源代码与 Top 思路分析(冲击 Top1 等效分数)

- **日期**:2026-07-29;当前成绩 Private 3.60965,Top1 3.58657,差距 0.021
- **前提事实**:Top1(30CrMnSiA/砍手豪)及前 10 名均**只有文字复盘、无完整开源代码**;GitHub 可得的最高名次完整代码为 **21st(bestpredicts/ELO,stacking Private≈3.601)** 与 63rd(takehiro177)

## 一、开源代码逐仓分析

### 1.1 bestpredicts/ELO(21st,21/4200)——主要挖掘对象

结构:两位队员的 notebook 集(zxs/、DY/、大量 checkpoint),最强单模 CV 3.6348(优于我们 3.6436),stacking LB 3.669 / PB 3.601。README 自述六类特征,代码核验结论:

| 招数 | 代码证据 | 我们现状 | 可移植性 |
|---|---|---|---|
| 聚合特征 + 参考日推算 | pre_0222:`purchase_date_first - relativedelta(months=month_lag_first)` 推 reference_date | 已有(v2 观察月) | 已覆盖 |
| **new×hist 系统性算术交叉(+-*/)** | best_single_model 特征名 `new_hist_X_-_hist_X`、`..._/_...`,先生成后筛选 | 仅 4 个比值 | **高,立即移植** |
| **CountVector 序列特征** | `CountVectorizer(token_pattern=u"\b\w+\b")` 对每卡类目/商户序列 | 无 | **高:TFIDF+TruncatedSVD 无依赖等价实现** |
| word2vec / node2vec 嵌入 | gen_node2vec_input 等 notebook | 无(本机无 gensim) | 中:SVD 嵌入先行,w2v 留后续 |
| **outlier 率目标编码** | `groupby(item)['outliers'].agg(click/count)`(CTR 风格) | 无 | 中:须折内计算防泄漏,不适合本轮无人值守,列为下轮 |
| NN / NFFM 基模型 | nn_model、nffm 系列 notebook | 无(torch 2.4+CUDA 可用) | 中:留下轮 |
| 组合量 `p_vs_m = amount/(|month_lag|+1)` | pre_0222 | 有同族(amount_month_ratio) | 已覆盖 |

### 1.2 takehiro177/Kaggle-Elo-63th-Solution(63rd)

单文件管线,增量招数:**众数聚合**(city/merchant_category/state/subsector/month_lag 的 mode 作为特征)——其 apply 实现在 29M 行上不可用,需向量化重写(pair 计数 + idxmax)。其余(month_diff、双表聚合)与我们重合。

## 二、Top 写方案(无代码)思路萃取

1. **Top1**:观察期锚定 + 围绕"参考月后窗口"的行为特征 + outlier 精细建模 + 大型多层融合;可复现部分为特征思想(我们 v2 已做浅层,深层=按 month_lag 对齐的逐月序列建模,对应本轮 month_lag pivot/斜率);
2. **7th(senkin13)**:FFM + count 向量;对应我们的序列嵌入方向;
3. **通用**:huber/fair 目标函数对 outlier 更稳(多队报告有效)、多 seed 平均降方差、month_lag 透视逐月序列。

## 三、差距归因与本轮(v4)作战计划

差距 0.021 分解:特征深度 ≈0.005-0.008(可追)、模型农场规模 ≈0.005-0.01(边际成本极高)、私有洞察 ≈0.003-0.005(部分可复现)。**现实目标:3.598-3.605(等效金/银牌分数);官方名次因封榜不可获得。**

v4 移植清单(全部无新依赖、无人值守稳健):
1. month_lag 透视(0..-6 逐月 count/sum)+ 月度金额斜率(cov/var 向量化)——Top1 逐月序列思想;
2. new 表按 month_lag=1/2 拆分聚合;
3. 众数聚合(63rd,向量化重写);
4. new×hist 算术交叉扩展(-、/ 共约 12 列,21st);
5. 序列嵌入:每卡 merchant_category_id 与 merchant_id 序列 → TFIDF → TruncatedSVD 8 维×2(21st CountVector 家族);
6. huber-LGB 第 4 基模型入 stacking(outlier 稳健);
7. 3 seed(2019/42/777)全套 + clean+pp,预测平均后提交。

下轮候选(本轮不做,理由):outlier 率目标编码(需折内重构防泄漏)、torch NN 与 w2v 嵌入(调试风险)、模型农场扩容(算力时间性价比)。
