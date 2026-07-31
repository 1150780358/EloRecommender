# Elo Top 方案与开源代码对比分析(逐方案 vs baseline)

- **日期**:2026-07-31
- **材料来源**:Kaggle 讨论区原帖(经渲染抓取,URL 见各节)、作者 GitHub 仓库、赛后总结博客;私榜前 40 名经 kaggle CLI 实拉核对名次。18th 一节为二手转述(已标注),其余均为一手材料。
- **检索未果**(明确说明,不作猜测):2nd/3rd/4th/6th/8th/9th/14th/15th/17th/20th/23rd–27th 无公开 writeup;1st 作者的知乎长文未能获取原文(其 Kaggle/知乎主页已确认)。
- **⚠️ 修正此前调研结论**(`29-opensource_analysis.md`):前 10 名中 **7th(senkin13)有完整开源代码仓** `github.com/senkin13/kaggle/tree/master/elo`(lgb/nn/ffm/graph/nested_model/ensemble 等脚本齐全),此前"前 10 仅文字复盘、无完整开源代码"的表述应修正为"**前 10 仅 7th 开源完整代码**;其余可得完整代码为 21st 与 63rd"。
- **配套文档**:官方数据分析 `31-官方数据文件分析.md`;初版开源仓核验 `29-opensource_analysis.md`;我们的 TE 实践 `30-v6_target_encoding.md`。

## 〇、对比基线(baseline)定义

下文所有"较 baseline 增量"均以**社区公开 kernel 级基线**为参照(官方不提供 baseline 代码):

- **构成**:基础 groupby 聚合(计数/金额 sum/mean/max/min/std、去重基数)+ month_diff 近期性 + 简单日期特征,单一 LGB 五折;即 63rd `ELOmybaseline.py` 的骨架,也是当年多数公开 kernel 的水平;
- **成绩档位**:Public ≈3.69、Private ≈3.61 上下(对照:我们 v1 三模型 stacking 首提 Private 3.61276;私榜金牌线 18 名 = 3.60078,Top1 = 3.57875);
- **含义**:从 baseline 到 Top1 全程约 0.03,前排方案的每一分增量都来自下文列出的"baseline 之外多做的事"。

## 一、私榜前 22 名总览(kaggle CLI 实拉)

| 名次 | 队伍/作者 | Private | 公开材料 |
|---|---|---|---|
| 1 | Look alive(砍手豪 solo) | 3.57875 | 讨论帖 82036(trick 帖) |
| 2 | 陪elo一起过年 | 3.59019 | 无 |
| 3 | GideonTeo | 3.59319 | 无 |
| 4 | TH & daishu & Q | 3.59375 | 无 |
| 5 | Evgeny Patekha(solo) | 3.59422 | 讨论帖 82314 |
| 6 | Lucky stars | 3.59707 | 无 |
| 7 | You'll Never Overfitting Alone(senkin13) | 3.59779 | 讨论帖 82055 + **完整代码仓** |
| 10 | [ods.ai] YuryBolkonskiy | 3.59904 | 讨论帖 82093 |
| 11 | Stack It All | 3.59940 | 讨论帖 82127 |
| 13 | Loyalty overrated(raddar) | — | 洞察帖 82088(方案主体未公开) |
| 16 | nlgn(Anthony Chiu,solo) | 3.60065 | 讨论帖 82166 |
| 18 | pocket 所在队 | ≈3.6007 | 讨论帖 82107(二手转述) |
| 19 | hmdhmd | 3.60099 | 讨论帖 82178 |
| 21 | Trust Your CV(bestpredicts 团队) | 3.60142 | 讨论帖 82235 + **完整代码仓** |
| 22 | Grand Rookie(Orange) | 3.60146 | 讨论帖 82057 |

(8th/9th/12th/14th/15th/17th/20th 无公开材料,略;28th/31st/55th/63rd 等见 §三)

## 二、前排方案逐个分析(较 baseline 的不同)

### 2.1 — 1st:砍手豪(solo,Private 3.57875)

> 来源:kaggle.com/c/elo-merchant-category-recommendation/discussion/82036

**较 baseline 的增量**:

1. **离群软融合公式**(核心 trick,较同特征直接回归 CV +0.015):`final = p_outlier × (-33.219) + (1 − p_outlier) × pred_no_outlier` —— 用二分类概率对哨兵值与 clean 回归做**期望值加权**,而非 baseline 的单一回归、也非常见的 top-k 硬置 -33;
2. **行级 meta feature(嫁接法,分类器最强特征)**:把 train 的 target merge 到**逐笔交易行**,训练行级模型,再把行级预测按卡聚合(min/max/mean)作为卡级特征——把"卡级监督信号"下放到交易粒度再收回,是 baseline 完全没有的信息通道;
3. **xentropy 归一化训练**:把 target 归一化到 [0,1] 后用 LGB xentropy objective 按分类方式训练回归(源自 Avito 赛的做法);
4. 关键数字:离群二分类 AUC **0.914**(我们 0.904);clean 回归 CV **1.545**。

**对我们的启示**:软融合我们已实践(clean+pp 即同族);**行级 meta feature 未做,是最高优先级的空白**;AUC 差距 0.01 说明分类器特征仍有空间。

### 2.2 — 5th:Evgeny Patekha(solo,Private 3.59422)

> 来源:discussion/82314

**较 baseline 的增量**:

1. **五模型分段架构**:① 全量回归 → ② 离群二分类(阈值 0.015 把 train/test 切成低/高概率两段)→ ③ 低概率段回归(对漏网离群**降权 0.4** 而非剔除)→ ④ 高概率段回归(**以分类概率 + 全量回归预测为核心特征**,以模型替代 -33 硬后处理——作者自述是躲过私榜洗牌的关键)→ ⑤ 分段拼接后再与全量回归 blend;
2. **两步加权 target encoding**(其特色):先对特征组合(authorized_flag+category_1+subsector_id、merchant_id 等)算目标均值 join 回**交易表**,再按 card_id 聚合;TE 与聚合均按 month_lag 加权衰减;双重 out-of-fold + 正则防泄漏;
3. **极限特征筛选**:数千特征精选到约 **100 个**;
4. 关键数字:分类器单模 AUC 0.9131、4 模型 blend 0.9141;最终本地 RMSE 3.609。

**对我们的启示**:其 TE 与我们 v6 同范式但更深(交易粒度、组合键、时间加权)——TE 键集扩展方向有据可循,但其双重 OOF 防泄漏的严谨度正是我们 TE-v2 事故缺失的;"以模型替代硬后处理"与我们 v5 期望值融合结论一致。

### 2.3 — 7th:senkin13(Private 3.59779,**前 10 唯一开源完整代码**)

> 来源:discussion/82055;代码 github.com/senkin13/kaggle/tree/master/elo

**较 baseline 的增量**:

1. **8 种数据集切分**分别出特征:仅 hist / hist(authorized=1) / 仅 new / hist(auth)+new / hist+new / hist+merchants / new+merchants / 全并——baseline 只有 hist、new 两套;
2. 特征族:raddar 金额去匿名化后**重算一套聚合**;相邻交易的日期差/金额差(interval);交互强特 `new.purchase_date.max() / hist.purchase_date.max()`;**tf-idf+TruncatedSVD(5 维)** 与 **word2vec** 压缩每卡商户/类目序列;代码仓另有 card–merchant 二部图 **deepwalk/node2vec 64 维嵌入**;
3. **行级 meta model**(同 1st 的嫁接法):行级 LGB 预测按卡 min/sum 聚合,**CV/LB 均 +0.005~0.006**,top20 重要度里占 6 席;
4. **null importance 特征选择**,12 套特征集(200~700 维);
5. 模型与融合:最优单模 LGB 385 特征 **CV 3.6144 / Private 3.593(单模即可排第 3)**;stage1 = 12 LGB + 40 NN,stage2 结论:**二层用 NN/Ridge 可行、LGB/ExtraTrees 严重过拟合**;
6. 负结果(同样宝贵):**isotonic 校准 CV/LB +0.005 但 Private 变差(其 shake-down 主因)**;花 100 次提交 LB probing 探出公榜 24 个离群,伪标签无增益。

**对我们的启示**:三处与我们的实验互相印证——isotonic 校准害处(我们 v5 证伪)、二层 LGB 过拟合(我们二层用 BayesianRidge)、单模上限主要由特征决定;**行级 meta model 与图嵌入是我们未覆盖的两大特征通道,且有现成代码可核**。

### 2.4 — 10th:YuryBolkonskiy(Private 3.59904)

> 来源:discussion/82093

**较 baseline 的增量**:

1. 特征做到约 **6,500 个**:CountVectorizer(merchant_id/subsector/类目)+PCA/SVD;**month_lag 逐月 pivot**(逐月 count/sum/std/min/max);逐月金额对前期的比值(间隔 2/4/6);**预测 month_lag=+3/+4 的金额再取比值(外推未来月)**;Rolling mean 与指数平滑刻画逐月序列;交易间隔时长统计;
2. **特征选择三管齐下**:Boruta(8 小时选 500)、三家 GBDT 重要度、**对抗验证删除区分 train/test 的特征**;
3. **模型农场**:47 个含离群模型 + 35 个去离群模型分别 stacking(CV 3.637 / 1.544),按离群概率替换 10,000 个低概率样本的预测;最优单模 CatBoost CV 3.645;分类器 ROC 0.907。

**对我们的启示**:month_lag pivot 我们 v4 已做;**未来月外推、对抗验证特征筛选未做**;82 模型农场印证差距归因中"模型规模 ≈0.005–0.01 但边际成本极高"的判断。

### 2.5 — 11th:Stack It All(Private 3.59940)

> 来源:discussion/82127

**较 baseline 的增量**:

1. 两套特征集(1000+ 与 200+):duration/count 复合(`durations/sqrt(counts)` 等);类别频次族(frequence/Maxfrequence/比率);card/merchant/类目/城市**访问序列 seq2seq embedding**;hist×new 的 diff/ratio 全家桶;**创意筛选:按相关性矩阵给每个特征配"最不相关伙伴"再造一批交叉聚合,产出强特**;
2. **32 个模型(LGB/CAT/XGB/H2O RF/GBM)Bayesian Ridge stacking**,CV 3.630X;
3. **保守后处理**:4 个不同分类器各取 top100 离群,**四集合求交集仅 21 个**置 -33(LB 3.675→3.666);两个最终提交 Private 均 3.599;
4. 无效清单:NN、AutoEncoder 异常检测、PCA/TSNE、FM/FFM、Isolation Forest 等。

**对我们的启示**:二层 BayesianRidge 与我们 v5 终选一致;"多分类器交集"是比单模 top-k 稳健得多的硬后处理形态(我们证伪的是单分类器 top-k 硬替换)。

### 2.6 — 13th:raddar(洞察贡献者,方案主体未公开)

> 来源:discussion/82088 及其三个公开 kernel

非完整方案,但其三项**数据洞察被前排广泛引用**,均为 baseline 层面之下的"数据理解"增量:target 真实含义推断、**purchase_amount 去匿名化**(7th 直接引用为新特征底料,我们管线亦采用)、凭历史交易+target **反推模拟未观测的未来数据**。

### 2.7 — 16th:Anthony Chiu(solo 首金,Private 3.60065)

> 来源:discussion/82166

**较 baseline 的增量**(方案重心在**特征选择与稳健性**,而非特征数量):

1. 训练前 **KS 检验**删除 train/test 分布不一致的特征;训练后**修改版 null importance(permutation)** 排名,再按保留 90%→10% 循环搜索最优特征量;
2. **修改版三模型法**:clean 模型不用真实非离群标签训练,而用**分类器预测的非离群样本**(train/test 都按 `pred ≥ quantile(0.9)` 划分)——让 CV 能如实模拟"分类误差向下游继承",提升略小但更稳;
3. 最终 blend 16 个模型;**不做任何 -33 手工后处理**;最优 3-model CV 3.63469 / Private 3.604。

**对我们的启示**:KS/对抗验证类特征筛选我们未做;其"让 CV 模拟误差传递"的思想与我们"判据升级(OOF+偏移)"同源,但落在了架构设计层面,值得借鉴。

### 2.8 — 18th:pocket 所在队(Private ≈3.6007)⚠️ 二手转述

> 原帖 82107 渲染失败,内容转自 amalog.hateblo.jp 总结博客

**较 baseline 的增量**:**最后一笔交易日期是 magic feature**;按 authorized_flag==0、city_id==-1 等**条件切片聚合**;仅聚合近期交易;**训练一个"预测 magic feature"的模型,用其残差作特征**;二层 Ridge 优于 LGB;最终 Ridge 融合 [离群概率, 去离群预测, 全量预测] 三路。

### 2.9 — 19th:hmdhmd(Private 3.60099)

> 来源:discussion/82178

**较 baseline 的增量**:

1. **one-hot 商户聚合**:先按离群占比筛商户(`groupby(merchant_id)['outlier'].agg`,如出现 >500 次且离群率 >2% 的商户)再 one-hot——**与我们 v6 TE 同一信息源(商户×离群率),实现形态不同**;
2. 最优单模 = 12 折 LGB、300 特征,CV 3.634 / Private 3.605;最终多特征集 LGB blend,Private 3.600;
3. **完全无后处理,且其最优 CV 模型恰好就是最优私榜模型**——"trust your CV" 的极端案例;
4. 无效清单(对我们特别有参考价值):**直接 target encoding 无效**(印证 TE 成败在防泄漏实现,而非概念)、NN/RNN/XGB/CAT/FFM、SVD/LDA、Boruta;赛后发现行级预测特征可 Private +0.003,当时未敢用。

### 2.10 — 22nd:Orange(Private 3.60146)

> 来源:discussion/82057

**较 baseline 的增量**(自报以公开 kernel 为 baseline,特征做到约 1 万):

1. **month_lag 阈值子集重算**:取 month_lag ≥ -3、≥ -6 的子集,把全部 baseline 特征**原样重算一遍**(lag_3_*/lag_6_*),自报**至少 +0.010** —— 性价比极高的"近期窗口"族;
2. **偏好商户画像**:找每卡 preferred merchant_id,join merchants.csv 的属性(sales_range/numerical_1/category_4 等)做特征——merchants 表的少数有效用法之一;
3. word2vec 用 "what-and-when-and-where" 格式**把交易造句**后训练;
4. 也实现了 1st 同款软融合公式,**但最终提交未选它**;信条 "trust your local cv"。

## 三、开源代码仓与 20 名外补充

### 3.1 — 21st:bestpredicts/ELO(队 "Trust Your CV",Private 3.60142)—— 我们的主要挖掘对象

> 来源:discussion/82235;代码 github.com/bestpredicts/ELO(已在 `refs/ELO/` 逐仓核验,详见 `29-opensource_analysis.md`)

**较 baseline 的增量**:六类特征——聚合、**new×hist 四则算术交叉**、日期、**CountVectorizer 序列特征**、word2vec、**NN oof 特征**;仓库另有 node2vec 输入生成、NFFM、去离群单模与组合 notebook;**约 70 个 LGB+NN 模型 oof stacking:CV 3.627 / Private 3.601**(单模 LB 仅 3.68 档,融合规模补足)。

**我们的移植成果**:算术交叉与 TFIDF-SVD 序列嵌入(v4,Private −0.002)、**折外 outlier 率 TE(v6,收紧其泄漏后 Private −0.00118,当前最优 3.60635 的来源)**;未移植:NN oof 特征、node2vec、NFFM。

### 3.2 — 63rd:takehiro177(我们的 baseline 参照物)

> 代码 `refs/Kaggle-Elo-63th-Solution/`(1209 行 baseline + 543 行 finalline,已实读)

本质 = baseline 骨架 + **众数聚合**(city/类目/state/subsector/month_lag 的 mode)+ month_diff;其 apply 实现在 29M 行上不可扩展,我们已向量化重写移植(v4)。它与前排的差距正好反衬 §二各项增量的价值。

### 3.3 — 其他有公开材料的名次

| 名次 | 要点(较 baseline 增量) | 来源 |
|---|---|---|
| ~12–17(佚名帖 82375) | **多标准切分聚合**(hist/new × month_lag 逐月/区间/rank × 日期 × authorized × category 组合切片);`groupby(card_id).rank()` 造 month_rank/date_rank 强特;**每卡众数类别作"真类别特征"**;merchants 仅 category_4/merchant_group_id 有用;93 特征;报 TE/tfidf/FFM 无效 | discussion/82375 |
| 28th bangdasun | merchant 类目 **LDA 嵌入**;6 LGB + 2 kernel blend;做了 top-k 置 -33 与不后处理两版提交——**胜出的是不后处理版** | GitHub README |
| 31st/32nd | Ridge/Lasso 三路融合(20 模型 stack 回归 + 12 模型 stack 分类 + 9 模型 stack 均值替换回归);多样性来自 poisson loss/target 分桶多分类/80 个欠采样分类器;CV 3.6253 | discussion/82084、82126 |
| 55th | tf-idf 计数特征**加 Ridge 中间层**再入模(+0.002);w2v/fasttext 算 hist–new 相似度(+0.001);RFE+LGB(RF booster)筛特征(+0.006);改造公开 3-model kernel 使 clean 模型也出全量 OOF 从而可 CV(+0.003) | discussion/82062 |
| 方法论帖 | 73937 "Less is More"(少特征+强选择);77537 "Reducing the gap between CV and LB"(分布检验删特征,16th 引用) | — |

## 四、横向对照:各方案在关键维度上的选择

| 维度 | 前排做法(名次) | 我们的现状 |
|---|---|---|
| 离群处理 | **软融合期望值**(1st/22nd);分段建模替代硬后处理(5th);多分类器交集置-33(11th);**完全不后处理**(16th/19th/28th 胜出提交) | clean+概率软融合(同 1st 族)✔;单分类器 top-k 硬替换已证伪 ✔ |
| target encoding | 两步加权 TE 强(5th);**直接 TE 无效**(19th/82375) | 折内 TE 有效(v6)✔;泄漏版事故(TE-v2)与 19th 结论互证 |
| 行级 meta model(target 嫁接交易行) | **1st(分类器最强特征)/7th(+0.005~0.006)/19th(赛后 +0.003)** | **未做——最大空白** |
| 序列/嵌入 | tfidf+SVD、w2v(7th/21st/22nd);CountVector+SVD(10th);seq2seq(11th);LDA(28th);node2vec/deepwalk(7th/21st) | TFIDF-SVD ✔(v4);w2v/图嵌入未做 |
| month_lag 时序 | 逐月 pivot(10th);**阈值子集重算 +0.010**(22nd);month_rank(82375);未来月外推(10th) | pivot/斜率 ✔(v4);子集重算、外推未做 |
| 特征选择 | null importance(7th/16th);KS/对抗验证(10th/16th);Boruta(10th);RFE(55th);"Less is More"(73937) | 方差+重要性两道闸;分布检验类未做 |
| 融合规模 | 52 模型(7th)/82(10th)/32(11th)/70(21st);二层 **NN/Ridge 可行、LGB 过拟合**(7th) | 10+ 模型、二层 BayesianRidge ✔(与 7th/11th 结论一致) |
| 验证哲学 | 全员 "trust your CV";isotonic 校准 Private 变差(7th);LB probing 无增益(7th) | 纯 OOF 择优 + 私榜标尺 ✔;isotonic 证伪与 7th 互证 |

## 五、结论:频次统计与我们的行动含义

1. **出现频次最高的增量**(≥3 个方案):序列/嵌入特征(7)、大规模 stacking(6)、month_lag 时序深挖(4)、行级 meta model(3)、分布检验类特征选择(3)、离群软融合/分段(4);
2. **我们已覆盖**:软融合、month_lag pivot、TFIDF-SVD、BayesianRidge 二层、折内 TE、trust-CV 纪律——覆盖了高频项的约六成;
3. **明确的空白(按预期收益排序)**:
   - **行级 meta model**:1st/7th 两个头部方案的强特征,7th 量化 +0.005~0.006,19th 赛后验证 Private +0.003;实现成本中等,防泄漏要点明确(行级模型须折外)→ **建议列为下一实验最高优先级**;
   - **month_lag 阈值子集重算**(22nd,自报 +0.010):纯工程量,零新依赖;
   - w2v/node2vec 嵌入(7th/21st 均有代码可核);对抗验证/KS 特征筛选(10th/16th);
4. **多项独立互证增强了我们已有结论的置信度**:isotonic 有害(7th ↔ 我们 v5)、二层线性/贝叶斯优于 LGB(7th/11th/18th ↔ 我们 v5)、直接 TE 无效而防泄漏 TE 有效(19th/5th ↔ 我们 v6 与 TE-v2 事故)、不后处理常胜过硬后处理(16th/19th/28th ↔ 我们 v1 战役);
5. **差距归因维持原判**:头部在"特征通道数量 × 模型农场规模"上全面领先,单点 trick(1st 软融合)只解释一部分;我们以 10+ 模型达到 59 名等效,资源效率已高,继续逼近金牌线的合理路径是 §5.3 的空白清单而非扩农场。
