#!/bin/bash
# v4 冲高战役:开源招数移植特征 + huber 第4基模型 + 3-seed 平均(nohup+setsid 脱离会话)
# 日志 outputs/logs/campaign2.log;各阶段自动提交并回填 docs/submissions.md
cd /data/user/cwz/EloRecommender || exit 1
PY=/data/guest/anaconda3/bin/python
exec >> outputs/logs/campaign2.log 2>&1

echo "===== CAMPAIGN2 START $(date '+%F %T') ====="

echo "----- [A] v4 特征重建 + 5套CV(seed 2019)-----"
rm -f data/processed/features.parquet
$PY src/elo_pipeline.py || { echo "A_FAILED $(date '+%F %T')"; exit 1; }
$PY src/submit_and_log.py outputs/submission_stack.csv "v4 opensource-port feats, 4-model stack"
if $PY src/exp1_clean_postprocess.py; then
    $PY src/submit_and_log.py outputs/submission_clean_pp.csv "v4 $(cat outputs/exp1_desc.txt)"
fi
mkdir -p outputs/seed2019
cp outputs/{oof_predictions.csv,test_clf_prob.csv,submission_stack.csv,submission_clean_pp.csv} outputs/seed2019/

for S in 42 777; do
    echo "----- [B] seed $S 全套复训 -----"
    ELO_SEED=$S $PY src/elo_pipeline.py || { echo "B${S}_FAILED"; continue; }
    ELO_SEED=$S $PY src/exp1_clean_postprocess.py || echo "B${S}_pp_FAILED"
    mkdir -p outputs/seed$S
    cp outputs/{oof_predictions.csv,test_clf_prob.csv,submission_stack.csv,submission_clean_pp.csv} outputs/seed$S/
done

echo "----- [C] 3-seed 平均与提交 -----"
if $PY src/avg_seeds.py; then
    $PY src/submit_and_log.py outputs/submission_seedavg_clean_pp.csv "v4 3-seed avg clean+pp"
    $PY src/submit_and_log.py outputs/submission_seedavg_stack.csv "v4 3-seed avg stack"
fi

echo "===== CAMPAIGN2 DONE $(date '+%F %T') ====="
