# -*- coding: utf-8 -*-
"""提交并自动回填 docs/submissions.md:python src/submit_and_log.py <csv路径> <备注>"""
import csv
import datetime
import io
import os
import subprocess
import sys
import time

COMP = "elo-merchant-category-recommendation"
f, msg = sys.argv[1], sys.argv[2]

ok = False
for i in range(3):  # 网络抖动重试
    r = subprocess.run(["kaggle", "competitions", "submit", "-c", COMP, "-f", f, "-m", msg],
                       capture_output=True, text=True)
    if "Successfully" in (r.stdout + r.stderr):
        ok = True
        break
    print(f"[submit] 第{i + 1}次失败: {r.stdout} {r.stderr}", flush=True)
    time.sleep(30)
if not ok:
    print("[submit] SUBMIT_FAILED", flush=True)
    sys.exit(1)

pub = priv = None
for _ in range(30):  # 轮询评分,最长约 10 分钟
    time.sleep(20)
    r = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v"],
                       capture_output=True, text=True)
    try:
        rows = list(csv.DictReader(io.StringIO(r.stdout)))
    except Exception:
        continue
    if rows and rows[0].get("fileName") == os.path.basename(f) \
            and rows[0].get("status", "").endswith("COMPLETE"):
        pub, priv = rows[0]["publicScore"], rows[0]["privateScore"]
        break

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
with open("docs/submissions.md", "a") as fp:
    fp.write(f"| {now} | {os.path.basename(f)} | {msg} | {pub} | {priv} |\n")
print(f"[submit] SCORED public={pub} private={priv} (已回填 docs/submissions.md)", flush=True)
