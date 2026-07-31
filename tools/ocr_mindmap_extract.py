# -*- coding: utf-8 -*-
"""思维导图 PDF → 层级 Markdown 提取工具。

原理:导图 PDF 无文字层,用 RapidOCR 切块识别取得每段文字的坐标;
"逻辑图"布局中节点 x 坐标随层级右移,据此做一维聚类重建层级树,
按 y 序深度优先输出为 Markdown(1-5 级用 #,更深用缩进列表)。

用法:
    python tools/ocr_mindmap_extract.py                # 全量 6 页,约 3-10 分钟
    python tools/ocr_mindmap_extract.py --sample       # 仅第 1 页左上角小样,快速验证
输出:
    与 PDF 同目录的 同名.md;中间结果 同名.nodes.json(调层级参数用)
可调参数见下方 CONFIG。若某页层级错乱,微调 LEVEL_GAP 后重跑即可。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")   # 本机小任务>16线程反而慢
import json
import sys

import fitz
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# ---------------- CONFIG ----------------
PDF = "/data/user/cwz/EloRecommender/花生十三言语思维导图完整版.pdf"
ZOOM = 1.5          # 渲染倍率;字号已够大,1.5x 足以喂 OCR
TILE = 1600         # 切块边长(渲染像素)
OVERLAP = 180       # 相邻块重叠,避免切断文字行
MIN_SCORE = 0.55    # OCR 置信度阈值
LEVEL_GAP = 55      # x 聚类间隔(渲染像素);层级缩进小于此值会被并为同级
LINE_MERGE = 0.6    # 同节点多行合并:行距 < 0.6*行高 且 x 对齐
# ----------------------------------------


def page_tiles(page):
    """渲染整页并产出 (x_off, y_off, ndarray) 切块,空白块跳过。"""
    pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    step = TILE - OVERLAP
    for y0 in range(0, img.shape[0], step):
        for x0 in range(0, img.shape[1], step):
            t = img[y0:y0 + TILE, x0:x0 + TILE]
            if t.size == 0 or t.min() > 235:          # 纯白块
                continue
            yield x0, y0, np.ascontiguousarray(t)


def ocr_page(ocr, page):
    """OCR 一页,返回 [{x,y,h,text,score}](全局坐标,已去重、合并多行)。"""
    raw = []
    for x0, y0, tile in page_tiles(page):
        res, _ = ocr(tile)
        for quad, text, score in (res or []):
            if float(score) < MIN_SCORE or not text.strip():
                continue
            xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
            raw.append(dict(x=x0 + min(xs), y=y0 + (min(ys) + max(ys)) / 2,
                            h=max(ys) - min(ys), text=text.strip(), score=float(score)))
    # 重叠区去重:文本相同且中心距近者留高分
    raw.sort(key=lambda r: -r["score"])
    kept = []
    for r in raw:
        dup = any(abs(r["x"] - k["x"]) < k["h"] and abs(r["y"] - k["y"]) < k["h"] * 0.8
                  and (r["text"] == k["text"] or r["text"] in k["text"] or k["text"] in r["text"])
                  for k in kept)
        if not dup:
            kept.append(r)
    # 同节点多行合并:x 左对齐、行距小
    kept.sort(key=lambda r: (round(r["x"] / 15), r["y"]))
    nodes = []
    for r in kept:
        p = nodes[-1] if nodes else None
        if p and abs(r["x"] - p["x"]) < 15 and 0 < r["y"] - p["y"] < (p["h"] + r["h"]) / 2 * (1 + LINE_MERGE):
            p["text"] += r["text"]
            p["y"] = (p["y"] + r["y"]) / 2
        else:
            nodes.append(dict(r))
    return nodes


def assign_levels(nodes):
    """按 x 左缘一维聚类 → 层级号。"""
    xs = sorted(n["x"] for n in nodes)
    bounds = [xs[0]]
    for a, b in zip(xs, xs[1:]):
        if b - a > LEVEL_GAP:
            bounds.append(b)
    for n in nodes:
        n["level"] = max(i for i, s in enumerate(bounds) if n["x"] >= s - LEVEL_GAP / 2)
    return len(bounds)


def build_tree(nodes):
    """父节点 = 上一层中 y 最近者(导图父节点垂直居中于子树带),再深度优先展平。"""
    by_level = {}
    for n in nodes:
        by_level.setdefault(n["level"], []).append(n)
    for n in nodes:
        n["children"] = []
    roots = sorted(by_level.get(0, []), key=lambda n: n["y"])
    for lv in sorted(by_level)[1:]:
        parents = by_level.get(lv - 1) or by_level[min(by_level)]
        for n in sorted(by_level[lv], key=lambda n: n["y"]):
            min(parents, key=lambda p: abs(p["y"] - n["y"]))["children"].append(n)
    out = []
    def dfs(n, d):
        out.append((d, n["text"]))
        for c in sorted(n["children"], key=lambda c: c["y"]):
            dfs(c, d + 1)
    for r in roots:
        dfs(r, 0)
    return out


def to_markdown(flat, page_no):
    lines = [f"\n<!-- ===== 第 {page_no} 页 ===== -->\n"]
    for depth, text in flat:
        if depth <= 4:
            lines.append("#" * (depth + 1) + " " + text)
        else:
            lines.append("  " * (depth - 5) + "- " + text)
    return "\n".join(lines) + "\n"


def main():
    sample = "--sample" in sys.argv
    doc = fitz.open(PDF)
    ocr = RapidOCR()
    md, dump = [], []
    pages = range(1) if sample else range(len(doc))
    for i in pages:
        page = doc[i]
        if sample:
            page.set_cropbox(fitz.Rect(0, 0, 1800, 1100))   # 左上角小样
        nodes = ocr_page(ocr, page)
        if not nodes:
            print(f"[{i+1}/{len(doc)}] 无识别结果", flush=True)
            continue
        n_lv = assign_levels(nodes)
        flat = build_tree(nodes)
        md.append(to_markdown(flat, i + 1))
        dump.append(dict(page=i + 1, levels=n_lv, nodes=[
            {k: n[k] for k in ("x", "y", "text", "level")} for n in nodes]))
        print(f"[{i+1}/{len(doc)}] 节点 {len(nodes)} 个,层级 {n_lv} 层", flush=True)
    if sample:
        print("\n----- 小样输出预览 -----")
        print("\n".join(md))
        return
    base = os.path.splitext(PDF)[0]
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(f"# 花生十三言语思维导图(OCR 提取)\n")
        f.write("".join(md))
    with open(base + ".nodes.json", "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=1)
    print(f"\n完成 → {base}.md(供个人笔记使用;注意勿随仓库提交/公开传播)")


if __name__ == "__main__":
    main()
