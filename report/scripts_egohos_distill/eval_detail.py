"""eval_detail.py — 把 hand F1 这个数字拆开看（避免过度解读）

回答: 0.7832 到底是什么？
  它 = 「手 vs 背景」二值像素 F1（左右手合并，全验证集像素全局累计）

本脚本额外给出:
  1. 复核二值 hand F1（应与训练日志一致）
  2. 3 类混淆矩阵（背景/左手/右手）→ 看模型是否分得开左右手
  3. 每类 IoU / 精确率 / 召回率
  4. 帧级指标：有多少帧"完全漏检手"（预测全背景）
  5. 区域质量：预测的手区域是否连贯（连通域数）

用法: python eval_detail.py --ckpt runs_v5/C4_l0.5/best.pt --data data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_student import SegData, Student


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="val")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    size = tuple(a.get("size", [224, 224]))
    t_keys = (a.get("teacher_keys") or [a.get("teacher_key", "stage3")]) if a.get("distill_feat") else []
    net = Student(a["backbone"], use_aux=a["use_aux_heads"], size=size,
                  distill=a.get("distill_feat", False),
                  t_ch={k: {"stage2": 256, "stage3": 512, "stage4": 1024}[k] for k in t_keys})
    net.load_state_dict(ck["model"])
    net.to(dev).eval()
    print(f"checkpoint: {args.ckpt}")
    print(f"骨架 {a['backbone']} | aux={a['use_aux_heads']} | distill={a.get('distill_feat')} "
          f"| keys={t_keys} | λ_feat={a.get('lambda_feat')} | size={size}")
    print(f"训练时记录的最佳 hand F1 = {ck.get('metrics', {}).get('hand', {}).get('f1', '?')}\n")

    ds = SegData(Path(args.data), args.split, size, aug=False)
    ld = torch.utils.data.DataLoader(ds, args.bs, shuffle=False, num_workers=2)

    # 全局像素累计
    cm = np.zeros((3, 3), dtype=np.int64)          # 行=GT, 列=Pred
    tp = fp = fn = tn = 0
    n_frames = 0
    n_frames_total_miss = 0                        # 预测"全背景"但 GT 有手
    n_frames_empty_pred_with_hand = 0
    n_frames_left_pred = 0
    n_frames_right_pred = 0
    conn_counts = []
    for x, yh, yc, yo, _ in ld:
        x = x.to(dev)
        out = net(x)
        ph = out["hand"].argmax(1).cpu().numpy()
        gh = yh.numpy()
        # 二值 F1
        pred = ph > 0
        gt = gh > 0
        tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum()); tn += int((~pred & ~gt).sum())
        # 3x3 混淆
        for b in range(gh.shape[0]):
            g, p = gh[b].ravel(), ph[b].ravel()
            idx = g * 3 + p
            cm += np.bincount(idx, minlength=9).reshape(3, 3)
            n_frames += 1
            if gt[b].sum() > 0 and pred[b].sum() == 0:
                n_frames_total_miss += 1
            if (ph[b] == 1).sum() > 0:
                n_frames_left_pred += 1
            if (ph[b] == 2).sum() > 0:
                n_frames_right_pred += 1
        # 连通域（用简单上下采样近似：统计每帧预测手像素的 8x8 块数）
        for b in range(ph.shape[0]):
            m = (ph[b] > 0).astype(np.uint8)
            blk = m[: m.shape[0] // 8 * 8, : m.shape[1] // 8 * 8].reshape(
                m.shape[0] // 8, 8, m.shape[1] // 8, 8).max(axis=(1, 3))
            conn_counts.append(int(blk.sum()))

    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    iou = tp / max(1, tp + fp + fn)

    print("=" * 62)
    print("① 二值「手 vs 背景」像素级指标（= 训练日志里那个 hand F1）")
    print(f"   F1={f1:.4f}  IoU={iou:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")
    print(f"   像素计数 TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}")

    print("\n② 3 类混淆矩阵（行=真值, 列=预测; 单位=万像素）")
    names = ["背景", "左手", "右手"]
    print("            " + "".join(f"{n:>10}" for n in names))
    for i, n in enumerate(names):
        print(f"  真值{n:<4}" + "".join(f"{cm[i, j]/1e4:>10.1f}" for j in range(3)))

    print("\n③ 每类 IoU / 精确率 / 召回率（像素级）")
    for c, n in enumerate(names):
        t = cm[c, c]
        p = cm[:, c].sum()      # 预测为该类的总数
        g = cm[c, :].sum()      # 真值为该类的总数
        pr = t / max(1, p); rc = t / max(1, g)
        ii = t / max(1, p + g - t)
        print(f"   {n:<4} IoU={ii:.4f} P={pr:.4f} R={rc:.4f}  (真值像素 {g/1e4:.1f}万)")
    acc = np.trace(cm) / max(1, cm.sum())
    print(f"   3 类总体像素准确率 = {acc:.4f}")

    print("\n④ 左右手是否分得开")
    left_as_right = cm[1, 2] / max(1, cm[1].sum())
    right_as_left = cm[2, 1] / max(1, cm[2].sum())
    left_rec = cm[1, 1] / max(1, cm[1].sum())
    right_rec = cm[2, 2] / max(1, cm[2].sum())
    print(f"   左手召回={left_rec:.4f}  右手召回={right_rec:.4f}")
    print(f"   左手被误判为右手比例={left_as_right:.4f}   右手被误判为左手比例={right_as_left:.4f}")
    print(f"   预测含左手的帧数={n_frames_left_pred}/{n_frames}  含右手={n_frames_right_pred}/{n_frames}")

    print("\n⑤ 帧级失败模式")
    print(f"   真值有手但预测全背景（完全漏检）的帧: {n_frames_total_miss}/{n_frames} "
          f"({100*n_frames_total_miss/max(1,n_frames):.1f}%)")
    print(f"   预测手区域的 8×8 块数（均值，越大说明越碎/越散）: "
          f"{np.mean(conn_counts):.1f} ± {np.std(conn_counts):.1f}")


if __name__ == "__main__":
    main()
