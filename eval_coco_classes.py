"""eval_coco_classes.py — 逐类评估：COCO 检测器能否说出「具体是什么」

对每个物体（用 2-3 帧），列出检测器输出的类别+置信度，
并与"期望的 COCO 类别"对照。
"""
import glob
import os
from collections import defaultdict

import cv2
import numpy as np

BASE = "/mnt/storage/siat_cjh/proj/hoi_data/egotactile"

# 期望的 COCO 类别（None = COCO 里根本没有这个类别）
EXPECT = {
    "Apple": "apple", "Banana": "banana", "Carrot": "carrot", "Orange": "orange",
    "Tomato": None,            # COCO 无 tomato
    "TennisBall": "sports ball",
    "CestbonWater-555ml": "bottle", "Sprite-500ml": "bottle",
    "CocaCola-500ml": "bottle",
    "CocaCola-330ml": None,    # COCO 无 can
    "Pepsi-330ml": None,
    "SamyangSpicyChickenCupNoodles": "cup",
    "LaysChips-Tube-OriginalFlavor": None,
    "Dumbbell": None,
    "Clip": None,
}


def main():
    from ultralytics import YOLO
    model = YOLO("yolo11s.pt")
    SKIP = {"person", "dining table", "chair"}

    by_obj = defaultdict(list)
    for f in sorted(glob.glob(f"{BASE}/frames/*.jpg")):
        obj = os.path.basename(f).split("_p")[0]
        if obj not in EXPECT:
            continue
        img = cv2.imread(f)
        r = model.predict(img, conf=0.15, iou=0.5, verbose=False)[0]
        dets = []
        if r.boxes is not None:
            for i in range(len(r.boxes)):
                c = float(r.boxes.conf[i]); k = int(r.boxes.cls[i]); nm = r.names[k]
                if nm in SKIP:
                    continue
                xy = r.boxes.xyxy[i].cpu().numpy()
                dets.append((c, nm, (xy[2] - xy[0]) * (xy[3] - xy[1])))
        dets.sort(key=lambda z: -z[2])
        by_obj[obj].append(dets[:3])

    print(f"{'物体':<34} {'期望类别':<12} {'检测输出（按框大小排序）':<52} 结论")
    n_ok = n_partial = n_fail = 0
    for obj, exp in EXPECT.items():
        if obj not in by_obj:
            print(f"{obj:<34} {str(exp):<12} (无帧)"); continue
        allnames = []
        for ds in by_obj[obj]:
            allnames += [n for _, n, _ in ds]
        top = []
        for ds in by_obj[obj]:
            top.append(", ".join(f"{n}({c:.2f})" for c, n, _ in ds) or "—")
        if exp is None:
            verdict = "COCO无此类 → 只能给近似类"
            n_partial += 1
        elif any(n == exp for n in allnames):
            verdict = "✅ 说出正确类别"
            n_ok += 1
        elif allnames:
            verdict = "⚠️ 类别错误"
            n_fail += 1
        else:
            verdict = "❌ 未检出"
            n_fail += 1
        print(f"{obj:<34} {str(exp):<12} {' | '.join(t[:50] for t in top):<52} {verdict}")

    print(f"\n汇总: 类别正确 {n_ok} | 只能近似/错误 {n_partial + n_fail} | 共 {len(EXPECT)} 类")
    print("COCO 有对应类别的: apple, banana, carrot, orange, sports ball, bottle, cup")


if __name__ == "__main__":
    main()
