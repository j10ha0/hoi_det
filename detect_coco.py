"""detect_coco.py — 用标准视觉检测模型（COCO 预训练 YOLO）检测物体

不需要任何文本提示：模型直接输出 类别 + 框 + 置信度
COCO 80 类里包含: apple, banana, orange, carrot, bottle, cup, bowl,
                  sports ball(网球), wine glass, dining table, person ...
"""
import glob
import os

import cv2
import numpy as np
import torch

BASE = "/mnt/storage/siat_cjh/proj/hoi_data/egotactile"


def main():
    from ultralytics import YOLO
    model = YOLO("yolo11s.pt")          # COCO 预训练，自动下载
    print("模型: yolo11s.pt (COCO 80 类)\n")

    frames = sorted(glob.glob(f"{BASE}/nc_frames4/*.jpg"))
    print(f"{'帧/物体':<34} {'检测到的物体 (类别 置信 框)':<70}")
    for f in frames:
        obj = os.path.basename(f).replace("_nc.jpg", "")
        img = cv2.imread(f)
        res = model.predict(img, conf=0.15, iou=0.5, verbose=False)[0]
        names = res.names
        if res.boxes is None or len(res.boxes) == 0:
            print(f"{obj:<34} ✗ 无检测")
            continue
        dets = []
        for i in range(len(res.boxes)):
            c = float(res.boxes.conf[i]); k = int(res.boxes.cls[i])
            xy = res.boxes.xyxy[i].cpu().numpy()
            dets.append((c, names[k], xy))
        dets.sort(key=lambda z: -z[0])
        # 只显示非 person/table 的，按置信排序
        show = [d for d in dets if d[1] not in ("person", "dining table", "chair")]
        s = "  ".join(f"{n}({c:.2f})[{x[0]:.0f},{x[1]:.0f},{x[2]:.0f},{x[3]:.0f}]"
                      for c, n, x in show[:3])
        print(f"{obj:<34} {s if s else '(只有 person/table)'}")


if __name__ == "__main__":
    main()
