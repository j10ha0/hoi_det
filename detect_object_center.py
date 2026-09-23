"""detect_object_center.py — 用 YOLO-World（开放词汇检测）检测物体中心

思路:
  EgoTactile 的目录名就是物体名（免费真值）→ 直接作为【文本提示】
  → YOLO-World 零样本检测该物体 → 取 bbox 中心 = 物体中心

对比旧方案（绿幕连通域）:
  · 旧: 前景 - 手 → 连通域 → 取最近 → 质心   ← 实测失败（手掩膜吞掉物体）
  · 新: 文本提示 → 检测框 → 框中心            ← 直接、可靠
"""
import glob
import json
import os

import cv2
import numpy as np
import torch

for _n, _t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("unicode", str), ("str", str)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)
np.nan = float("nan"); np.inf = float("inf")

BASE = "/mnt/storage/siat_cjh/proj/hoi_data/egotactile"

# 物体名 → 英文提示词（YOLO-World 用）
PROMPT = {
    "Apple": "apple", "Banana": "banana", "BellPepper": "bell pepper",
    "Carrot": "carrot", "Clip": "clip", "Corn": "corn", "Cucumber": "cucumber",
    "Dumbbell": "dumbbell", "Orange": "orange", "TennisBall": "tennis ball",
    "Tomato": "tomato",
    "CestbonWater-555ml": "water bottle", "GantenWater-560ml": "water bottle",
    "NongfuSpringWater-550ml-Green": "water bottle",
    "NongfuSpringWater-550ml-Red": "water bottle", "Sprite-500ml": "soda bottle",
    "CocaCola-330ml": "soda can", "CocaCola-500ml": "soda bottle",
    "Pepsi-330ml": "soda can", "WatsonsSodaWater-330ml-Black": "soda can",
    "WatsonsSodaWater-330ml-Red": "soda can",
    "LaysChips-Tube-CucumberFlavor": "chips can",
    "LaysChips-Tube-OriginalFlavor": "chips can",
    "SamyangSpicyChickenCupNoodles": "cup noodles",
    "ShinRamyunCupNoodles": "cup noodles",
}


def main():
    from ultralytics import YOLOWorld
    frames = sorted(glob.glob(f"{BASE}/nc_frames4/*.jpg"))
    print(f"测试 {len(frames)} 张「未接触」帧\n")
    model = YOLOWorld("yolov8s-worldv2.pt")
    ok = 0
    for f in frames:
        obj = os.path.basename(f).replace("_nc.jpg", "")
        prompt = PROMPT.get(obj, obj.lower())
        model.set_classes([prompt])
        img = cv2.imread(f)
        res = model.predict(img, conf=0.05, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            print(f"{obj:<34} 提示[{prompt:<14}] ✗ 未检出")
            continue
        b = res.boxes
        conf = float(b.conf.max())
        i = int(b.conf.argmax())
        xyxy = b.xyxy[i].cpu().numpy()
        cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
        ok += 1
        print(f"{obj:<34} 提示[{prompt:<14}] ✅ 置信 {conf:.2f}  "
              f"bbox=({xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f})  "
              f"中心=({cx:.0f},{cy:.0f})  w={xyxy[2]-xyxy[0]:.0f} h={xyxy[3]-xyxy[1]:.0f}")
    print(f"\n检出率: {ok}/{len(frames)}")


if __name__ == "__main__":
    main()
