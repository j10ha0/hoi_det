"""find_clean_demo.py — 从多帧里自动挑出「干净演示帧」

好的演示帧应该满足：
  ① 手部检得到（WiLoR）
  ② 找到的物体【与手几乎不重叠】（物体没被遮挡）
  ③ 物体方向与掌心法向对齐好（角度小）
  ④ 物体面积/形状合理（紧凑、不太小）
  ⑤ 只有一只手在画面中（避免另一只手干扰）

输出：按分数排序的前 N 帧 + 可视化
"""
import glob
import os

import cv2
import numpy as np
import torch

for _n, _t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("unicode", str), ("str", str)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)
np.nan = float("nan"); np.inf = float("inf")

FRAMES = "/mnt/storage/siat_cjh/proj/hoi_data/scan_frames"
OUT = "/mnt/storage/siat_cjh/proj/hoi_data"
ANG_MAX = 60.0


def shape_features(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 300:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    X = pts - c
    w, v = np.linalg.eigh(X.T @ X / len(X))
    sv = np.sqrt(np.maximum(w, 0))
    evr = float(w[0] / (w[1] + 1e-9))
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    a = float(cv2.contourArea(cnt))
    per = float(cv2.arcLength(cnt, True)) + 1e-6
    circ = 4 * np.pi * a / per ** 2
    sol = a / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-6)
    x, y, bw, bh = cv2.boundingRect(cnt)
    return dict(center=c, major=v[:, 1], sv=sv, evr=evr, area=int(mask.sum()),
                circularity=circ, solidity=sol, extent=a / (bw * bh + 1e-6),
                bbox=(x, y, bw, bh))


def classify(f):
    if f["evr"] > 0.55:
        if f["circularity"] > 0.80:
            return "球/圆盘", (0, 165, 255)
        if f["extent"] > 0.85:
            return "方块/矩形", (255, 128, 0)
        return "不规则", (128, 128, 128)
    return ("细长(圆柱/长条)", (255, 255, 0)) if f["evr"] < 0.15 \
        else ("中等长宽比(棱柱/盒)", (0, 255, 0))


def main():
    from ultralytics import FastSAM
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float32, verbose=False)
    fsam = FastSAM("FastSAM-s.pt")

    results = []
    for img_path in sorted(glob.glob(f"{FRAMES}/s*.jpg")):
        tag = os.path.basename(img_path)[:-4]
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        out = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not out:
            print(f"{tag}: ✗ 未检出手"); continue
        nh = len(out)
        o = max(out, key=lambda z: (z["hand_bbox"][2] - z["hand_bbox"][0]) *
                                   (z["hand_bbox"][3] - z["hand_bbox"][1]))
        hb = o["hand_bbox"]
        kp2d = np.asarray(o["wilor_preds"]["pred_keypoints_2d"]).reshape(-1, 2)
        kp3d = np.asarray(o["wilor_preds"]["pred_keypoints_3d"]).reshape(-1, 3)
        palm = (kp2d[0] + kp2d[5] + kp2d[17]) / 3
        n3 = np.cross(kp3d[5] - kp3d[0], kp3d[17] - kp3d[0])
        n3 = n3 / (np.linalg.norm(n3) + 1e-9)
        if int(o.get("is_right", 1)) == 0:
            n3 = -n3

        res = fsam(img, device="cuda:0", retina_masks=True, imgsz=1024,
                   conf=0.4, iou=0.9, verbose=False)[0]
        if res.masks is None:
            print(f"{tag}: ✗ 无分割"); continue
        masks = res.masks.data.cpu().numpy()
        hand_box = np.array(hb, dtype=float)
        hand_r = 0.5 * max(hand_box[2] - hand_box[0], hand_box[3] - hand_box[1])
        border = np.zeros((H, W), bool)
        border[:8, :] = border[-8:, :] = border[:, :8] = border[:, -8:] = True

        n2 = np.array([n3[0], n3[1]], float)
        n2 /= (np.linalg.norm(n2) + 1e-9)
        best = None
        for sgn, nn in [("+", n2), ("-", -n2)]:
            for m in masks:
                mb = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                area = int(mb.sum())
                if area < 5000 or area > 0.20 * W * H:
                    continue
                if (mb & border).sum() > 120:
                    continue
                f = shape_features(mb)
                if f is None:
                    continue
                x, y, bw, bh = f["bbox"]
                inter = max(0, min(x + bw, hand_box[2]) - max(x, hand_box[0])) * \
                        max(0, min(y + bh, hand_box[3]) - max(y, hand_box[1]))
                ov = inter / (bw * bh + 1e-6)           # ⭐ 与手重叠比（遮挡程度）
                d = f["center"] - palm
                dn = float(np.linalg.norm(d))
                if dn < 0.7 * hand_r:
                    continue
                v = d / (dn + 1e-9)
                ang = float(np.degrees(np.arccos(np.clip(np.dot(v, nn), -1, 1))))
                if ang > ANG_MAX:
                    continue
                # 评分：角度小、重叠低、面积适中（偏好 8k~60k）、只手
                s = (1 - ang / ANG_MAX) * 2.0 + (1 - ov) * 3.0 \
                    + (1.0 if 8000 < f["area"] < 60000 else 0.3) \
                    + (0.6 if nh == 1 else 0.0)
                if best is None or s > best[0]:
                    best = (s, ang, dn, ov, f, mb, nh)
        if best is None:
            print(f"{tag}: ✗ 无合格候选（手数{nh}）"); continue
        s, ang, dn, ov, f, mb, nh = best
        lb, _ = classify(f)
        print(f"{tag}: ✅ 分数{s:.2f} 手数{nh} 夹角{ang:5.1f}° 距离{dn:5.0f}px "
              f"手重叠{ov*100:4.0f}% 面积{f['area']:>6} → {lb}")
        results.append((s, tag, img, f, mb, ang, dn, ov, nh, n2, palm))

    results.sort(key=lambda z: -z[0])
    print(f"\n=== 最佳 {min(3,len(results))} 帧 ===")
    for s, tag, img, f, mb, ang, dn, ov, nh, n2, palm in results[:3]:
        lb, color = classify(f)
        vis = img.copy()
        vis[mb] = (vis[mb] * 0.45 + np.array(color, np.uint8) * 0.55).astype(np.uint8)
        x, y, bw, bh = f["bbox"]
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 4)
        c, major = f["center"], f["major"]
        L = 0.55 * max(bw, bh)
        cv2.line(vis, (int(c[0] + major[0] * L), int(c[1] + major[1] * L)),
                 (int(c[0] - major[0] * L), int(c[1] - major[1] * L)), (255, 255, 255), 6)
        perp = np.array([-major[1], major[0]])
        if np.dot(perp, palm - c) < 0:
            perp = -perp
        cv2.arrowedLine(vis, tuple(c.astype(int)),
                        (int(c[0] + perp[0] * L * 0.9), int(c[1] + perp[1] * L * 0.9)),
                        (255, 0, 255), 6, tipLength=0.25)
        cv2.arrowedLine(vis, tuple(palm.astype(int)),
                        (int(palm[0] + n2[0] * 220), int(palm[1] + n2[1] * 220)),
                        (0, 0, 255), 6, tipLength=0.2)
        cv2.circle(vis, tuple(palm.astype(int)), 10, (0, 0, 255), -1)
        txt = f"{tag}  {lb}  ang={ang:.0f}deg  d={dn:.0f}px  handOv={ov*100:.0f}%"
        cv2.putText(vis, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
        cv2.putText(vis, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        p = f"{OUT}/demo_best_{tag}.jpg"
        cv2.imwrite(p, vis)
        print(f"  已保存 {p}   (分数{s:.2f} 手数{nh})")


if __name__ == "__main__":
    main()
