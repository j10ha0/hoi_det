"""egotactile_demo.py — 在 EgoTactile（绿幕 ego 数据）上跑「最近物体 位置+shape」

优势：绿幕背景干净、单物体、物体名在文件名里（免费真值）→ 可定量验证
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

FRAMES = "/mnt/storage/siat_cjh/proj/hoi_data/egotactile/frames"
OUT = "/mnt/storage/siat_cjh/proj/hoi_data/egotactile"

# 真值（人工给出，用于对比判定结果）
GT = {"CocaCola-330ml": "圆柱(罐)", "Orange": "球", "Dumbbell": "细长(哑铃)",
      "Clip": "小物体(夹子)"}


def feats(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 300:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0); X = pts - c
    w, v = np.linalg.eigh(X.T @ X / len(X))
    sv = np.sqrt(np.maximum(w, 0)); evr = float(w[0] / (w[1] + 1e-9))
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    a = float(cv2.contourArea(cnt)); per = float(cv2.arcLength(cnt, True)) + 1e-6
    circ = 4 * np.pi * a / per ** 2
    sol = a / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-6)
    x, y, bw, bh = cv2.boundingRect(cnt)
    return dict(center=c, major=v[:, 1], sv=sv, evr=evr, area=int(mask.sum()),
                circularity=circ, solidity=sol, extent=a / (bw * bh + 1e-6),
                bbox=(x, y, bw, bh), aspect=max(bw, bh) / (min(bw, bh) + 1e-6))


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

    rows = []
    for img_path in sorted(glob.glob(f"{FRAMES}/*.jpg")):
        base = os.path.basename(img_path)[:-4]
        obj_gt = base.rsplit("_t", 1)[0]
        gt = GT.get(obj_gt, "?")
        img = cv2.imread(img_path)
        H, W = img.shape[:2]

        out = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not out:
            print(f"{base}: ✗ WiLoR 未检出手"); continue
        o = max(out, key=lambda z: (z["hand_bbox"][2] - z["hand_bbox"][0]) *
                                   (z["hand_bbox"][3] - z["hand_bbox"][1]))
        hb = o["hand_bbox"]
        kp2d = np.asarray(o["wilor_preds"]["pred_keypoints_2d"]).reshape(-1, 2)
        kp3d = np.asarray(o["wilor_preds"]["pred_keypoints_3d"]).reshape(-1, 3)
        palm = (kp2d[0] + kp2d[5] + kp2d[17]) / 3
        n3 = np.cross(kp3d[5] - kp3d[0], kp3d[17] - kp3d[0]); n3 /= np.linalg.norm(n3) + 1e-9

        res = fsam(img, device="cuda:0", retina_masks=True, imgsz=1024,
                   conf=0.4, iou=0.9, verbose=False)[0]
        if res.masks is None:
            print(f"{base}: ✗ 无分割"); continue
        masks = res.masks.data.cpu().numpy()
        hbox = np.array(hb, float); hr = 0.5 * max(hbox[2] - hbox[0], hbox[3] - hbox[1])
        border = np.zeros((H, W), bool)
        border[:8, :] = border[-8:, :] = border[:, :8] = border[:, -8:] = True
        n2 = np.array([n3[0], n3[1]], float); n2 /= (np.linalg.norm(n2) + 1e-9)

        best = None
        for sgn, nn in [("+", n2), ("-", -n2)]:
            for m in masks:
                mb = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                area = int(mb.sum())
                if area < 3000 or area > 0.35 * W * H:
                    continue
                if (mb & border).sum() > 150:
                    continue
                f = feats(mb)
                if f is None:
                    continue
                x, y, bw, bh = f["bbox"]
                inter = max(0, min(x + bw, hbox[2]) - max(x, hbox[0])) * \
                        max(0, min(y + bh, hbox[3]) - max(y, hbox[1]))
                ov = inter / (bw * bh + 1e-6)
                d = f["center"] - palm; dn = float(np.linalg.norm(d))
                if dn < 0.7 * hr:
                    continue
                v = d / (dn + 1e-9)
                ang = float(np.degrees(np.arccos(np.clip(np.dot(v, nn), -1, 1))))
                if ang > 60:
                    continue
                sc = (1 - ov) * 3 + (1 - ang / 60) * 2
                if best is None or sc > best[0]:
                    best = (sc, ang, dn, ov, f, mb, sgn)
        if best is None:
            print(f"{base}: ✗ 无合格候选"); continue
        sc, ang, dn, ov, f, mb, sgn = best
        lb, color = classify(f)
        ok = "✓" if ((("圆柱" in gt) and ("细长" in lb or "中等" in lb)) or
                     (("球" in gt) and ("球" in lb)) or
                     (("细长" in gt) and ("细长" in lb)) or
                     (("小物体" in gt))) else "?"
        print(f"{base:<22} 真值[{gt:<10}] → 判定[{lb:<16}] {ok}  "
              f"夹角{ang:5.1f}° 距离{dn:5.0f}px 重叠{ov*100:3.0f}% 长宽比{f['aspect']:.2f} "
              f"特征值比{f['evr']:.2f} 圆度{f['circularity']:.2f}")
        rows.append((base, gt, lb, ok, ang, dn, ov))

        vis = img.copy()
        vis[mb] = (vis[mb] * 0.45 + np.array(color, np.uint8) * 0.55).astype(np.uint8)
        x, y, bw, bh = f["bbox"]
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 4)
        c, major = f["center"], f["major"]; L = 0.55 * max(bw, bh)
        cv2.line(vis, (int(c[0] + major[0] * L), int(c[1] + major[1] * L)),
                 (int(c[0] - major[0] * L), int(c[1] - major[1] * L)), (255, 255, 255), 6)
        perp = np.array([-major[1], major[0]])
        if np.dot(perp, palm - c) < 0: perp = -perp
        cv2.arrowedLine(vis, tuple(c.astype(int)),
                        (int(c[0] + perp[0] * L * .9), int(c[1] + perp[1] * L * .9)),
                        (255, 0, 255), 6, tipLength=.25)
        cv2.arrowedLine(vis, tuple(palm.astype(int)),
                        (int(palm[0] + n2[0] * 200), int(palm[1] + n2[1] * 200)),
                        (0, 0, 255), 6, tipLength=.2)
        cv2.circle(vis, tuple(palm.astype(int)), 10, (0, 0, 255), -1)
        t = f"GT[{gt}]  pred[{lb}]  ang={ang:.0f} d={dn:.0f}px ov={ov*100:.0f}%"
        cv2.putText(vis, t, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(vis, t, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imwrite(f"{OUT}/vis_{base}.jpg", vis)

    print(f"\n=== 汇总 ({len(rows)} 帧) ===")
    for r in rows:
        print(f"  {r[0]:<22} 真值 {r[1]:<10} 判定 {r[2]:<16} {r[3]}")


if __name__ == "__main__":
    main()
