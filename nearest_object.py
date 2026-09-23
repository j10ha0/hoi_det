"""nearest_object.py — 检测「离手最近的那个物体」的位置与 shape

场景：DexYCB（干净单物体桌面场景，比 EgoDex 整洁）

流水线:
  1. WiLoR → 手部位置（掌心中心 + 手 bbox）
  2. FastSAM 全图分割 → 候选区域
  3. 过滤：去掉太小/太大的、去掉与手重叠的
  4. ⭐ 选【中心离掌心最近】的那个 = 目标物体
  5. 对目标掩膜做 PCA + 形状特征 → 判断 shape
  6. 可视化：掩膜 + bbox + 主轴 + 抓握方向（垂直于主轴）+ shape 标签
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

FRAMES = "/mnt/storage/siat_cjh/proj/hoi_data/dexycb_seq"
OUT = "/mnt/storage/siat_cjh/proj/hoi_data"


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
    area = float(cv2.contourArea(cnt))
    per = float(cv2.arcLength(cnt, True)) + 1e-6
    circ = 4 * np.pi * area / (per ** 2)
    hull = cv2.convexHull(cnt)
    solidity = area / (cv2.contourArea(hull) + 1e-6)
    x, y, bw, bh = cv2.boundingRect(cnt)
    extent = area / (bw * bh + 1e-6)
    return dict(center=c, major=v[:, 1], sv=sv, evr=evr, area=int(mask.sum()),
                circularity=circ, solidity=solidity, extent=extent,
                bbox=(x, y, bw, bh))


def classify(f):
    """用几何特征粗分 shape"""
    if f["evr"] > 0.55:
        if f["circularity"] > 0.80:
            return "球 / 圆盘", (0, 165, 255)
        if f["extent"] > 0.85:
            return "方块 / 矩形（俯视）", (255, 128, 0)
        return "不规则 / 团块", (128, 128, 128)
    else:
        if f["evr"] < 0.15:
            return "细长（圆柱 / 长条）", (255, 255, 0)
        return "中等长宽比（棱柱 / 盒）", (0, 255, 0)


def main():
    from ultralytics import FastSAM
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float32, verbose=False)
    fsam = FastSAM("FastSAM-s.pt")

    imgs = sorted(glob.glob(f"{FRAMES}/c*.jpg"))
    picks = [imgs[15], imgs[20], imgs[45], imgs[70]]
    MIN_AREA = 1500          # 最小面积（640x480 尺度）

    for img_path in picks:
        tag = os.path.basename(img_path)[:-4]
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        vis = img.copy()
        print(f"\n===== {tag}  ({W}x{H}) =====")

        # ① 手
        out = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not out:
            print("  未检测到手"); continue
        o = max(out, key=lambda z: (z["hand_bbox"][2] - z["hand_bbox"][0]) *
                                   (z["hand_bbox"][3] - z["hand_bbox"][1]))
        hb = o["hand_bbox"]
        kp2d = np.asarray(o["wilor_preds"]["pred_keypoints_2d"]).reshape(-1, 2)
        palm = ((kp2d[0] + kp2d[5] + kp2d[17]) / 3)
        print(f"  手: bbox={[int(v) for v in hb]}  掌心=({palm[0]:.0f},{palm[1]:.0f})")
        cv2.rectangle(vis, (int(hb[0]), int(hb[1])), (int(hb[2]), int(hb[3])), (0, 0, 255), 2)
        cv2.circle(vis, tuple(palm.astype(int)), 7, (0, 0, 255), -1)

        # ② 分割
        res = fsam(img, device="cuda:0", retina_masks=True, imgsz=1024,
                   conf=0.4, iou=0.9, verbose=False)[0]
        if res.masks is None:
            print("  未分割出区域"); continue
        masks = res.masks.data.cpu().numpy()
        print(f"  FastSAM 候选区域: {len(masks)}")

        # ③④ 过滤 + 选离手最近
        hand_box = np.array([hb[0], hb[1], hb[2], hb[3]], dtype=float)
        hand_r = 0.5 * max(hand_box[2] - hand_box[0], hand_box[3] - hand_box[1])
        best, best_d, cands = None, 1e18, []
        for m in masks:
            mb = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            area = int(mb.sum())
            if area < MIN_AREA or area > 0.35 * W * H:
                continue
            f = shape_features(mb)
            if f is None:
                continue
            x, y, bw, bh = f["bbox"]
            inter = max(0, min(x + bw, hand_box[2]) - max(x, hand_box[0])) * \
                    max(0, min(y + bh, hand_box[3]) - max(y, hand_box[1]))
            ov = inter / (bw * bh + 1e-6)
            if ov > 0.30:                      # 与手重叠过多 → 是手/手指
                continue
            d = float(np.linalg.norm(f["center"] - palm))
            if d < 0.6 * hand_r:
                continue                       # 太贴手 → 疑似手指碎块
            cands.append((d, f, mb))
        cands.sort(key=lambda z: z[0])
        for i, (d, f, _) in enumerate(cands[:3]):
            lb, _ = classify(f)
            print(f"    候选{i+1}: 距离 {d:6.0f}px 面积 {f['area']:>6} 特征值比 {f['evr']:.2f} "
                  f"→ {lb}")
        if not cands:
            print("  未找到合适物体"); continue
        best_d, (f, mb) = cands[0][0], (cands[0][1], cands[0][2])
        label, color = classify(f)
        print(f"  ⭐ 离手最近的物体: 距离 {best_d:.0f}px  面积 {f['area']}  "
              f"特征值比 {f['evr']:.2f}  圆度 {f['circularity']:.2f}  充实度 {f['solidity']:.2f}")
        print(f"     → shape 判定: 【{label}】")
        print(f"     → 位置(bbox): {f['bbox']}   中心: ({f['center'][0]:.0f},{f['center'][1]:.0f})")

        # ⑥ 可视化
        vis[mb] = (vis[mb] * 0.5 + np.array(color, dtype=np.uint8) * 0.5).astype(np.uint8)
        x, y, bw, bh = f["bbox"]
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 3)
        c, major = f["center"], f["major"]
        L = 0.55 * max(bw, bh)
        cv2.line(vis, (int(c[0] + major[0] * L), int(c[1] + major[1] * L)),
                 (int(c[0] - major[0] * L), int(c[1] - major[1] * L)), (255, 255, 255), 4)
        # 抓握方向（垂直于主轴，指向手）
        perp = np.array([-major[1], major[0]])
        if np.dot(perp, palm - c) < 0:
            perp = -perp
        cv2.arrowedLine(vis, tuple(c.astype(int)),
                        (int(c[0] + perp[0] * L * 0.9), int(c[1] + perp[1] * L * 0.9)),
                        (255, 0, 255), 4, tipLength=0.25)
        cv2.line(vis, tuple(c.astype(int)), tuple(palm.astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"nearest object: {label}  d={best_d:.0f}px",
                    (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(vis, f"nearest object: {label}  d={best_d:.0f}px",
                    (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(vis, "white=principal axis | magenta=grasp dir | red=hand",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis, "white=principal axis | magenta=grasp dir | red=hand",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        p = f"{OUT}/nearest_{tag}.jpg"
        cv2.imwrite(p, vis)
        print(f"  已保存 {p}")


if __name__ == "__main__":
    main()
