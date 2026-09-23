"""ego_nearest_object.py — EGO 视角：检测「手正在接近的最近物体」的位置与 shape

改进点（相对上一版）:
  ⭐ 判据从「中心距掌心最近」改为「在掌心法向方向上 + 距离最近」
     因为手指碎块虽然近，但方向不对；掌心朝向的那一侧才是"要抓的物体"

流水线:
  1. WiLoR → 手心位置 + 掌心法向（3D 关键点算）
  2. FastSAM 全图分割 → 候选区域
  3. 过滤: 面积、与手重叠、太贴手
  4. ⭐ 方向过滤: 物体中心方向 与 掌心法向 夹角 < 60°
  5. 在通过者中取距离最近 → 目标物体
  6. PCA + 形状特征 → shape 判定
  7. 可视化
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

FRAMES = "/mnt/storage/siat_cjh/proj/hoi_data/ego_frames"
OUT = "/mnt/storage/siat_cjh/proj/hoi_data"
MIN_AREA = 4000          # ego 图 1920x1080，物体更大
ANG_MAX = 60.0           # 物体中心方向与掌心法向的最大夹角


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
    ext = a / (bw * bh + 1e-6)
    return dict(center=c, major=v[:, 1], sv=sv, evr=evr, area=int(mask.sum()),
                circularity=circ, solidity=sol, extent=ext, bbox=(x, y, bw, bh))


def classify(f):
    if f["evr"] > 0.55:
        if f["circularity"] > 0.80:
            return "球 / 圆盘", (0, 165, 255)
        if f["extent"] > 0.85:
            return "方块 / 矩形（俯视）", (255, 128, 0)
        return "不规则", (128, 128, 128)
    return ("细长（圆柱 / 长条）", (255, 255, 0)) if f["evr"] < 0.15 \
        else ("中等长宽比（棱柱 / 盒）", (0, 255, 0))


def main():
    from ultralytics import FastSAM
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float32, verbose=False)
    fsam = FastSAM("FastSAM-s.pt")

    for img_path in sorted(glob.glob(f"{FRAMES}/e*.jpg")):
        tag = os.path.basename(img_path)[:-4]
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        vis = img.copy()
        print(f"\n===== {tag} ({W}x{H}) =====")

        out = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not out:
            print("  WiLoR 未检出手"); continue
        o = max(out, key=lambda z: (z["hand_bbox"][2] - z["hand_bbox"][0]) *
                                   (z["hand_bbox"][3] - z["hand_bbox"][1]))
        hb = o["hand_bbox"]
        kp2d = np.asarray(o["wilor_preds"]["pred_keypoints_2d"]).reshape(-1, 2)
        kp3d = np.asarray(o["wilor_preds"]["pred_keypoints_3d"]).reshape(-1, 3)
        is_right = int(o.get("is_right", 1))
        palm = (kp2d[0] + kp2d[5] + kp2d[17]) / 3

        n3 = np.cross(kp3d[5] - kp3d[0], kp3d[17] - kp3d[0])
        n3 = n3 / (np.linalg.norm(n3) + 1e-9)
        if is_right == 0:
            n3 = -n3
        print(f"  手: {'右' if is_right else '左'} bbox={[int(v) for v in hb]} "
              f"掌心=({palm[0]:.0f},{palm[1]:.0f})  掌心法向(nx,ny)=({n3[0]:+.2f},{n3[1]:+.2f})")

        res = fsam(img, device="cuda:0", retina_masks=True, imgsz=1024,
                   conf=0.4, iou=0.9, verbose=False)[0]
        if res.masks is None:
            print("  未分割出区域"); continue
        masks = res.masks.data.cpu().numpy()

        hand_box = np.array(hb, dtype=float)
        hand_r = 0.5 * max(hand_box[2] - hand_box[0], hand_box[3] - hand_box[1])
        border = np.zeros((H, W), bool)
        border[:8, :] = border[-8:, :] = border[:, :8] = border[:, -8:] = True

        def collect(n2):
            got = []
            for m in masks:
                mb = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                area = int(mb.sum())
                if area < MIN_AREA or area > 0.20 * W * H:      # 排除过小/过大（背景）
                    continue
                if (mb & border).sum() > 120:                     # 触边 → 背景
                    continue
                f = shape_features(mb)
                if f is None:
                    continue
                x, y, bw, bh = f["bbox"]
                inter = max(0, min(x + bw, hand_box[2]) - max(x, hand_box[0])) * \
                        max(0, min(y + bh, hand_box[3]) - max(y, hand_box[1]))
                if inter / (bw * bh + 1e-6) > 0.30:
                    continue
                d = f["center"] - palm
                dn = float(np.linalg.norm(d))
                if dn < 0.7 * hand_r:
                    continue
                v = d / (dn + 1e-9)
                ang = float(np.degrees(np.arccos(np.clip(np.dot(v, n2), -1, 1))))
                got.append((ang, dn, f, mb))
            return got

        n2 = np.array([n3[0], n3[1]], dtype=float)
        n2 /= (np.linalg.norm(n2) + 1e-9)
        # ⭐ 符号自动选择：两个方向都试，选「最佳候选更近」的那个
        best_choice, sign_used, allc = None, "+", []
        for sgn, cand_n in [("+", n2), ("-", -n2)]:
            allc = collect(cand_n)
            ok = [c for c in allc if c[0] < ANG_MAX]
            if ok:
                ok.sort(key=lambda z: z[1])
                if best_choice is None or ok[0][1] < best_choice[0]:
                    best_choice, sign_used, cands = (ok[0][1], ok), sgn, ok
        if best_choice is None:
            print("  未找到「掌心朝向方向上」的物体")
            allc.sort(key=lambda z: z[1])
            for i, (ang, d, f, _) in enumerate(allc[:4]):
                lb, _ = classify(f)
                print(f"      夹角 {ang:5.1f}°  距离 {d:6.0f}px  面积 {f['area']:>7} → {lb}")
            continue
        cands, n2 = best_choice[1], (n2 if sign_used == "+" else -n2)
        ang, d, f, mb = cands[0]
        print(f"  掌心法向符号自动选择: {sign_used}  (候选 {len(cands)} 个)")
        allc.sort(key=lambda z: z[1])
        for i, (a_, d_, f_, _) in enumerate(allc[:4]):
            lb, _ = classify(f_)
            mark = "✅" if a_ < ANG_MAX else "  "
            print(f"    {mark} 夹角 {a_:5.1f}°  距离 {d_:6.0f}px  面积 {f_['area']:>7} "
                  f"特征值比 {f_['evr']:.2f} → {lb}")

        label, color = classify(f)
        print(f"  ⭐ 目标物体: 夹角 {ang:.1f}°  距离 {d:.0f}px  面积 {f['area']}  → 【{label}】")

        vis[mb] = (vis[mb] * 0.45 + np.array(color, dtype=np.uint8) * 0.55).astype(np.uint8)
        x, y, bw, bh = f["bbox"]
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 4)
        c, major = f["center"], f["major"]
        L = 0.55 * max(bw, bh)
        cv2.line(vis, (int(c[0] + major[0] * L), int(c[1] + major[1] * L)),
                 (int(c[0] - major[0] * L), int(c[1] - major[1] * L)), (255, 255, 255), 5)
        perp = np.array([-major[1], major[0]])
        if np.dot(perp, palm - c) < 0:
            perp = -perp
        cv2.arrowedLine(vis, tuple(c.astype(int)),
                        (int(c[0] + perp[0] * L * 0.9), int(c[1] + perp[1] * L * 0.9)),
                        (255, 0, 255), 5, tipLength=0.25)
        cv2.arrowedLine(vis, tuple(palm.astype(int)),
                        (int(palm[0] + n2[0] * 200), int(palm[1] + n2[1] * 200)),
                        (0, 0, 255), 5, tipLength=0.2)
        cv2.circle(vis, tuple(palm.astype(int)), 9, (0, 0, 255), -1)
        txt = f"{label}  d={d:.0f}px  ang={ang:.0f}deg"
        cv2.putText(vis, txt, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(vis, txt, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        p = f"{OUT}/ego_{tag}.jpg"
        cv2.imwrite(p, vis)
        print(f"  已保存 {p}")


if __name__ == "__main__":
    main()
