"""egotactile_batch.py — EgoTactile 批量评测：手 → 最近物体 → 位置 + shape

判据（可定量）:
  · 球类物体 (Apple/Orange/TennisBall/Tomato) → 应判为「无主轴」
  · 其他物体 (罐/瓶/香蕉/胡萝卜/哑铃/夹子/管/杯面) → 应判为「有主轴」
输出:
  · 每个帧的判定 + 可视化 (vis/ 目录)
  · 汇总: 检出率 / 主轴判定一致率
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

BASE = "/mnt/storage/siat_cjh/proj/hoi_data/egotactile"
FRAMES = f"{BASE}/frames"
VIS = f"{BASE}/vis"
SPHERE = {"Apple", "Orange", "TennisBall", "Tomato"}


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
    """返回 (标签, 颜色, 是否有主轴)"""
    if f["evr"] > 0.55:
        if f["circularity"] > 0.80:
            return "球/圆盘", (0, 165, 255), False
        if f["extent"] > 0.85:
            return "方块/矩形", (255, 128, 0), False
        return "不规则", (128, 128, 128), False
    if f["evr"] < 0.15:
        return "细长(圆柱/长条)", (255, 255, 0), True
    return "中等长宽比(棱柱/盒)", (0, 255, 0), True


def main():
    os.makedirs(VIS, exist_ok=True)
    from ultralytics import FastSAM
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float32, verbose=False)
    fsam = FastSAM("FastSAM-s.pt")

    stat = dict(total=0, ok=0, no_hand=0, no_cand=0, axis_ok=0, axis_checked=0)
    detail = []
    for img_path in sorted(glob.glob(f"{FRAMES}/*.jpg")):
        base = os.path.basename(img_path)[:-4]
        obj = base.split("_p")[0]
        is_sphere = obj in SPHERE
        stat["total"] += 1
        img = cv2.imread(img_path); H, W = img.shape[:2]

        out = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not out:
            stat["no_hand"] += 1
            print(f"{base:<44} ✗ 未检出手"); continue
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
            stat["no_cand"] += 1
            print(f"{base:<44} ✗ 无分割"); continue
        masks = res.masks.data.cpu().numpy()
        hbox = np.array(hb, float); hr = 0.5 * max(hbox[2] - hbox[0], hbox[3] - hbox[1])
        border = np.zeros((H, W), bool)
        border[:8, :] = border[-8:, :] = border[:, :8] = border[:, -8:] = True
        n2 = np.array([n3[0], n3[1]], float); n2 /= (np.linalg.norm(n2) + 1e-9)

        best, sgn_used = None, "+"
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
            stat["no_cand"] += 1
            print(f"{base:<44} ✗ 无合格候选"); continue
        sc, ang, dn, ov, f, mb, sgn_used = best
        lb, color, has_axis = classify(f)
        agree = (has_axis != is_sphere)
        stat["ok"] += 1
        stat["axis_checked"] += 1
        if agree:
            stat["axis_ok"] += 1
        mark = "✓" if agree else "✗"
        print(f"{base:<44} GT[{'球' if is_sphere else '有轴':<4}] → {lb:<18} {mark} "
              f"ang{ang:5.1f}° d{dn:5.0f}px ov{ov*100:3.0f}% evr{f['evr']:.2f} circ{f['circularity']:.2f}")
        detail.append((base, is_sphere, lb, agree, ang, dn, ov, f["evr"]))

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
        t = f"{obj}  GT:{'sphere' if is_sphere else 'has-axis'}  pred:{lb}  ang={ang:.0f} d={dn:.0f}px"
        cv2.putText(vis, t, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(vis, t, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite(f"{VIS}/{base}.jpg", vis)

    print("\n=== 汇总 ===")
    print(f"  总帧数           {stat['total']}")
    print(f"  成功定位物体     {stat['ok']} ({100*stat['ok']/stat['total']:.0f}%)")
    print(f"  未检出手         {stat['no_hand']}")
    print(f"  无合格候选       {stat['no_cand']}")
    print(f"  ⭐ 主轴判定一致率 {stat['axis_ok']}/{stat['axis_checked']} "
          f"({100*stat['axis_ok']/max(1,stat['axis_checked']):.0f}%)")


if __name__ == "__main__":
    main()
