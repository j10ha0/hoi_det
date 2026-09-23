"""object_vec_v2.py — 改进的「物体向量」提取（解决部分掩膜问题）

对比两种方案:
  v1 (旧): FastSAM 全图分割 → 取离手最近区域 → PCA
           问题: 可能只分到物体一部分 / 手物黏连 → 主轴不准

  v2 (新): ⭐ 绿幕抠像 + 手部剔除 + 连通域选择
           ① 绿幕 → 前景(手+物体)
           ② WiLoR 2D 关键点凸包(膨胀) → 手掩膜 → 从前景剔除
           ③ 连通域 → 取离手最近且面积合理的
           ④ PCA → 主轴

对 EgoTactile（纯绿背景）几乎能得到完美物体掩膜。
部署场景（非绿幕）可把①换成 SAM 点提示 / 检测器，②③④不变。
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
VIS = f"{BASE}/vis_v2"
SPHERE = {"Apple", "Orange", "TennisBall", "Tomato"}


def green_foreground(img):
    """绿幕抠像 → 前景掩膜（手 + 物体）"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([90, 255, 255]))
    fg = cv2.bitwise_not(green)
    k = np.ones((7, 7), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    # 只保留最大连通域（去掉散落噪点）+ 手
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = np.zeros_like(fg)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 2000:
            keep[lab == i] = 255
    return keep


def hand_mask_from_kp(kp2d, shape, dilate=22):
    """手 + 前臂 掩膜（关键：必须排除前臂，否则前臂会被当成"细长物体"）"""
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    pts = kp2d.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
    # ⭐ 沿"腕→中指MCP"反方向延伸出前臂矩形
    wrist = kp2d[0].astype(float)
    mid = kp2d[9].astype(float)
    f = mid - wrist
    L = float(np.linalg.norm(f)) + 1e-6
    fh = f / L
    pp = np.array([-fh[1], fh[0]])
    quad = np.array([wrist + 2.2 * L * (-fh) + 1.0 * L * pp,
                     wrist + 2.2 * L * (-fh) - 1.0 * L * pp,
                     wrist - 1.0 * L * pp,
                     wrist + 1.0 * L * pp]).astype(np.int32)
    cv2.fillConvexPoly(m, quad, 255)
    m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m


def feats(mask, use_hull=False):
    """use_hull=True: 先取凸包并填满 → 修复"被手挖掉一块"的部分掩膜"""
    if use_hull:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        m2 = np.zeros_like(mask, np.uint8)
        cv2.fillConvexPoly(m2, hull, 1)
        mask = m2 > 0
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
    x, y, bw, bh = cv2.boundingRect(cnt)
    return dict(center=c, major=v[:, 1], evr=evr, area=int(mask.sum()),
                circularity=circ, bbox=(x, y, bw, bh),
                fill=float(mask.sum()) / (bw * bh + 1e-6))


def classify(f):
    if f["evr"] > 0.55:
        if f["circularity"] > 0.80:
            return "球/圆盘", (0, 165, 255), False
        return "不规则", (128, 128, 128), False
    if f["evr"] < 0.15:
        return "细长(圆柱/长条)", (255, 255, 0), True
    return "中等长宽比(棱柱/盒)", (0, 255, 0), True


def main():
    os.makedirs(VIS, exist_ok=True)
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float32, verbose=False)

    tot = ok = nohand = nocand = axis_ok = axis_n = 0
    axis_ok_raw = 0
    for img_path in sorted(glob.glob(f"{FRAMES}/*.jpg")):
        base = os.path.basename(img_path)[:-4]
        obj = base.split("_p")[0]
        is_sphere = obj in SPHERE
        tot += 1
        img = cv2.imread(img_path); H, W = img.shape[:2]

        out = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not out:
            nohand += 1; print(f"{base:<44} ✗ 未检出手"); continue
        o = max(out, key=lambda z: (z["hand_bbox"][2] - z["hand_bbox"][0]) *
                                   (z["hand_bbox"][3] - z["hand_bbox"][1]))
        kp2d = np.asarray(o["wilor_preds"]["pred_keypoints_2d"]).reshape(-1, 2)
        kp3d = np.asarray(o["wilor_preds"]["pred_keypoints_3d"]).reshape(-1, 3)
        palm = (kp2d[0] + kp2d[5] + kp2d[17]) / 3
        n3 = np.cross(kp3d[5] - kp3d[0], kp3d[17] - kp3d[0]); n3 /= np.linalg.norm(n3) + 1e-9
        n2 = np.array([n3[0], n3[1]]); n2 /= (np.linalg.norm(n2) + 1e-9)

        fg = green_foreground(img)
        hm = hand_mask_from_kp(kp2d, (H, W))
        obj_fg = cv2.bitwise_and(fg, cv2.bitwise_not(hm))
        obj_fg = cv2.morphologyEx(obj_fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        n, lab, stats, cents = cv2.connectedComponentsWithStats(obj_fg, 8)
        best = None
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if a < 3000 or a > 0.40 * W * H:
                continue
            m = (lab == i)
            f = feats(m)
            if f is None:
                continue
            d = f["center"] - palm; dn = float(np.linalg.norm(d))
            if dn < 60:
                continue
            v = d / (dn + 1e-9)
            ang = float(np.degrees(np.arccos(np.clip(np.dot(v, n2), -1, 1))))
            sc = (1 - ang / 90) * 2 + (1 if a > 8000 else 0.4)
            if best is None or sc > best[0]:
                best = (sc, ang, dn, f, m)
        if best is None:
            nocand += 1; print(f"{base:<44} ✗ 无候选（绿幕法）"); continue
        sc, ang, dn, f_raw, m = best
        f = feats(m, use_hull=True)          # ⭐ 用凸包修复部分掩膜
        if f is None:
            f = f_raw
        lb, color, has_axis = classify(f)
        lb_raw, _, has_axis_raw = classify(f_raw)
        agree = (has_axis != is_sphere)
        ok += 1; axis_n += 1
        if agree:
            axis_ok += 1
        if (has_axis_raw != is_sphere):
            axis_ok_raw += 1
        print(f"{base:<44} GT[{'球' if is_sphere else '有轴':<4}] → 凸包[{lb:<18}]{'✓' if agree else '✗'}"
              f"  原掩膜[{lb_raw:<18}]{'✓' if (has_axis_raw != is_sphere) else '✗'}"
              f"  evr{f['evr']:.2f}/{f_raw['evr']:.2f} fill{f_raw['fill']:.2f}")

        vis = img.copy()
        vis[m] = (vis[m] * 0.45 + np.array(color, np.uint8) * 0.55).astype(np.uint8)
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
        t = f"{obj} GT:{'sphere' if is_sphere else 'axis'} pred:{lb} ang={ang:.0f} d={dn:.0f}px"
        cv2.putText(vis, t, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(vis, t, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite(f"{VIS}/{base}.jpg", vis)

    print(f"\n=== v2（绿幕抠像+手剔除）汇总 ===")
    print(f"  总帧数 {tot} | 成功定位 {ok} ({100*ok/max(1,tot):.0f}%) | "
          f"未检出手 {nohand} | 无候选 {nocand}")
    print(f"  ⭐ 凸包修复后一致率 {axis_ok}/{axis_n} ({100*axis_ok/max(1,axis_n):.0f}%)")
    print(f"     原掩膜     一致率 {axis_ok_raw}/{axis_n} ({100*axis_ok_raw/max(1,axis_n):.0f}%)")


if __name__ == "__main__":
    main()
