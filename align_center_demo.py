"""align_center_demo.py — 简化版：掌心法向对准【物体中心】

相比旧版删掉了：PCA、主轴、特征值比、形状分类
只保留：
  手  : WiLoR → 掌心位置 + 掌心法向 n + 手的长轴 f
  物  : 绿幕抠像 - 手/前臂 → 连通域 → 取离手最近 → 【质心】即为物体中心
  几何: u = normalize(p_obj − p_palm)
        ang   = angle(n, u)                     总夹角
        θ     = 绕手长轴 f 的带符号需转角
        resid = ang − |θ|                       残余（单自由度腕旋做不到的部分）
  判定: ang < EPS → 已对准
"""
import glob
import math
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
VIS = f"{BASE}/vis_center"
EPS = 40.0          # 判定"已对准"的角度阈值（度）


def green_fg(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([90, 255, 255]))
    fg = cv2.bitwise_not(green)
    k = np.ones((7, 7), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.morphologyEx(fg, cv2.MORPH_OPEN, k))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = np.zeros_like(fg)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 1500:
            keep[lab == i] = 255
    return keep


def hand_forearm_mask(kp2d, shape, dilate=22):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(kp2d.astype(np.int32).reshape(-1, 1, 2)), 255)
    wrist, mid = kp2d[0].astype(float), kp2d[9].astype(float)
    f = mid - wrist; L = float(np.linalg.norm(f)) + 1e-6; fh = f / L
    pp = np.array([-fh[1], fh[0]])
    quad = np.array([wrist + 2.2 * L * (-fh) + 1.0 * L * pp,
                     wrist + 2.2 * L * (-fh) - 1.0 * L * pp,
                     wrist - 1.0 * L * pp,
                     wrist + 1.0 * L * pp]).astype(np.int32)
    cv2.fillConvexPoly(m, quad, 255)
    return cv2.dilate(m, np.ones((dilate, dilate), np.uint8))


def norm2(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def main():
    os.makedirs(VIS, exist_ok=True)
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float32, verbose=False)

    stat = dict(tot=0, ok=0, nohand=0, noobj=0, aligned=0, resid_big=0)
    sheets = []
    for img_path in sorted(glob.glob(f"{FRAMES}/*.jpg")):
        base = os.path.basename(img_path)[:-4]
        obj_name = base.split("_p")[0]
        stat["tot"] += 1
        img = cv2.imread(img_path); H, W = img.shape[:2]

        out = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not out:
            stat["nohand"] += 1; print(f"{base:<42} ✗ 未检出手"); continue
        o = max(out, key=lambda z: (z["hand_bbox"][2] - z["hand_bbox"][0]) *
                                   (z["hand_bbox"][3] - z["hand_bbox"][1]))
        kp2d = np.asarray(o["wilor_preds"]["pred_keypoints_2d"]).reshape(-1, 2)
        kp3d = np.asarray(o["wilor_preds"]["pred_keypoints_3d"]).reshape(-1, 3)

        # ---- 手：掌心中心 / 掌心法向 / 手的长轴（全部投影到 2D）----
        p_palm = (kp2d[0] + kp2d[5] + kp2d[17]) / 3
        n3 = np.cross(kp3d[5] - kp3d[0], kp3d[17] - kp3d[0]); n3 /= np.linalg.norm(n3) + 1e-9
        if int(o.get("is_right", 1)) == 0:
            n3 = -n3
        n2 = norm2([n3[0], n3[1]])                       # 掌心法向（2D）
        f2 = norm2(kp2d[9] - kp2d[0])                     # 手的长轴（腕→中指MCP）

        # ---- 物体：绿幕 → 去手/前臂 → 连通域 → 离手最近者的质心 ----
        fg = green_fg(img)
        om = cv2.bitwise_and(fg, cv2.bitwise_not(hand_forearm_mask(kp2d, (H, W))))
        om = cv2.morphologyEx(om, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lab, st, cent = cv2.connectedComponentsWithStats(om, 8)
        best, bd = None, 1e18
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if a < 3000 or a > 0.40 * W * H:
                continue
            c = cent[i]
            d = float(np.linalg.norm(c - p_palm))
            if d < 60:
                continue
            if d < bd:
                bd, best = d, (c, a, (lab == i))
        if best is None:
            stat["noobj"] += 1; print(f"{base:<42} ✗ 无物体候选"); continue
        p_obj, area, m = best
        stat["ok"] += 1

        # ---- 几何：对准误差 ----
        u = norm2(p_obj - p_palm)                         # 掌心应指向的方向
        ang = math.degrees(math.acos(float(np.clip(np.dot(n2, u), -1, 1))))
        # 绕手长轴的带符号需转角
        n_perp = n2 - np.dot(n2, f2) * f2
        u_perp = u - np.dot(u, f2) * f2
        nn, un = np.linalg.norm(n_perp), np.linalg.norm(u_perp)
        if nn > 1e-6 and un > 1e-6:
            n_perp, u_perp = n_perp / nn, u_perp / un
            cross = n_perp[0] * u_perp[1] - n_perp[1] * u_perp[0]
            theta = math.degrees(math.atan2(cross, float(np.dot(n_perp, u_perp))))
        else:
            theta = 0.0
        resid = ang - abs(theta)
        aligned = ang < EPS
        if aligned:
            stat["aligned"] += 1
        if resid > 35:
            stat["resid_big"] += 1

        print(f"{base:<42} 物体[{obj_name:<20}] ang{ang:5.1f}°  "
              f"θ{theta:+6.1f}°  resid{resid:5.1f}°  距离{bd:5.0f}px  "
              f"{'✅已对准' if aligned else '未对准'}")

        # ---- 可视化 ----
        vis = img.copy()
        vis[m] = (vis[m] * 0.55 + np.array([0, 255, 255], np.uint8) * 0.45).astype(np.uint8)
        pp = tuple(p_palm.astype(int)); po = tuple(p_obj.astype(int))
        cv2.circle(vis, po, 12, (0, 200, 0), -1)
        cv2.circle(vis, po, 12, (255, 255, 255), 2)
        cv2.line(vis, pp, po, (0, 200, 0), 3, cv2.LINE_AA)          # 掌心→物体中心
        L = 230
        cv2.arrowedLine(vis, pp, (int(pp[0] + n2[0] * L), int(pp[1] + n2[1] * L)),
                        (0, 0, 255), 6, tipLength=0.18)              # 掌心法向（红）
        cv2.arrowedLine(vis, pp, (int(pp[0] + u[0] * L), int(pp[1] + u[1] * L)),
                        (0, 200, 0), 5, tipLength=0.18)              # 应指向方向（绿）
        cv2.arrowedLine(vis, pp, (int(pp[0] - f2[0] * 130), int(pp[1] - f2[1] * 130)),
                        (255, 200, 0), 4, tipLength=0.2)             # 手长轴（青）
        cv2.circle(vis, pp, 10, (0, 0, 255), -1)
        # 角度弧
        a0 = math.degrees(math.atan2(n2[1], n2[0]))
        a1 = math.degrees(math.atan2(u[1], u[0]))
        cv2.ellipse(vis, pp, (100, 100), 0, min(a0, a1), max(a0, a1), (255, 255, 255), 3)
        t1 = f"{obj_name}   ang={ang:.0f}deg   theta={theta:+.0f}deg   resid={resid:.0f}deg"
        t2 = ("ALIGNED" if aligned else "NOT ALIGNED") + \
             ("  [need arm move: resid>35]" if resid > 35 else "")
        cv2.putText(vis, t1, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 5)
        cv2.putText(vis, t1, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        cv2.putText(vis, t2, (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 200, 0) if aligned else (0, 0, 255), 5)
        cv2.putText(vis, t2, (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.imwrite(f"{VIS}/{base}.jpg", vis)
        sheets.append(f"{VIS}/{base}.jpg")

    print(f"\n=== 汇总（对准物体中心版）===")
    print(f"  总帧数 {stat['tot']} | 成功 {stat['ok']} | 未检出手 {stat['nohand']} | 无物体 {stat['noobj']}")
    print(f"  ✅ 判定「已对准」(ang<{EPS:.0f}°) : {stat['aligned']}/{stat['ok']}")
    print(f"  ⚠️ 残余误差>35° (需移动手臂): {stat['resid_big']}/{stat['ok']}")

    # 拼图
    from PIL import Image, ImageDraw
    th = 420; ims = []
    for f in sheets:
        im = Image.open(f).convert("RGB")
        im = im.resize((th, int(im.height * th / im.width)))
        ims.append(im)
    if ims:
        cols = 4; rows = math.ceil(len(ims) / cols)
        cw = max(i.width for i in ims); ch = max(i.height for i in ims)
        sheet = Image.new("RGB", (cols * cw + (cols + 1) * 6, rows * ch + (rows + 1) * 6), (235, 235, 235))
        for k, im in enumerate(ims):
            r, c = divmod(k, cols)
            sheet.paste(im, (6 + c * (cw + 6), 6 + r * (ch + 6)))
        sheet.save(f"{BASE}/overview_center.jpg", quality=82)
        print(f"  总览拼图: {BASE}/overview_center.jpg")


if __name__ == "__main__":
    main()
