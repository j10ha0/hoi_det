"""align_center_demo2.py — 简化版 v2：掌心法向对准【物体中心】

修正两点:
  ① 掌心法向【符号】用全局标定（数据驱动，一次定下来）—— 避免 160° 的假角度
  ② 几何简化：直接用 2D 带符号夹角（更直观、无退化）
       ang = angle(n2, u2)                → 偏差多大
       Δ   = signed(n2 → u2)              → 往哪转、转多少（图像平面）
       aligned ⟺ ang < EPS
     "对应旋前还是旋后" 需要一次性标定（此处打印 Δ 符号供你核对）
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
FRAMES = f"{BASE}/nc_frames"
VIS = f"{BASE}/vis_nc"
EPS = 40.0


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
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def ang_between(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1, 1))))


def signed_angle(a, b):
    """a → b 的带符号夹角（度，图像坐标）"""
    cr = a[0] * b[1] - a[1] * b[0]
    return math.degrees(math.atan2(cr, float(np.dot(a, b))))


def main():
    os.makedirs(VIS, exist_ok=True)
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float32, verbose=False)

    # ---------- Pass 1: 收集数据 + 全局定符号 ----------
    recs = []
    for img_path in sorted(glob.glob(f"{FRAMES}/*.jpg")):
        base = os.path.basename(img_path)[:-4]
        img = cv2.imread(img_path); H, W = img.shape[:2]
        out = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not out:
            print(f"{base:<42} ✗ 未检出手"); continue
        o = max(out, key=lambda z: (z["hand_bbox"][2] - z["hand_bbox"][0]) *
                                   (z["hand_bbox"][3] - z["hand_bbox"][1]))
        kp2d = np.asarray(o["wilor_preds"]["pred_keypoints_2d"]).reshape(-1, 2)
        kp3d = np.asarray(o["wilor_preds"]["pred_keypoints_3d"]).reshape(-1, 3)
        p_palm = (kp2d[0] + kp2d[5] + kp2d[17]) / 3
        n3 = np.cross(kp3d[9] - kp3d[0], kp3d[17] - kp3d[5]); n3 /= np.linalg.norm(n3) + 1e-9
        n2_raw = norm2([n3[0], n3[1]])

        fg = green_fg(img)
        om = cv2.bitwise_and(fg, cv2.bitwise_not(hand_forearm_mask(kp2d, (H, W))))
        om = cv2.morphologyEx(om, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lab, st, cent = cv2.connectedComponentsWithStats(om, 8)
        best, bd = None, 1e18
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if a < 3000 or a > 0.40 * W * H:
                continue
            c = cent[i]; d = float(np.linalg.norm(c - p_palm))
            if d < 60:
                continue
            if d < bd:
                bd, best = d, (c, a, (lab == i))
        if best is None:
            print(f"{base:<42} ✗ 无物体候选"); continue
        p_obj, area, m = best
        u = norm2(p_obj - p_palm)
        recs.append(dict(base=base, img=img, p_palm=p_palm, n2_raw=n2_raw, u=u,
                         p_obj=p_obj, m=m, d=bd,
                         obj=base.split("_p")[0],
                         hand="R" if int(o.get("is_right", 1)) else "L"))
        print(f"[收集] {base:<40} 手={recs[-1]['hand']} 距离{bd:5.0f}px")

    # 全局符号标定：选让「掌心中位夹角」更小的那个符号
    a_plus = [ang_between(r["n2_raw"], r["u"]) for r in recs]
    a_minus = [ang_between(-r["n2_raw"], r["u"]) for r in recs]
    SIGN = 1 if np.median(a_plus) <= np.median(a_minus) else -1
    print(f"\n=== 掌心法向符号标定 ===")
    print(f"  sign=+1 中位夹角 {np.median(a_plus):6.1f}°")
    print(f"  sign=-1 中位夹角 {np.median(a_minus):6.1f}°")
    print(f"  → 选用 sign = {SIGN:+d}")

    # ---------- Pass 2: 输出 ----------
    stat = dict(ok=0, aligned=0)
    sheets = []
    print(f"\n{'帧':<40} {'手':>3} {'偏差ang':>8} {'需转Δ':>8} {'距离':>7}  判定")
    for r in recs:
        n2 = r["n2_raw"] * SIGN
        ang = ang_between(n2, r["u"])
        delta = signed_angle(n2, r["u"])
        aligned = ang < EPS
        stat["ok"] += 1
        if aligned:
            stat["aligned"] += 1
        print(f"{r['base']:<40} {r['hand']:>3} {ang:7.1f}° {delta:+7.1f}° {r['d']:6.0f}px  "
              f"{'✅已对准' if aligned else '未对准'}")
        img = r["img"]; H, W = img.shape[:2]
        vis = img.copy()
        m = r["m"]
        vis[m] = (vis[m] * 0.55 + np.array([0, 255, 255], np.uint8) * 0.45).astype(np.uint8)
        pp = tuple(r["p_palm"].astype(int)); po = tuple(r["p_obj"].astype(int))
        cv2.circle(vis, po, 13, (0, 200, 0), -1); cv2.circle(vis, po, 13, (255, 255, 255), 2)
        cv2.line(vis, pp, po, (0, 200, 0), 3, cv2.LINE_AA)
        L = 240
        cv2.arrowedLine(vis, pp, (int(pp[0] + n2[0] * L), int(pp[1] + n2[1] * L)),
                        (0, 0, 255), 6, tipLength=0.18)
        cv2.arrowedLine(vis, pp, (int(pp[0] + r["u"][0] * L), int(pp[1] + r["u"][1] * L)),
                        (0, 200, 0), 5, tipLength=0.18)
        cv2.circle(vis, pp, 10, (0, 0, 255), -1)
        a0 = math.degrees(math.atan2(n2[1], n2[0])); a1 = math.degrees(math.atan2(r["u"][1], r["u"][0]))
        cv2.ellipse(vis, pp, (105, 105), 0, min(a0, a1), max(a0, a1), (255, 255, 255), 3)
        t1 = f"{r['obj']}   ang={ang:.0f}deg   turn={delta:+.0f}deg"
        t2 = "ALIGNED" if aligned else "NOT ALIGNED"
        cv2.putText(vis, t1, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 5)
        cv2.putText(vis, t1, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        cv2.putText(vis, t2, (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 200, 0) if aligned else (0, 0, 255), 4)
        cv2.putText(vis, t2, (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite(f"{VIS}/{r['base']}.jpg", vis)
        sheets.append(f"{VIS}/{r['base']}.jpg")

    print(f"\n=== 汇总 ===")
    print(f"  成功处理 {stat['ok']} 帧 | 判定已对准(ang<{EPS:.0f}°) {stat['aligned']} 帧 "
          f"({100*stat['aligned']/max(1,stat['ok']):.0f}%)")
    from PIL import Image
    th = 420; ims = [Image.open(f).convert("RGB") for f in sheets]
    ims = [i.resize((th, int(i.height * th / i.width))) for i in ims]
    if ims:
        cols = 4; rows = math.ceil(len(ims) / cols)
        cw = max(i.width for i in ims); ch = max(i.height for i in ims)
        sh = Image.new("RGB", (cols * cw + (cols + 1) * 6, rows * ch + (rows + 1) * 6), (235, 235, 235))
        for k, im in enumerate(ims):
            rr, cc = divmod(k, cols)
            sh.paste(im, (6 + cc * (cw + 6), 6 + rr * (ch + 6)))
        sh.save(f"{BASE}/overview_center.jpg", quality=82)
        print(f"  总览拼图: {BASE}/overview_center.jpg")


if __name__ == "__main__":
    main()
