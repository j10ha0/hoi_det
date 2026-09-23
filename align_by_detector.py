"""align_by_detector.py — 用【标准检测模型】的框中心做对准目标

流程:
  ① YOLO (COCO) → 检测物体 → 取最大框（排除 person/table/chair）→ 框中心 p_obj
  ② WiLoR       → 掌心位置 p_palm + 掌心法向 n
  ③ u = normalize(p_obj − p_palm);  ang = angle(n, u);  Δ = signed(n → u)
  ④ aligned ⟺ ang < EPS
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
VIS = f"{BASE}/vis_det"
EPS = 40.0
SKIP = {"person", "dining table", "chair", "cup"}   # cup 常误检为持物手附近的容器


def norm2(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def main():
    os.makedirs(VIS, exist_ok=True)
    from ultralytics import YOLO
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    det = YOLO("yolo11s.pt")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float32, verbose=False)

    recs = []
    print(f"{'帧':<34} {'检测框(类别/置信/中心)':<44} {'掌心法向':>12}")
    for f in sorted(glob.glob(f"{BASE}/nc_frames4/*.jpg")):
        base = os.path.basename(f)[:-4]
        obj_name = base.replace("_nc", "")
        img = cv2.imread(f); H, W = img.shape[:2]

        r = det.predict(img, conf=0.15, iou=0.5, verbose=False)[0]
        best = None
        if r.boxes is not None and len(r.boxes):
            for i in range(len(r.boxes)):
                c = float(r.boxes.conf[i]); k = int(r.boxes.cls[i]); nm = r.names[k]
                if nm in SKIP:
                    continue
                xy = r.boxes.xyxy[i].cpu().numpy()
                a = (xy[2] - xy[0]) * (xy[3] - xy[1])
                if best is None or a > best[0]:
                    best = (a, nm, c, xy)
        if best is None:
            print(f"{base:<34} ✗ 未检出目标物体"); continue
        _, nm, conf, xy = best
        p_obj = np.array([(xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2])

        out = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not out:
            print(f"{base:<34} ✗ WiLoR 未检出手"); continue
        o = max(out, key=lambda z: (z["hand_bbox"][2] - z["hand_bbox"][0]) *
                                   (z["hand_bbox"][3] - z["hand_bbox"][1]))
        kp2d = np.asarray(o["wilor_preds"]["pred_keypoints_2d"]).reshape(-1, 2)
        kp3d = np.asarray(o["wilor_preds"]["pred_keypoints_3d"]).reshape(-1, 3)
        p_palm = (kp2d[0] + kp2d[5] + kp2d[17]) / 3
        n3 = np.cross(kp3d[9] - kp3d[0], kp3d[17] - kp3d[5]); n3 /= np.linalg.norm(n3) + 1e-9
        n2 = norm2([n3[0], n3[1]])

        recs.append(dict(base=base, img=img, p_obj=p_obj, xy=xy, nm=nm, conf=conf,
                         p_palm=p_palm, n2=n2, obj_name=obj_name,
                         hand="R" if int(o.get("is_right", 1)) else "L"))
        print(f"{base:<34} {nm:<12}({conf:.2f}) 中心=({p_obj[0]:.0f},{p_obj[1]:.0f})   "
              f"({n2[0]:+.2f},{n2[1]:+.2f})")

    # 全局符号标定
    if recs:
        ap = [math.degrees(math.acos(float(np.clip(np.dot(r['n2'], norm2(r['p_obj'] - r['p_palm'])), -1, 1)))) for r in recs]
        am = [math.degrees(math.acos(float(np.clip(np.dot(-r['n2'], norm2(r['p_obj'] - r['p_palm'])), -1, 1)))) for r in recs]
        SIGN = 1 if np.median(ap) <= np.median(am) else -1
        print(f"\n符号标定: +1 中位 {np.median(ap):.1f}°  -1 中位 {np.median(am):.1f}°  → 选 {SIGN:+d}")

        print(f"\n{'帧':<34} {'检测':<12} {'ang':>7} {'Δ':>7} {'距离':>7}  判定")
        n_al = 0
        sheets = []
        for r in recs:
            n2 = r["n2"] * SIGN
            u = norm2(r["p_obj"] - r["p_palm"])
            ang = math.degrees(math.acos(float(np.clip(np.dot(n2, u), -1, 1))))
            cr = n2[0] * u[1] - n2[1] * u[0]
            delta = math.degrees(math.atan2(cr, float(np.dot(n2, u))))
            d = float(np.linalg.norm(r["p_obj"] - r["p_palm"]))
            al = ang < EPS
            n_al += int(al)
            print(f"{r['base']:<34} {r['nm']:<12} {ang:6.1f}° {delta:+6.1f}° {d:6.0f}px  "
                  f"{'✅已对准' if al else '未对准'}")
            img = r["img"]; vis = img.copy()
            x1, y1, x2, y2 = [int(v) for v in r["xy"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 4)
            cv2.putText(vis, f"{r['nm']} {r['conf']:.2f}", (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 3)
            pp = tuple(r["p_palm"].astype(int)); po = tuple(r["p_obj"].astype(int))
            cv2.circle(vis, po, 12, (0, 200, 0), -1); cv2.circle(vis, po, 12, (255, 255, 255), 2)
            cv2.line(vis, pp, po, (0, 200, 0), 3, cv2.LINE_AA)
            L = 240
            cv2.arrowedLine(vis, pp, (int(pp[0] + n2[0] * L), int(pp[1] + n2[1] * L)),
                            (0, 0, 255), 6, tipLength=0.18)
            cv2.circle(vis, pp, 10, (0, 0, 255), -1)
            a0 = math.degrees(math.atan2(n2[1], n2[0])); a1 = math.degrees(math.atan2(u[1], u[0]))
            cv2.ellipse(vis, pp, (105, 105), 0, min(a0, a1), max(a0, a1), (255, 255, 255), 3)
            t1 = f"{r['obj_name']} -> det:{r['nm']}  ang={ang:.0f}deg turn={delta:+.0f}deg"
            t2 = "ALIGNED" if al else "NOT ALIGNED"
            for t, y, col in ((t1, 42, (255, 255, 255)), (t2, 78, (0, 200, 0) if al else (0, 0, 255))):
                cv2.putText(vis, t, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 5)
                cv2.putText(vis, t, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, col, 2)
            cv2.imwrite(f"{VIS}/{r['base']}.jpg", vis)
            sheets.append(f"{VIS}/{r['base']}.jpg")
        print(f"\n已对准 {n_al}/{len(recs)} ({100*n_al/max(1,len(recs)):.0f}%)")

        from PIL import Image
        th = 430; ims = [Image.open(s).convert("RGB") for s in sheets]
        ims = [i.resize((th, int(i.height * th / i.width))) for i in ims]
        if ims:
            cols = 4; rows = math.ceil(len(ims) / cols)
            cw = max(i.width for i in ims); chh = max(i.height for i in ims)
            sh = Image.new("RGB", (cols * cw + (cols + 1) * 6, rows * chh + (rows + 1) * 6), (235, 235, 235))
            for k, im in enumerate(ims):
                rr, cc = divmod(k, cols)
                sh.paste(im, (6 + cc * (cw + 6), 6 + rr * (chh + 6)))
            sh.save(f"{BASE}/overview_det.jpg", quality=82)
            print(f"总览拼图: {BASE}/overview_det.jpg")


if __name__ == "__main__":
    main()
