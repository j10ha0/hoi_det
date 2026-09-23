"""wilor_demo_vis.py — 用 WiLoR 标注一张测试图，输出可视化

图里画三样东西:
  ① WiLoR 预测的 21 个手部关键点（红）+ 骨架连线
  ② WiLoR 的 global_orient → 手部三个局部坐标轴（箭头，相机坐标系）
  ③ EgoDex 真值骨架（绿，用相机内参投影）→ 肉眼对比 WiLoR 准不准
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

# ---- numpy 兼容补丁（chumpy/MANO 需要）----
for _n, _t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("unicode", str), ("str", str)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)
np.nan = float("nan"); np.inf = float("inf")

PARQ = "/mnt/storage/siat_cjh/proj/hoi_data/egodex_ep0.parquet"
IMG = "/mnt/storage/siat_cjh/proj/hoi_data/gt_frames/n00300.jpg"
OUT = "/mnt/storage/siat_cjh/proj/hoi_data/wilor_vis.jpg"
FRAME_N = 300

HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
              (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


def to_mat(x):
    return np.vstack([np.asarray(r, dtype=float).ravel() for r in x])


def main():
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)

    img_bgr = cv2.imread(IMG)
    H, W = img_bgr.shape[:2]
    print(f"图像: {W}x{H}")

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device={dev}")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float32, verbose=False)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out = pipe.predict(img_rgb)
    print(f"检测到 {len(out)} 只手")
    if not out:
        print("未检测到手"); return

    vis = img_bgr.copy()

    # ---- ① WiLoR 关键点 ----
    for i, o in enumerate(out):
        wp = o["wilor_preds"]
        kp = np.asarray(wp["pred_keypoints_2d"]).reshape(-1, 2)
        for a, b in HAND_EDGES:
            cv2.line(vis, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (0, 0, 255), 2)
        for p in kp:
            cv2.circle(vis, tuple(p.astype(int)), 3, (0, 255, 255), -1)
        print(f"  手{i+1}: is_right={o.get('is_right')} bbox={[round(v) for v in o['hand_bbox']]}")

        # ---- ② global_orient → 三个局部轴 ----
        go = np.asarray(wp["global_orient"]).reshape(-1, 3)[0].astype(np.float64)
        Rm, _ = cv2.Rodrigues(go)
        print(f"  global_orient(axis-angle) = {np.round(go, 4)}")
        print(f"  旋转矩阵 R = \n{np.round(Rm, 3)}")
        # 手根位置（用预测相机平移近似）
        t = np.asarray(wp["pred_cam_t_full"]).reshape(-1, 3)[0]
        f = float(np.asarray(wp["scaled_focal_length"]).ravel()[0])
        print(f"  pred_cam_t_full = {np.round(t,3)}   scaled_focal_length = {f:.1f}")

        def project(X):
            x = f * X[0] / X[2] + W / 2
            y = f * X[1] / X[2] + H / 2
            return int(x), int(y)

        # 轴向箭头：从腕部关键点出发，方向用正交近似 (v_x, v_y)
        origin2d = tuple(kp[0].astype(int))
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]   # BGR：X蓝 Y绿 Z红
        for ax in range(3):
            d = np.array([Rm[0, ax], Rm[1, ax]], dtype=float)
            n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            d = d / n * 140
            p1 = (int(origin2d[0] + d[0]), int(origin2d[1] + d[1]))
            cv2.arrowedLine(vis, origin2d, p1, colors[ax], 3, tipLength=0.25)
            cv2.putText(vis, "XYZ"[ax], p1, cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[ax], 2)

    # ---- ③ EgoDex 真值骨架 ----
    df = pd.read_parquet(PARQ)
    Kraw = df["camera_intrinsics"].values[FRAME_N]
    K = np.vstack([np.asarray(r, dtype=float).ravel() for r in Kraw]).reshape(3, 3)
    print(f"\nEgoDex 相机内参 K = \n{np.round(K,2)}")

    joint_cols = [f"observation.state.right{j}" for j in
                  ["Hand", "IndexFingerKnuckle", "IndexFingerTip", "MiddleFingerTip",
                   "RingFingerTip", "LittleFingerTip", "ThumbTip", "Forearm"]]
    pts = {}
    for c in joint_cols:
        if c in df.columns:
            X = to_mat(df[c].values[FRAME_N])[:3, 3]
            p = K @ X
            if p[2] > 0.01:
                pts[c.split(".")[-1]] = (int(p[0] / p[2]), int(p[1] / p[2]))
    for name, p in pts.items():
        cv2.circle(vis, p, 6, (0, 255, 0), 2)
        cv2.putText(vis, name[:9], (p[0] + 6, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(vis, "WiLoR keypoints (red/yellow) | hand axes RGB=XYZ | GT joints (green)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(OUT, vis)
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
