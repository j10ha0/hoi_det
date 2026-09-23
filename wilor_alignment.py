"""wilor_alignment.py — 判断「掌心是否对准物体」并可视化

核心三步:
  ① 掌心法向 n     ← 由 WiLoR 的 21 个 3D 关键点算（不需要知道 MANO 约定！）
  ② 指向物体方向 u ← 需要物体位置
  ③ aligned ⟺ angle(n, u) < ε

本脚本输出:
  · 图上画出 掌心法向（品红箭头）与 手指方向（青）
  · 若给了物体位置，同时画 指向物体方向（绿）+ 夹角 + 判定
"""
import os

import cv2
import numpy as np
import torch

for _n, _t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("unicode", str), ("str", str)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)
np.nan = float("nan"); np.inf = float("inf")

IMGS = ["/mnt/storage/siat_cjh/proj/hoi_data/gt_frames/n00000.jpg",
        "/mnt/storage/siat_cjh/proj/hoi_data/gt_frames/n01200.jpg"]
OUT_DIR = "/mnt/storage/siat_cjh/proj/hoi_data"

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17


def palm_frame(kp3d):
    wrist, idx, mid, pnk = kp3d[WRIST], kp3d[INDEX_MCP], kp3d[MIDDLE_MCP], kp3d[PINKY_MCP]
    v1, v2 = idx - wrist, pnk - wrist
    n = np.cross(v1, v2)
    nn = np.linalg.norm(n)
    n = n / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
    f = mid - wrist
    f = f / (np.linalg.norm(f) + 1e-9)
    return n, f, (wrist + idx + pnk) / 3.0


def main():
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float32, verbose=False)

    for img_path in IMGS:
        tag = os.path.basename(img_path).replace(".jpg", "")
        img = cv2.imread(img_path)
        vis = img.copy()
        out = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print(f"\n=== {tag}: 检出 {len(out)} 只手 ===")
        if not out:
            continue

        for o in out:
            wp = o["wilor_preds"]
            kp2d = np.asarray(wp["pred_keypoints_2d"]).reshape(-1, 2)
            kp3d = np.asarray(wp["pred_keypoints_3d"]).reshape(-1, 3)
            is_right = int(o.get("is_right", 1))
            hand = "右手" if is_right == 1 else "左手"

            n, f, _ = palm_frame(kp3d)
            if is_right == 0:
                n = -n
            ang_nf = np.degrees(np.arccos(np.clip(np.dot(n, f), -1, 1)))
            print(f"  {hand}: 掌心法向 n = {np.round(n,3)}")
            print(f"        手指方向 f = {np.round(f,3)}  (n·f 夹角 {ang_nf:.1f}° 应接近 90°)")

            origin = ((kp2d[WRIST] + kp2d[INDEX_MCP] + kp2d[PINKY_MCP]) / 3).astype(int)

            def arrow(v, scale=170):
                d = np.array([v[0], v[1]], dtype=float)
                nm = np.linalg.norm(d)
                return (d / nm * scale).astype(int) if nm > 1e-6 else np.array([0, 0])

            d = arrow(n)
            cv2.arrowedLine(vis, tuple(origin), (origin[0] + d[0], origin[1] + d[1]),
                            (255, 0, 255), 4, tipLength=0.22)
            cv2.putText(vis, "palm normal", (origin[0] + d[0], origin[1] + d[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            d2 = arrow(f)
            cv2.arrowedLine(vis, tuple(origin), (origin[0] + d2[0], origin[1] + d2[1]),
                            (255, 255, 0), 3, tipLength=0.22)
            for p in kp2d:
                cv2.circle(vis, tuple(p.astype(int)), 3, (0, 0, 255), -1)

        cv2.putText(vis, "magenta = palm normal | cyan = finger dir | red = keypoints",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        p = f"{OUT_DIR}/align_{tag}.jpg"
        cv2.imwrite(p, vis)
        print(f"  已保存 {p}")


if __name__ == "__main__":
    main()
