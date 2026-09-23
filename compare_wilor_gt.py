"""compare_wilor_gt.py — 实测 WiLoR 追踪「腕部旋转变化」的精度

方法（与坐标系约定无关）:
    GT  : EgoDex 真值 4x4 手部姿态（ARKit）—— 右手/左手分开取
    PRED: WiLoR 的 global_orient（MANO 根旋转，相机坐标系）

    不比绝对姿态（MANO 与 ARKit 约定不同），改比【相对旋转】:
        ΔR(i) = R(0)^T @ R(i)
    若两套约定只差一个固定旋转，ΔR 应完全一致（约定自动抵消）。
    误差 = 相对旋转之间的测地距离（度）= "追踪腕旋变化的能力"
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd
import torch

for _n, _t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("unicode", str), ("str", str)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)
np.nan = float("nan"); np.inf = float("inf")

PARQ = "/mnt/storage/siat_cjh/proj/hoi_data/egodex_ep0.parquet"
FRAMES = "/mnt/storage/siat_cjh/proj/hoi_data/gt_frames"


def to_mat(x):
    return np.vstack([np.asarray(r, dtype=float).ravel() for r in x])


def geo(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def main():
    df = pd.read_parquet(PARQ)
    files = sorted(glob.glob(f"{FRAMES}/n*.jpg"))
    fidx = [int(os.path.basename(f)[1:6]) for f in files]
    print(f"抽帧 {len(files)} 张: {fidx[:4]} ... {fidx[-2:]}")

    Rgt = {0: [to_mat(x)[:3, :3] for x in df["observation.state.leftHand"].values],   # 0=左手
           1: [to_mat(x)[:3, :3] for x in df["observation.state.rightHand"].values]}  # 1=右手

    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float32, verbose=False)

    # 按手别收集
    preds = {0: [], 1: []}      # 帧号 + R
    for k, f in enumerate(files):
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        out = pipe.predict(img)
        for o in out:
            h = int(o.get("is_right", 1))
            go = np.asarray(o["wilor_preds"]["global_orient"]).reshape(-1, 3)[0].astype(np.float64)
            Rm, _ = cv2.Rodrigues(go)
            preds.setdefault(h, []).append((fidx[k], Rm))
        print(f"  帧{fidx[k]}: 检出 {len(out)} 手 ({[int(o.get('is_right',1)) for o in out]})")

    print("\n=== 结果：WiLoR 追踪「腕部旋转变化」的误差 ===")
    allerr = []
    for h, name in [(1, "右手"), (0, "左手")]:
        lst = preds.get(h, [])
        if len(lst) < 3:
            print(f"  {name}: 样本不足（{len(lst)}）")
            continue
        idx = [x[0] for x in lst]
        Rm = np.stack([x[1] for x in lst])
        gt = np.stack([Rgt[h][i] for i in idx])
        d_gt = np.stack([gt[0].T @ R for R in gt])
        d_pr = np.stack([Rm[0].T @ R for R in Rm])
        errs = np.array([geo(d_gt[i], d_pr[i]) for i in range(len(Rm))])
        allerr.append(errs)
        print(f"  {name} (n={len(errs)}): 中位数 {np.median(errs):5.1f}°  "
              f"均值 {errs.mean():5.1f}°  最大 {errs.max():5.1f}°")
    if allerr:
        e = np.concatenate(allerr)
        print(f"\n  合计 (n={len(e)}): 中位数 {np.median(e):.1f}°  均值 {e.mean():.1f}°")
        print(f"  判据: <15° 三分类可用 | 15-30° 只能粗用 | >30° 不可用")
        np.save("/mnt/storage/siat_cjh/proj/hoi_data/wilor_err.npy", e)


if __name__ == "__main__":
    main()
