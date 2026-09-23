"""h1_label_probe.py — 验证 H1 能否「零人工标注」导出

思路:
  H1（手到达物体附近）本质是一个【事件】：手什么时候开始抓。
  从 EgoDex 的 25 个手部关节可以算出「手指闭合度」：
      闭合度 = -mean( 指尖到掌心 的距离 )   （越小 = 越张开）
  闭合度曲线出现「上升沿」= 抓握起始时刻 → 用它给帧打标签。

本脚本验证:
  1. 闭合度信号是否干净可测
  2. 能否在 UNABLE 检测到明确的事件（上升沿）
  3. 事件数量是否与 episode 数吻合（每个 episode 一次抓取 = 一个事件）
"""
import math
import sys

import numpy as np
import pandas as pd

PARQ = "/mnt/storage/siat_cjh/proj/hoi_data/egodex_ep0.parquet"
FINGERS = ["Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"]


def to_mat(x):
    return np.vstack([np.asarray(r, dtype=float).ravel() for r in x])


def main():
    df = pd.read_parquet(PARQ)
    print(f"数据: {len(df)} 帧, {df['episode_index'].nunique()} 个 episode, "
          f"{df['task_index'].nunique()} 个任务")

    for side in ["right", "left"]:
        palm = np.stack([to_mat(x)[:3, 3] for x in df[f"observation.state.{side}Hand"].values])
        wrist = np.stack([to_mat(x)[:3, 3] for x in df[f"observation.state.{side}Forearm"].values])
        tips = {}
        for f in FINGERS:
            try:
                tips[f] = np.stack([to_mat(x)[:3, 3]
                                    for x in df[f"observation.state.{side}{f}Tip"].values])
            except KeyError:
                pass
        if not tips:
            print(f"  [{side}] 无指尖字段"); continue

        # 手部尺度的归一化基准：腕 → 中指掌骨 (若缺失用腕-掌距离)
        scale = np.linalg.norm(palm - wrist, axis=1) + 1e-6
        # 闭合度 = -(指尖到掌心平均距离)/尺度  →  越大=越闭合
        d = np.mean([np.linalg.norm(t - palm, axis=1) for t in tips.values()], axis=0)
        closure = -d / scale

        valid = ~np.isnan(closure) & (scale > 0.02)
        c = closure.copy(); c[~valid] = np.nan
        print(f"\n=== {side} 手 ===")
        print(f"  闭合度范围: {np.nanmin(c):.2f} ~ {np.nanmax(c):.2f} "
              f"(有效帧 {valid.sum()}/{len(c)})")

        # 每个 episode 内做检测：找闭合度从低到高的显著上升沿
        n_ev = 0
        for ep, g in df.groupby("episode_index"):
            idx = g.index.values
            cc = c[idx]
            if np.all(np.isnan(cc)):
                continue
            lo, hi = np.nanpercentile(cc, 10), np.nanpercentile(cc, 90)
            if hi - lo < 0.15:          # 变化太小，无抓取动作
                continue
            thr = lo + 0.5 * (hi - lo)  # 阈值：低30%与高30%的中点
            open_ = cc < thr
            # 上升沿：open_ 从 True 变 False
            edges = np.where(open_[:-1] & ~open_[1:])[0]
            if len(edges):
                n_ev += 1
        print(f"  检出「抓握起始」事件的 episode 数: {n_ev} / {df['episode_index'].nunique()}")

    # 导出闭合度曲线供画图
    np.save("/mnt/storage/siat_cjh/proj/hoi_data/h1_closure_right.npy", c)
    print("\n已导出闭合度曲线 h1_closure_right.npy")


if __name__ == "__main__":
    main()
