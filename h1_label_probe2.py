"""h1_label_probe2.py — 改进版：用「持续闭合」检测抓握事件，导出 H1 三态标签

改进点（相对 v1）:
  1. 用【每段自己的 P5~P95 范围】做相对阈值（而非固定 0.15 跨度）
  2. 要求「连续 N 帧闭合」才算抓到（避免抖动误触发）
  3. 输出 H1 三态标签 + 各类帧数统计（判断是否类别不平衡）
"""
import numpy as np
import pandas as pd

PARQ = "/mnt/storage/siat_cjh/proj/hoi_data/egodex_ep0.parquet"
FINGERS = ["Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"]
SUSTAIN = 5          # 连续闭合帧数
NEAR_MARGIN = 15     # 事件前多少帧算「接近中」


def to_mat(x):
    return np.vstack([np.asarray(r, dtype=float).ravel() for r in x])


def main():
    df = pd.read_parquet(PARQ)
    palm = np.stack([to_mat(x)[:3, 3] for x in df["observation.state.rightHand"].values])
    wrist = np.stack([to_mat(x)[:3, 3] for x in df["observation.state.rightForearm"].values])
    tips = {f: np.stack([to_mat(x)[:3, 3]
                         for x in df[f"observation.state.right{f}Tip"].values]) for f in FINGERS}
    scale = np.linalg.norm(palm - wrist, axis=1) + 1e-6
    closure = -np.mean([np.linalg.norm(t - palm, axis=1) for t in tips.values()], axis=0) / scale

    labels = np.full(len(df), -1, dtype=int)      # -1 未判定
    rows = []
    n_ok = 0
    for ep, g in df.groupby("episode_index"):
        idx = g.index.values
        c = closure[idx]
        p5, p95 = np.percentile(c, 5), np.percentile(c, 95)
        if p95 - p5 < 0.05:
            rows.append((ep, len(idx), "变化过小", 0)); continue
        thr = p5 + 0.35 * (p95 - p5)
        closed = c > thr
        # 找第一个「连续 SUSTAIN 帧闭合」的位置 = 抓握起始事件
        onset = None
        run = 0
        for i, v in enumerate(closed):
            run = run + 1 if v else 0
            if run >= SUSTAIN:
                onset = i - SUSTAIN + 1
                break
        if onset is None:
            rows.append((ep, len(idx), "未检出", 0)); continue
        n_ok += 1
        # 三态标签
        lab = np.zeros(len(idx), dtype=int)            # 0=远（未接近）
        lo = max(0, onset - NEAR_MARGIN)
        lab[lo:onset] = 1                               # 1=接近中
        lab[onset:] = 2                                 # 2=已接触
        labels[idx] = lab
        rows.append((ep, len(idx), "OK", onset))

    print(f"{'ep':>4} {'帧':>5} {'状态':>8} {'事件帧':>7}")
    for ep, n, st, onset in rows:
        print(f"{ep:>4} {n:>5} {st:>8} {onset:>7}")
    print(f"\n成功检出事件: {n_ok}/{df['episode_index'].nunique()} 个 episode")

    ok = labels >= 0
    print(f"\n=== H1 三态标签统计（{ok.sum()} 帧）===")
    names = {0: "远（未接近）", 1: "接近中", 2: "已接触（闭合）"}
    for k in [0, 1, 2]:
        n = int((labels == k).sum())
        print(f"  {names[k]:<14} {n:>5} 帧  ({100*n/max(1,ok.sum()):.1f}%)")
    np.save("/mnt/storage/siat_cjh/proj/hoi_data/h1_labels.npy", labels)
    print("\n已导出 h1_labels.npy（-1=未判定）")


if __name__ == "__main__":
    main()
