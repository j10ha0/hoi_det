"""align_label_probe.py — 验证「对准误差」标签能否从公开数据（EgoDex）造出来

核心思路（⭐ 不需要物体位姿！）:
    对准误差的【自监督代理】= 抓握时刻的腕旋角 − 当前腕旋角
    即："距离最终抓握姿态还差多少旋转"
    → 只需【手部姿态】，EgoDex 就有 → 公开数据可用！

步骤:
  1. 逐帧算 手相对前臂 的相对旋转 R_rel = R_forearm^T @ R_hand
  2. 分解出 3 个轴上的旋转分量，看哪个轴上"抓握过程中变化最大" = 腕旋轴
  3. 用「手指闭合度」找抓握时刻 → 取该时刻的腕旋角作为"目标"
  4. ê(t) = 目标 − 当前（wrap 到 ±180°）
  5. 统计:  分布 / 各类占比（判断是否类别不平衡）
"""
import numpy as np
import pandas as pd

PARQ = "/mnt/storage/siat_cjh/proj/hoi_data/egodex_ep0.parquet"
FINGERS = ["Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"]
SUSTAIN, NEAR = 5, 15


def to_mat(x):
    return np.vstack([np.asarray(r, dtype=float).ravel() for r in x])


def euler_xyz(R):
    """返回绕 x/y/z 的三个旋转角（度）"""
    sy = -R[2, 0]
    sy = np.clip(sy, -1, 1)
    ry = np.arcsin(sy)
    rx = np.arctan2(R[2, 1], R[2, 2])
    rz = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([rx, ry, rz])


def wrap180(a):
    return (a + 180) % 360 - 180


def main():
    df = pd.read_parquet(PARQ)
    side = "right"
    R_h = [to_mat(x)[:3, :3] for x in df[f"observation.state.{side}Hand"].values]
    R_f = [to_mat(x)[:3, :3] for x in df[f"observation.state.{side}Forearm"].values]
    palm = np.stack([to_mat(x)[:3, 3] for x in df[f"observation.state.{side}Hand"].values])
    tips = {f: np.stack([to_mat(x)[:3, 3] for x in df[f"observation.state.{side}{f}Tip"].values])
            for f in FINGERS}
    scale = np.linalg.norm(
        palm - np.stack([to_mat(x)[:3, 3] for x in df[f"observation.state.{side}Forearm"].values]),
        axis=1) + 1e-6
    closure = -np.mean([np.linalg.norm(t - palm, axis=1) for t in tips.values()], axis=0) / scale

    # 1) 相对旋转的三个欧拉分量
    eul = np.array([euler_xyz(R_f[i].T @ R_h[i]) for i in range(len(df))])
    print("=== 相对旋转三分量的全局变化幅度（标准差，度）===")
    for k, name in enumerate(["roll(绕x)", "pitch(绕y)", "yaw(绕z)"]):
        print(f"  {name:<12} std={eul[:,k].std():6.2f}  范围 "
              f"{eul[:,k].min():7.1f} ~ {eul[:,k].max():7.1f}")

    # 2) 每个 episode 里，从"张开"到"抓握"过程中变化最大的分量 = 腕旋轴
    changes = np.zeros(3)
    targets = np.full(len(df), np.nan)
    for ep, g in df.groupby("episode_index"):
        idx = g.index.values
        c = closure[idx]
        p5, p95 = np.percentile(c, 5), np.percentile(c, 95)
        if p95 - p5 < 0.05:
            continue
        closed = c > (p5 + 0.35 * (p95 - p5))
        onset = None
        run = 0
        for i, v in enumerate(closed):
            run = run + 1 if v else 0
            if run >= SUSTAIN:
                onset = i - SUSTAIN + 1
                break
        if onset is None:
            continue
        # 该 episode 内三分量从开→闭的变化
        open_part = eul[idx[:max(1, onset)]][:, :]
        closed_part = eul[idx[onset:]]
        if len(closed_part) < 5:
            continue
        d = np.abs(np.median(closed_part, axis=0) - np.median(open_part, axis=0))
        changes += np.minimum(d, 360 - d)      # 环形距离
        # 目标腕旋 = 抓握期间该分量的中位数
        targets[idx] = np.median(closed_part, axis=0)[0]   # 先用第 0 分量（roll）

    print("\n=== 抓握过程中各分量变化（累计，判断哪个是腕旋轴）===")
    for k, name in enumerate(["roll(绕x)", "pitch(绕y)", "yaw(绕z)"]):
        print(f"  {name:<12} 累计变化 {changes[k]:8.1f}")

    axis = int(np.argmax(changes))
    print(f"\n→ 推断的腕旋轴: 第 {axis} 分量")

    # 3) 用推断出的轴重算标签
    e = eul[:, axis]
    tgt = np.full(len(df), np.nan)
    for ep, g in df.groupby("episode_index"):
        idx = g.index.values
        c = closure[idx]
        p5, p95 = np.percentile(c, 5), np.percentile(c, 95)
        if p95 - p5 < 0.05:
            continue
        closed = c > (p5 + 0.35 * (p95 - p5))
        onset, run = None, 0
        for i, v in enumerate(closed):
            run = run + 1 if v else 0
            if run >= SUSTAIN:
                onset = i - SUSTAIN + 1
                break
        if onset is None:
            continue
        seg = e[idx[onset:]]
        if len(seg) >= 5:
            tgt[idx] = np.median(seg)

    err = wrap180(e - tgt)
    ok = ~np.isnan(err)
    print(f"\n=== 对准误差 ê 分布（{ok.sum()} 帧）===")
    qs = [1, 5, 25, 50, 75, 95, 99]
    print("  分位: " + "  ".join(f"P{q}={np.nanpercentile(err[ok],q):6.1f}" for q in qs))
    print(f"  |ê|>30° 的帧: {int((np.abs(err[ok])>30).sum())} ({100*(np.abs(err[ok])>30).mean():.1f}%)")
    print(f"  |ê|>60° 的帧: {int((np.abs(err[ok])>60).sum())} ({100*(np.abs(err[ok])>60).mean():.1f}%)")

    # 4) 三分类标签
    TH = 15
    lab = np.where(err > TH, 0, np.where(err < -TH, 1, 2))   # 0=须旋前向, 1=须反向, 2=已对准
    names = ["向 +（一类旋转）", "向 −（另一类旋转）", "已对准"]
    print(f"\n=== 三分类标签（阈值 ±{TH}°）===")
    for k, n in enumerate(names):
        cnt = int((lab[ok] == k).sum())
        print(f"  {n:<18} {cnt:>5} ({100*cnt/ok.sum():5.1f}%)")
    np.save("/mnt/storage/siat_cjh/proj/hoi_data/align_err.npy", err)


if __name__ == "__main__":
    main()
