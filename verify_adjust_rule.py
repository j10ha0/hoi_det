"""verify_adjust_rule.py — 验证「调整量」算法 + 用 EgoDex 真值检验定义合理性

【算法：从"掌心法向 vs 物体方向"算出"绕轴转多少"】

  手的长轴      f = normalize(middle_MCP − wrist)        （≈ 前臂长轴，腕旋就绕它转）
  掌心法向      n = normalize(cross(idx_MCP−wrist, pinky_MCP−wrist))
  目标方向      u = normalize(p_obj − palm_center)       （需要物体位置）

  ① 把 n、u 投到「垂直于 f」的平面内（只保留腕旋能改的分量）
        a = n − (n·f) f
        b = u − (u·f) f
  ② 两者在该平面内的带符号夹角 = 需要绕长轴转的角度
        θ = atan2( dot(cross(a, b), f), dot(a, b) )
        θ > 0 → 一个方向；θ < 0 → 另一个方向；|θ| = 转多少度

【无法用单自由度腕旋实现的部分 = "残余"】
        residual = 总夹角 − |θ|        （需要腕屈伸/尺桡偏，1-DOF 腕做不到）

【本脚本的验证】EgoDex 没有物体位置，所以用【抓握时刻的掌心法向】当目标
  （人在抓握时掌心必然对着物体 → 该时刻的法向就是"对准"的目标）
  然后看：接近抓握的过程中，|θ| 是否单调减小到 ~0
  → 若是，则"掌心法向对准"这个定义成立
"""
import numpy as np
import pandas as pd

PARQ = "/mnt/storage/siat_cjh/proj/hoi_data/egodex_ep0.parquet"
FINGERS = ["Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"]
SUSTAIN = 5
PRE = 90          # 只看抓握前 90 帧（3 秒）


def to_mat(x):
    return np.vstack([np.asarray(r, dtype=float).ravel() for r in x])


def main():
    df = pd.read_parquet(PARQ)
    side = "right"
    P = {j: np.stack([to_mat(x)[:3, 3] for x in df[f"observation.state.{side}{j}"].values])
         for j in ["Hand", "Forearm", "IndexFingerKnuckle", "MiddleFingerKnuckle",
                   "PinkyFingerKnuckle" if f"observation.state.{side}PinkyFingerKnuckle" in df.columns
                   else "LittleFingerKnuckle"]}
    wrist = P["Hand"]                     # MANO: root ≈ 腕
    idx = P["IndexFingerKnuckle"]
    mid = P["MiddleFingerKnuckle"]
    pnk = P["PinkyFingerKnuckle"] if "PinkyFingerKnuckle" in P else P["LittleFingerKnuckle"]
    print("关节字段:", list(P.keys()))

    # 掌心法向 n 与手长轴 f（逐帧）
    n_all = np.cross(idx - wrist, pnk - wrist)
    n_all /= (np.linalg.norm(n_all, axis=1, keepdims=True) + 1e-9)
    f_all = mid - wrist
    f_all /= (np.linalg.norm(f_all, axis=1, keepdims=True) + 1e-9)
    palm_center = (wrist + idx + pnk) / 3.0

    # 手指闭合度 → 找抓握 onset
    tips = {k: np.stack([to_mat(x)[:3, 3]
                         for x in df[f"observation.state.{side}{k}Tip"].values]) for k in FINGERS}
    scale = np.linalg.norm(palm_center - P["Forearm"], axis=1) + 1e-6
    closure = -np.mean([np.linalg.norm(t - palm_center, axis=1) for t in tips.values()],
                       axis=0) / scale

    allT, allTh = [], []
    n_ok = 0
    for ep, g in df.groupby("episode_index"):
        i = g.index.values
        c = closure[i]
        p5, p95 = np.percentile(c, 5), np.percentile(c, 95)
        if p95 - p5 < 0.05:
            continue
        closed = c > (p5 + 0.35 * (p95 - p5))
        onset, run = None, 0
        for k, v in enumerate(closed):
            run = run + 1 if v else 0
            if run >= SUSTAIN:
                onset = k - SUSTAIN + 1
                break
        if onset is None or onset < 20:
            continue
        # 目标：抓握时刻的掌心法向（取 onset 后 20 帧中位数）
        seg = n_all[i[onset:onset + 20]]
        if len(seg) < 5:
            continue
        n_tgt = seg.mean(axis=0)
        n_tgt /= (np.linalg.norm(n_tgt) + 1e-9)

        lo = max(0, onset - PRE)
        for k in range(lo, onset):
            gi = i[k]
            f = f_all[gi]
            a = n_all[gi] - np.dot(n_all[gi], f) * f
            b = n_tgt - np.dot(n_tgt, f) * f
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-6 or nb < 1e-6:
                continue
            a, b = a / na, b / nb
            th = np.degrees(np.arctan2(np.dot(np.cross(a, b), f), np.dot(a, b)))
            allT.append(k - onset)          # 距抓握的帧数（负=抓握前）
            allTh.append(abs(th))
        n_ok += 1

    allT, allTh = np.array(allT), np.array(allTh)
    print(f"\n有效 episode: {n_ok}   样本帧: {len(allTh)}")
    print("=== 距抓握不同时间处的 |θ|（度，绕长轴需转的角度）===")
    for lo, hi in [(-90, -75), (-75, -60), (-60, -45), (-45, -30), (-30, -15), (-15, 0)]:
        m = (allT >= lo) & (allT < hi)
        if m.sum():
            print(f"  抓握前 {abs(lo):>3}~{abs(hi):>3} 帧: |θ| 中位数 {np.median(allTh[m]):5.1f}°  "
                  f"均值 {allTh[m].mean():5.1f}°  (n={m.sum()})")
    print(f"\n  整体: 中位数 {np.median(allTh):.1f}°   抓握前一瞬(-15~0帧) "
          f"{np.median(allTh[(allT >= -15) & (allT < 0)]):.1f}°")
    print("\n  预期：越接近抓握，|θ| 越小（→ 若成立，说明定义合理）")
    np.savez("/mnt/storage/siat_cjh/proj/hoi_data/adjust_curve.npz", t=allT, th=allTh)


if __name__ == "__main__":
    main()
