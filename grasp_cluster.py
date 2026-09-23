"""grasp_cluster.py — 从手型特征聚类出「抓握类型」的合适类数

特征（全部【尺度无关 + 旋转无关】，只用手的内部几何）:
  1-5  : 各手指关节弯曲角  (index/middle/ring/pinky 用 MCP-PIP-DIP；thumb 用 MCP-IP-TIP)
  6-10 : 各指尖到掌心距离 / 掌长
  11   : 拇指尖-食指尖对合距离 / 掌长
  13   : 食指根-小指根张开度 / 掌长
  12   : 指尖两两平均距离 / 掌长
  14   : 平均指尖-腕距离 / 掌长
        ↓ 标准化 → KMeans(k=2..10) → 轮廓系数 / CH 指数 选择最佳 k
"""
import glob
import math
import os

import numpy as np
import torch

for _n, _t in [("bool", bool), ("int", int), ("float", float), ("complex", complex),
               ("object", object), ("unicode", str), ("str", str)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)
np.nan = float("nan"); np.inf = float("inf")

BASE = "/mnt/storage/siat_cjh/proj/hoi_data/egotactile"
FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
           "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
TIPS = [4, 8, 12, 16, 20]
MCPS = [2, 5, 9, 13, 17]


def angle3(a, b, c):
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(v1 / n1, v2 / n2), -1, 1))))


def features(kp):
    """kp: (21,3) 3D 关键点（相机坐标系）"""
    wrist = kp[0]
    mid_mcp = kp[9]
    palm = float(np.linalg.norm(mid_mcp - wrist)) + 1e-9
    palm_center = (kp[0] + kp[5] + kp[17]) / 3.0
    f = []
    # 1-5 手指弯曲角
    for name, (a, b, c, d) in FINGERS.items():
        if name == "thumb":
            f.append(angle3(kp[a], kp[b], kp[d]))       # 拇指 MCP-IP-TIP
        else:
            f.append(angle3(kp[a], kp[b], kp[c]))       # 其余 MCP-PIP-DIP
            f.append(angle3(kp[b], kp[a], kp[c]))       # 辅助：MCP 处
    # 只保留 5 个主弯曲角
    amain = [angle3(kp[FINGERS[n][0]], kp[FINGERS[n][1]], kp[FINGERS[n][3]])
             if n == "thumb" else
             angle3(kp[FINGERS[n][0]], kp[FINGERS[n][1]], kp[FINGERS[n][2]])
             for n in FINGERS]
    feats = list(amain)                                  # 5
    # 6-10 指尖到掌心距离比
    feats += [float(np.linalg.norm(kp[t] - palm_center)) / palm for t in TIPS]
    # 11 拇指尖-食指尖对合
    feats.append(float(np.linalg.norm(kp[4] - kp[8])) / palm)
    # 12 张开度（食指根-小指根）
    feats.append(float(np.linalg.norm(kp[5] - kp[17])) / palm)
    # 13 指尖两两平均距离
    ds = [np.linalg.norm(kp[TIPS[i]] - kp[TIPS[j]])
          for i in range(5) for j in range(i + 1, 5)]
    feats.append(float(np.mean(ds)) / palm)
    # 14 平均指尖-腕距离
    feats.append(float(np.mean([np.linalg.norm(kp[t] - wrist) for t in TIPS])) / palm)
    return np.array(feats, dtype=float)


def main():
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=torch.float32, verbose=False)

    X, meta = [], []
    for f in sorted(glob.glob(f"{BASE}/grasp_frames/*.jpg")):
        obj = os.path.basename(f).split("_g")[0]
        import cv2
        img = cv2.imread(f)
        out = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not out:
            continue
        o = max(out, key=lambda z: (z["hand_bbox"][2] - z["hand_bbox"][0]) *
                                   (z["hand_bbox"][3] - z["hand_bbox"][1]))
        kp = np.asarray(o["wilor_preds"]["pred_keypoints_3d"]).reshape(-1, 3)
        X.append(features(kp)); meta.append(obj)
    X = np.array(X)
    print(f"有效样本: {len(X)}  特征维度: {X.shape[1]}")
    print(f"涉及物体 {len(set(meta))} 类\n")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    Xs = StandardScaler().fit_transform(X)

    print(f"{'k':>3} {'轮廓系数':>10} {'CH指数':>10} {'最小类占比':>10}")
    best = (None, -2)
    res = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
        lab = km.labels_
        sil = silhouette_score(Xs, lab)
        ch = calinski_harabasz_score(Xs, lab)
        cnt = np.bincount(lab, minlength=k)
        frac = cnt.min() / len(lab)
        res.append((k, sil, ch, frac, lab, km))
        print(f"{k:>3} {sil:>10.3f} {ch:>10.1f} {frac:>10.2f}")
        if sil > best[1]:
            best = (k, sil, lab, km)

    k, sil, lab, km = best[0], best[1], best[2], best[3]
    print(f"\n⭐ 轮廓系数最佳的 k = {k}  (silhouette={sil:.3f})")

    print(f"\n=== k={k} 的各簇特征（相对全体均值，+ 偏高 / - 偏低）===")
    names = ["拇指弯曲", "食指弯曲", "中指弯曲", "无名弯曲", "小指弯曲",
             "拇指尖距", "食指尖距", "中指尖距", "无名尖距", "小指尖距",
             "拇-食对合", "张 开 度", "指尖分散", "指尖-腕距"]
    z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    for c in range(k):
        m = lab == c
        prof = z[m].mean(0)
        top = np.argsort(-np.abs(prof))[:5]
        desc = "  ".join(f"{names[i]}{'+' if prof[i] > 0 else '-'}{abs(prof[i]):.1f}" for i in top)
        objs = sorted(set(np.array(meta)[m]))
        print(f"  簇{c} (n={m.sum():>2})  {desc}")
        print(f"         含物体: {', '.join(objs[:8])}")

    # 物体 → 主簇（看是否按物体分组）
    print(f"\n=== 每个物体主要落在哪个簇 ===")
    for obj in sorted(set(meta)):
        li = [lab[i] for i, m in enumerate(meta) if m == obj]
        cnt = np.bincount(li, minlength=k)
        print(f"  {obj:<34} " + " ".join(f"簇{i}:{c}" for i, c in enumerate(cnt) if c))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = [r[0] for r in res]; sils = [r[1] for r in res]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(ks, sils, marker="o"); ax[0].axvline(k, ls="--", color="r")
        ax[0].set_xlabel("k (cluster count)"); ax[0].set_ylabel("silhouette")
        ax[0].set_title("Choose k by silhouette"); ax[0].grid(alpha=.3)
        from sklearn.decomposition import PCA
        P = PCA(2).fit_transform(Xs)
        sc = ax[1].scatter(P[:, 0], P[:, 1], c=lab, cmap="tab10", s=42)
        ax[1].set_title(f"KMeans k={k} (PCA 2D)"); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{BASE}/grasp_cluster_k.png", dpi=140)
        print(f"\n图: {BASE}/grasp_cluster_k.png")
    except Exception as e:
        print(f"(画图失败: {e})")


if __name__ == "__main__":
    main()
