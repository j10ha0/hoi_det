"""object_axis_pca.py — 从物体掩膜用 PCA 求「物体轴向」与「目标掌心方向」

流程:
  掩膜 → PCA → 主轴 → (特征值比判断是否真有主轴) → 目标方向（垂直于主轴、指向物体）

目标方向的取法（关键）:
  设主轴单位向量为 a，候选方向 = ±a⊥（a 旋转 90°）
  选【与 (物体中心 − 手心方向) 点积为正】的那一个
  → 即"掌心应朝物体所在的那一侧"

用法:
  python object_axis_pca.py                 # 合成演示（含球体拒绝）
  python object_axis_pca.py --img X.jpg --box x1 y1 x2 y2   # 真实图 + 框内 GrabCut 分割
"""
import argparse
import os

import cv2
import numpy as np

OUT = "/mnt/storage/siat_cjh/proj/hoi_data"


def pca_axis(mask):
    """对掩膜像素做 PCA，返回 (中心, 主轴, 特征值, 是否细长)"""
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    c = pts.mean(axis=0)
    X = pts - c
    cov = X.T @ X / len(X)
    w, v = np.linalg.eigh(cov)              # 升序
    major = v[:, 1]                          # 主轴（最大特征值对应）
    ratio = float(w[0] / (w[1] + 1e-9))      # 次/主 → 越接近 1 越"圆"
    elongate = ratio < 0.45                  # 经验阈值：细长
    return c, major, np.sqrt(w), elongate


def target_dir_from_axis(major, obj_center, hand_pt):
    """目标掌心方向 = 垂直于主轴、且指向物体那一侧"""
    perp = np.array([-major[1], major[0]])
    to_obj = obj_center - hand_pt
    if np.dot(perp, to_obj) < 0:
        perp = -perp
    return perp / (np.linalg.norm(perp) + 1e-9)


def draw(canvas, mask, hand_pt, name, color=(0, 200, 255)):
    r = pca_axis(mask)
    if r is None:
        print(f"  {name:<14} 掩膜太小")
        return
    c, major, sv, elong = r
    area = int(mask.sum())
    evr = float(sv[0] ** 2 / (sv[1] ** 2 + 1e-9))     # 特征值比（用于判定）
    if not elong:
        print(f"  {name:<14} 面积 {area:>6}  奇异值 {sv[1]:6.1f}/{sv[0]:6.1f}  "
              f"特征值比 {evr:.2f}  → ⭕ 无明显主轴（类球）→ **跳过对准**")
        cv2.circle(canvas, tuple(c.astype(int)), 16, (0, 0, 255), 3)
        cv2.putText(canvas, f"{name}: no axis (sphere-like)", (int(c[0]) + 20, int(c[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return
    tgt = target_dir_from_axis(major, c, np.array(hand_pt, dtype=float))
    ang = np.degrees(np.arctan2(major[1], major[0]))
    print(f"  {name:<14} 面积 {area:>6}  奇异值 {sv[1]:6.1f}/{sv[0]:6.1f}  特征值比 {evr:.2f}"
          f"  → ✅ 主轴 {ang:7.1f}°")
    L = 130
    p0 = tuple(c.astype(int))
    # 主轴（青）
    p1 = (int(c[0] + major[0] * L), int(c[1] + major[1] * L))
    p2 = (int(c[0] - major[0] * L), int(c[1] - major[1] * L))
    cv2.line(canvas, p1, p2, (255, 255, 0), 3)
    # 目标掌心方向（品红，垂直于主轴）
    p3 = (int(c[0] + tgt[0] * L), int(c[1] + tgt[1] * L))
    cv2.arrowedLine(canvas, p0, p3, (255, 0, 255), 4, tipLength=0.2)
    cv2.putText(canvas, "target palm normal", (p3[0], p3[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 0, 255), 2)
    cv2.putText(canvas, name, (p0[0] - 40, p0[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def synthetic_demo():
    """合成演示：4 种形状，含球体自动拒绝"""
    W, H = 1100, 700
    img = np.full((H, W, 3), 235, np.uint8)
    cv2.rectangle(img, (0, 520), (W, H), (205, 205, 200), -1)      # 桌面
    masks, names = [], []

    # ① 圆柱（细长矩形）
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (80, 300), (400, 360), 255, -1); masks.append(m); names.append("cylinder")
    # ② 球（圆）
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (520, 330), 55, 255, -1); masks.append(m); names.append("sphere")
    # ③ 薄片（薄而宽的矩形，斜放）
    m = np.zeros((H, W), np.uint8)
    box = cv2.boxPoints(((720, 320), (240, 46), 25))
    cv2.fillPoly(m, [box.astype(np.int32)], 255); masks.append(m); names.append("plate")
    # ④ 棱柱（斜矩形）
    m = np.zeros((H, W), np.uint8)
    box = cv2.boxPoints(((940, 330), (170, 90), -35))
    cv2.fillPoly(m, [box.astype(np.int32)], 255); masks.append(m); names.append("prism")

    hand = (520, 640)          # 假设手心在下方中央（模拟头戴俯视）
    print("=== 合成演示：PCA 求物体轴向 ===")
    for m, n in zip(masks, names):
        draw(img, m > 0, hand, n, (0, 0, 0))
    cv2.circle(img, hand, 14, (0, 0, 0), -1)
    cv2.putText(img, "hand (palm center)", (hand[0] + 20, hand[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "cyan = object principal axis | magenta = target palm normal",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    p = f"{OUT}/pca_synthetic.jpg"
    cv2.imwrite(p, img)
    print(f"已保存 {p}")


def real_demo(img_path, box):
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    print(f"=== 真实图: {W}x{H}  框={box} ===")
    mask = np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    x1, y1, x2, y2 = box
    cv2.grabCut(img, mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    print(f"  GrabCut 前景区块: {int((m>0).sum())} 像素")
    vis = img.copy()
    vis[m > 0] = (vis[m > 0] * 0.55 + np.array([0, 255, 255]) * 0.45).astype(np.uint8)
    hand = ((x1 + x2) // 2, min(H - 6, y2 + 120))
    draw(vis, m > 0, hand, "object")
    p = f"{OUT}/pca_real.jpg"
    cv2.imwrite(p, vis)
    print(f"已保存 {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=None)
    ap.add_argument("--box", type=int, nargs=4, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    synthetic_demo()
    if a.img and a.box:
        real_demo(a.img, a.box)
