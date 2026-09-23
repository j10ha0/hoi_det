"""object_axis_realdemo.py — 真实图：FastSAM 全图分割 → 每个掩膜 PCA → 轴向/目标方向

流程（这就是你部署时的实际流水线）:
  1. FastSAM 全图分割（everything 模式）→ 一堆候选掩膜
  2. 过滤（面积过大/过小、长宽比）
  3. 每个掩膜做 PCA → 主轴 + 特征值比
  4. 特征值比 → 有主轴（细长）还是无主轴（类球）
  5. 画出来
"""
import glob
import os

import cv2
import numpy as np

OUT = "/mnt/storage/siat_cjh/proj/hoi_data"


def pca(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    X = pts - c
    cov = X.T @ X / len(X)
    w, v = np.linalg.eigh(cov)
    sv = np.sqrt(np.maximum(w, 0))
    evr = float(w[0] / (w[1] + 1e-9))
    return c, v[:, 1], sv, evr


def main():
    from ultralytics import FastSAM
    dev = "cuda:0"
    model = FastSAM("FastSAM-s.pt")

    for img_path in ["gt_frames/n00300.jpg", "gt_frames/n01200.jpg"]:
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        res = model(img, device=dev, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, verbose=False)[0]
        if res.masks is None:
            print(f"{img_path}: 未分割出任何区域")
            continue
        masks = res.masks.data.cpu().numpy()
        print(f"\n=== {os.path.basename(img_path)} ({W}x{H}) 分割出 {len(masks)} 个区域 ===")
        vis = img.copy()
        n_axis = n_circ = 0
        for k, m in enumerate(masks):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            area = int(m.sum())
            if area < 1500 or area > 0.5 * W * H:      # 过滤太小/太大
                continue
            r = pca(m)
            if r is None:
                continue
            c, major, sv, evr = r
            elongate = evr < 0.45
            color = (255, 255, 0) if elongate else (0, 0, 255)
            L = 0.5 * sv[1]
            p1 = (int(c[0] + major[0] * L), int(c[1] + major[1] * L))
            p2 = (int(c[0] - major[0] * L), int(c[1] - major[1] * L))
            cv2.line(vis, p1, p2, color, 4)
            cv2.circle(vis, tuple(c.astype(int)), 8, color, -1)
            if elongate:
                n_axis += 1
                ang = np.degrees(np.arctan2(major[1], major[0]))
                print(f"  区域{k:>2} 面积{area:>7} 特征值比 {evr:.2f} → ✅ 有主轴 {ang:6.1f}°")
            else:
                n_circ += 1
                print(f"  区域{k:>2} 面积{area:>7} 特征值比 {evr:.2f} → ⭕ 类球/方形，跳过")
        print(f"  小结: 有主轴 {n_axis} 个，类球 {n_circ} 个")
        cv2.putText(vis, "cyan = has principal axis | red = axis-less (sphere-like)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        p = f"{OUT}/pca_fastsam_{os.path.basename(img_path)[:-4]}.jpg"
        cv2.imwrite(p, vis)
        print(f"  已保存 {p}")


if __name__ == "__main__":
    main()
