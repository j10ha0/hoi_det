#!/usr/bin/env python3
"""pick_grasp_frames.py — 从 EgoTactile 挑「抓握中」的帧（压力高）

抓握类型只在【握持中】才有定义 → 用压力标注挑高压力帧
每个物体取 5 帧（分布在不同的抓握循环里，增加多样性）
"""
import glob
import json
import os
import subprocess

import numpy as np

BASE = "/mnt/storage/siat_cjh/proj/hoi_data/egotactile"
FF = "/mnt/storage/siat_cjh/conda_envs/egohos/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2"
HI = 500.0          # 高压力阈值 = 抓握中
N_PER_OBJ = 5


def pressure(entry):
    s = 0.0
    for h in ("RH", "LH"):
        if h in entry and "sensor_256" in entry[h]:
            try:
                s += float(np.sum(np.asarray(entry[h]["sensor_256"], dtype=float)))
            except Exception:
                pass
    return s


def main():
    outdir = f"{BASE}/grasp_frames"
    os.makedirs(outdir, exist_ok=True)
    for f in glob.glob(f"{outdir}/*.jpg"):
        os.remove(f)

    objs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{BASE}/*.mp4"))
    picks = []
    print(f"{'物体':<34} {'抓握条目':>8} {'选取帧':>28}")
    for o in objs:
        jp = f"{BASE}/{o}.json"
        if not os.path.exists(jp) or os.path.getsize(jp) < 1000:
            continue
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        t = np.array([pressure(e) for e in d])
        ct = np.array([e.get("camera_timestamp", 0.0) for e in d])
        hi = np.nonzero(t > HI)[0]
        if len(hi) == 0:
            print(f"{o:<34} {0:>8} 无抓握段"); continue
        # 取若干高压力条目（按时间均匀采样）
        sel = hi[np.linspace(0, len(hi) - 1, min(N_PER_OBJ, len(hi))).astype(int)]
        frames = [int(round(float(ct[i] - ct[0]) * 30.0)) for i in sel]
        picks.append((o, frames))
        print(f"{o:<34} {len(hi):>8} {str(frames):>28}")

    n = 0
    for o, frames in picks:
        for k, fr in enumerate(frames):
            subprocess.run([FF, "-v", "error", "-ss", f"{fr/30:.3f}", "-i", f"{BASE}/{o}.mp4",
                            "-frames:v", "1", "-q:v", "2",
                            f"{outdir}/{o}_g{k}.jpg", "-y"], check=False)
            n += 1
    got = len(glob.glob(f"{outdir}/*.jpg"))
    print(f"\n抽出 {got} 张抓握帧 → {outdir}")


if __name__ == "__main__":
    main()
