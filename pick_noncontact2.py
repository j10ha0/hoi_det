#!/usr/bin/env python3
"""pick_noncontact2.py — 用压力标注挑「手接近但未接触」的帧（修正版）

修正：
  · 只累加 RH/LH['sensor_256']（原来把 timestamp/anomaly 也算进去了）
  · 压力值在低(~50)与高(~700)间振荡 → 一个视频含多次抓握循环
  · 非接触帧 = 压力低；取「低压→高压」跃变【之前】若干帧 = 正在接近
"""
import glob
import json
import os
import re
import subprocess

import numpy as np

BASE = "/mnt/storage/siat_cjh/proj/hoi_data/egotactile"
FF = "/mnt/storage/siat_cjh/conda_envs/egohos/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2"
LO, HI = 200.0, 400.0          # 低于 LO=未接触；高于 HI=已抓握
PRE_ENTRIES = 6                # 跃变前取几条（≈0.35s）


def pressure(entry):
    s = 0.0
    for h in ("RH", "LH"):
        if h in entry and "sensor_256" in entry[h]:
            try:
                s += float(np.sum(np.asarray(entry[h]["sensor_256"], dtype=float)))
            except Exception:
                pass
    return s


def duration(path):
    r = subprocess.run([FF, "-i", path], capture_output=True, text=True, errors="ignore")
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        return 25.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def main():
    objs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{BASE}/*.mp4"))
    outdir = f"{BASE}/nc_frames2"
    os.makedirs(outdir, exist_ok=True)
    for f in glob.glob(f"{outdir}/*.jpg"):
        os.remove(f)

    picks = []
    print(f"{'物体':<34} {'条目':>5} {'非接触条目':>10} {'跃变条目':>8} {'选取帧':>7} {'秒':>6}")
    for o in objs:
        jp = f"{BASE}/{o}.json"
        if not os.path.exists(jp) or os.path.getsize(jp) < 1000:
            continue
        try:
            d = json.load(open(jp))
        except Exception:
            continue
        t = np.array([pressure(e) for e in d])
        low = t < LO
        hi = t > HI
        # 找 低压→高压 跃变
        trans = [i for i in range(1, len(t)) if low[i - 1] and hi[i]]
        if not trans:
            trans = [i for i in range(1, len(t)) if t[i] - t[i - 1] > 300]
        if not trans:
            print(f"{o:<34} {len(t):>5} {int(low.sum()):>10} {'无':>8}"); continue
        k = trans[0]
        j = max(0, k - PRE_ENTRIES)
        ct = np.array([e.get("camera_timestamp", 0.0) for e in d])
        vf = int(round(float(ct[j] - ct[0]) * 30.0))   # 距起点秒数 × 30
        picks.append((o, vf))
        print(f"{o:<34} {len(t):>5} {int(low.sum()):>10} {k:>8} {vf:>7} {vf/30:>6.1f}")

    for o, fr in picks:
        subprocess.run([FF, "-v", "error", "-ss", f"{fr/30:.3f}", "-i", f"{BASE}/{o}.mp4",
                        "-frames:v", "1", "-q:v", "2", f"{outdir}/{o}_nc.jpg", "-y"], check=False)
    print(f"\n抽取 {len(os.listdir(outdir))} 张非接触帧 → {outdir}")


if __name__ == "__main__":
    main()
