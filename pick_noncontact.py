#!/usr/bin/env python3
"""pick_noncontact.py — 用 EgoTactile 的压力标注挑出「手未接触物体」的帧

思路（正是你要的场景）:
  data.json 逐帧记录 162 个压力传感器 → 压力≈0 = 未接触
  找出「首次接触」的条目 → 取它之前的视频帧 = 手正在接近但未接触

json 与视频的采样率不同（压力 ~17Hz，视频 30fps）→ 用条目数线性映射
"""
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

BASE = "/mnt/storage/siat_cjh/proj/hoi_data/egotactile"
FF = "/mnt/storage/siat_cjh/conda_envs/egohos/lib/python3.8/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2"
HF = "https://huggingface.co/datasets/HustleHard/EgoTactile/resolve/main/Raw_data/bare_hand/p001"


def total_pressure(entry):
    s = 0.0
    for hand in ("RH", "LH"):
        if hand in entry:
            for k, v in entry[hand].items():
                try:
                    s += float(np.sum(np.asarray(v, dtype=float)))
                except (ValueError, TypeError):
                    pass
    return s


def video_frames(path):
    r = subprocess.run([FF, "-i", path], capture_output=True, text=True, errors="ignore")
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0
    return dur * 30.0


def main():
    objs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{BASE}/*.mp4"))
    print(f"处理 {len(objs)} 个物体\n")
    print(f"{'物体':<34} {'json条目':>8} {'首触条目':>8} {'视频帧估':>8} {'选取帧':>7} {'选取秒':>7}")
    picks = []
    for o in objs:
        jp = f"{BASE}/{o}.json"
        if not os.path.exists(jp):
            subprocess.run(["curl", "-sL", "--max-time", "30", "-o", jp,
                            f"{HF}/{o}/repeat0000/data.json"], check=False)
        if not os.path.exists(jp) or os.path.getsize(jp) < 1000:
            print(f"{o:<34} ✗ 无标注"); continue
        try:
            d = json.load(open(jp))
        except Exception as e:
            print(f"{o:<34} ✗ 解析失败 {e}"); continue
        t = np.array([total_pressure(e) for e in d])
        mp4 = f"{BASE}/{o}.mp4"
        nvf = video_frames(mp4) if os.path.exists(mp4) else 800
        ratio = nvf / max(1, len(d))
        nz = np.nonzero(t > 5)[0]
        if len(nz) == 0:
            print(f"{o:<34} {len(d):>8} {'无接触':>8}"); continue
        first = int(nz[0])
        vf_contact = int(first * ratio)
        pick = max(0, vf_contact - 20)           # 接触前 20 帧（约 0.7s）
        picks.append((o, pick))
        print(f"{o:<34} {len(d):>8} {first:>8} {vf_contact:>8} {pick:>7} {pick/30:>7.1f}")

    # 抽帧
    outdir = f"{BASE}/nc_frames"
    os.makedirs(outdir, exist_ok=True)
    for o, fr in picks:
        t = fr / 30.0
        subprocess.run([FF, "-v", "error", "-ss", f"{t:.3f}", "-i", f"{BASE}/{o}.mp4",
                        "-frames:v", "1", "-q:v", "2", f"{outdir}/{o}_nc.jpg", "-y"], check=False)
    print(f"\n抽取了 {len(os.listdir(outdir))} 张「非接触」帧 → {outdir}")


if __name__ == "__main__":
    main()
