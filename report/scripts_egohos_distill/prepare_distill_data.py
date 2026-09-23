"""prepare_distill_data.py — 把 EgoHOS 的伪标签整理成「学生训练」可用格式

EgoHOS 输出（每段视频一个目录）：
    <stem>/images/*.jpg          原图
    <stem>/pred_twohands/*.png   手分割  0=背景 1=左手 2=右手
    <stem>/pred_cb/*.png         接触    0=背景 1=接触区域
    <stem>/pred_obj1/*.png       物体    0=背景 3=一级交互物体

输出（学生训练用）：
    <out>/frames/<split>_<stem>_<idx>.jpg
    <out>/mask_hands/...   mask_contact/...   mask_object/...
    <out>/meta.json         每帧统计 + 过滤原因
    <out>/stats.json        总体统计

过滤规则（伪标签有噪声，必须筛）：
    - 必须检测到手（pred_twohands 非空）
    - 物体/接触是否为空，作为「状态」保留而不是丢弃（无接触也是有效样本！）
    - 完全空帧（连手都没有）丢弃
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def frame_ids(stem_dir: Path):
    files = sorted(glob.glob(str(stem_dir / "images" / "*.jpg")))
    return [Path(f).stem for f in files]


def load_mask(path: Path, shape=None):
    if not path.exists():
        return None
    a = np.array(Image.open(path))
    if shape is not None and a.shape[:2] != shape:
        a = np.array(Image.fromarray(a).resize((shape[1], shape[0]), Image.NEAREST))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego-root", default="/mnt/storage/siat_cjh/proj/EgoHOS/testvideos",
                    help="EgoHOS 输出根目录（含 <stem>/ 子目录）")
    ap.add_argument("--out", default="/mnt/storage/siat_cjh/proj/EgoHOS_distill/data")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="手动指定 split，格式 stem=train 或 stem=val；默认按顺序 1:1")
    ap.add_argument("--min-hand-px", type=int, default=200,
                    help="手的最小像素数（少于此视为检测失败，丢弃）")
    ap.add_argument("--copy", action="store_true", help="复制图片而非建符号链接")
    ap.add_argument("--limit", type=int, default=0, help="每段视频最多取多少帧（0=全部）")
    args = ap.parse_args()

    root = Path(args.ego_root)
    out = Path(args.out)
    stems = sorted([d.name for d in root.iterdir()
                    if d.is_dir() and (d / "images").exists()])
    if not stems:
        print(f"[错误] {root} 下没找到 EgoHOS 输出目录")
        return

    # 划分 split（避免同一视频同时出现在 train/val）
    if args.splits:
        m = dict(s.split("=") for s in args.splits)
    else:
        m = {s: ("train" if i == 0 else "val") for i, s in enumerate(stems)}

    for sub in ["frames", "mask_hands", "mask_contact", "mask_object"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    meta, stats = [], Counter()
    for stem in stems:
        split = m.get(stem, "val")
        sd = root / stem
        ids = frame_ids(sd)
        if args.limit:
            ids = ids[: args.limit]
        kept = 0
        for fid in ids:
            img_p = sd / "images" / f"{fid}.jpg"
            img = Image.open(img_p)
            W, H = img.size
            th = load_mask(sd / "pred_twohands" / f"{fid}.png", (H, W))
            cb = load_mask(sd / "pred_cb" / f"{fid}.png", (H, W))
            ob = load_mask(sd / "pred_obj1" / f"{fid}.png", (H, W))
            if th is None:
                stats["drop_no_hand_mask"] += 1
                continue
            n_hand = int((th > 0).sum())
            if n_hand < args.min_hand_px:
                stats["drop_hand_too_small"] += 1
                continue
            n_contact = int((cb > 0).sum()) if cb is not None else 0
            n_obj = int((ob == 3).sum()) if ob is not None else 0

            name = f"{split}_{stem}_{fid}"
            dst_img = out / "frames" / f"{name}.jpg"
            if args.copy:
                shutil.copy2(img_p, dst_img)
            else:
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                os.symlink(img_p.resolve(), dst_img)
            Image.fromarray(th.astype(np.uint8)).save(out / "mask_hands" / f"{name}.png")
            if cb is not None:
                Image.fromarray(cb.astype(np.uint8)).save(out / "mask_contact" / f"{name}.png")
            if ob is not None:
                Image.fromarray(ob.astype(np.uint8)).save(out / "mask_object" / f"{name}.png")

            rec = dict(name=name, split=split, stem=stem, frame=fid,
                       h=H, w=W, n_hand=n_hand, n_contact=n_contact, n_object=n_obj,
                       left_px=int((th == 1).sum()), right_px=int((th == 2).sum()),
                       has_contact=n_contact > 0, has_object=n_obj > 0)
            meta.append(rec)
            stats[f"keep_{split}"] += 1
            stats["keep_contact" if rec["has_contact"] else "keep_no_contact"] += 1
            stats["keep_object" if rec["has_object"] else "keep_no_object"] += 1
            kept += 1

        print(f"[{stem}] split={split} 帧={len(ids)} 保留={kept}")

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(dict(stats), f, ensure_ascii=False, indent=2)

    print("\n=== 统计 ===")
    for k, v in sorted(stats.items()):
        print(f"  {k:<22} {v}")
    n = len(meta)
    if n:
        print(f"\n  总保留 {n} 帧")
        print(f"  有接触 {stats['keep_contact']} ({100*stats['keep_contact']/n:.1f}%)")
        print(f"  有物体 {stats['keep_object']} ({100*stats['keep_object']/n:.1f}%)")
        print(f"  接触+物体都有: {sum(1 for r in meta if r['has_contact'] and r['has_object'])}")
    print(f"\n输出目录: {out.resolve()}")


if __name__ == "__main__":
    main()
