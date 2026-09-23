"""extract_teacher_feats.py — 提取 EgoHOS(Swin-B) 的中间层特征用于蒸馏

为什么用 stage-3（stride 16）：
    ViT-Tiny(patch16) 在 224×224 输入下 → 14×14 token 网格
    Swin-B stage-3 (stride 16) 在 224×224 下 → 14×14 特征图
    → 空间尺寸天然对齐，可直接做特征蒸馏（1×1 卷积对齐通道 + 余弦损失）

输入：data/frames/*.jpg（由 prepare_distill_data.py 生成）
输出：data/teacher_feat.npz
       - names: (N,) 帧名
       - stage3: (N, C3, 14, 14) float16
       - stage4: (N, C4, 7, 7) float16   （可选，深层语义）
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mmseg.apis import init_segmentor

CFG = "/mnt/storage/siat_cjh/proj/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/seg_twohands_ccda.py"
CKPT = "/mnt/storage/siat_cjh/proj/EgoHOS/mmsegmentation/work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/mnt/storage/siat_cjh/proj/EgoHOS_distill/data/frames")
    ap.add_argument("--out", default="/mnt/storage/siat_cjh/proj/EgoHOS_distill/data/teacher_feat.npz")
    ap.add_argument("--size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dev = "cuda"
    H, W = args.size

    files = sorted(glob.glob(str(Path(args.frames) / "*.jpg")))
    names = [Path(f).stem for f in files]
    print(f"帧数: {len(files)}  输入尺寸: {W}x{H}", flush=True)

    model = init_segmentor(CFG, CKPT, device=dev)
    bb = model.backbone                       # SwinTransformer
    bb.eval()
    print(f"backbone: {type(bb).__name__}", flush=True)

    mean, std = MEAN.to(dev), STD.to(dev)
    f2, f3, f4, ok_names = [], [], [], []
    with torch.no_grad():
        # 先探测一次输出结构
        x0 = torch.zeros(1, 3, H, W, device=dev)
        try:
            outs = bb(x0)
            shapes = ([tuple(o.shape) for o in outs] if isinstance(outs, (list, tuple))
                      else tuple(outs.shape))
            print(f"backbone 输出: {shapes}", flush=True)
        except Exception as e:
            print(f"[警告] 直接调用 backbone 失败({type(e).__name__})，改用 hook")
            outs = None

        for i, fp in enumerate(files):
            img = Image.open(fp).convert("RGB").resize((W, H), Image.BILINEAR)
            x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
            x = (x - mean) / std
            o = bb(x)
            if isinstance(o, (list, tuple)):
                s2, s3, s4 = o[1], o[2], o[3]
            else:
                s2 = s3 = s4 = o
            if s2.shape[-2:] != (H // 8, W // 8):
                s2 = F.interpolate(s2, size=(H // 8, W // 8), mode="bilinear", align_corners=False)
            if s3.shape[-2:] != (H // 16, W // 16):
                s3 = F.interpolate(s3, size=(H // 16, W // 16), mode="bilinear", align_corners=False)
            if s4.shape[-2:] != (H // 32, W // 32):
                s4 = F.interpolate(s4, size=(H // 32, W // 32), mode="bilinear", align_corners=False)
            f2.append(s2[0].half().cpu().numpy())
            f3.append(s3[0].half().cpu().numpy())
            f4.append(s4[0].half().cpu().numpy())
            ok_names.append(names[i])
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)

    stage2 = np.stack(f2); stage3 = np.stack(f3); stage4 = np.stack(f4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, names=np.array(ok_names),
                        stage2=stage2, stage3=stage3, stage4=stage4)
    mb = Path(args.out).stat().st_size / 1e6
    print(f"\n保存 {args.out}  ({mb:.0f} MB)")
    print(f"  stage2: {stage2.shape}")
    print(f"  stage3: {stage3.shape}")
    print(f"  stage4: {stage4.shape}")


if __name__ == "__main__":
    main()
