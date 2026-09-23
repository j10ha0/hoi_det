"""bench_student.py — 实测学生模型的延迟/参数量（之前一直没测）

对比 EgoHOS 教师（三模型串联 196.1 ms / 5.1 FPS）
"""
from __future__ import annotations

import argparse
import time

import torch

from train_student import Student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs_v5/C4_l0.5/best.pt")
    ap.add_argument("--size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    dev = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    t_keys = (a.get("teacher_keys") or [a.get("teacher_key", "stage3")]) if a.get("distill_feat") else []
    net = Student(a["backbone"], use_aux=a["use_aux_heads"], size=tuple(args.size),
                  distill=a.get("distill_feat", False),
                  t_ch={k: {"stage2": 256, "stage3": 512, "stage4": 1024}[k] for k in t_keys})
    net.load_state_dict(ck["model"])
    net.to(dev).eval()

    n_par = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"模型: {a['backbone']} | aux={a['use_aux_heads']} | distill={a.get('distill_feat')} keys={t_keys}")
    print(f"参数量: {n_par:.2f} M")
    print(f"输入: {args.size[1]}x{args.size[0]}  batch={args.bs}\n")

    x = torch.randn(args.bs, 3, *args.size, device=dev)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(10):
            net(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.n):
            net(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.n
    print(f"延迟: {dt*1000:.2f} ms/批  →  {args.bs/dt:.1f} 帧/秒 (FPS)")
    print(f"峰值显存: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # FP16
    try:
        net.half()
        xh = x.half()
        with torch.no_grad():
            for _ in range(10):
                net(xh)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.n):
                net(xh)
            torch.cuda.synchronize()
            dt16 = (time.perf_counter() - t0) / args.n
        print(f"FP16 延迟: {dt16*1000:.2f} ms/批  →  {args.bs/dt16:.1f} FPS  (加速 {dt/dt16:.2f}x)")
    except Exception as e:
        print(f"(FP16 测试失败: {type(e).__name__}: {str(e)[:60]})")

    print(f"\n对比教师 EgoHOS: 196.1 ms / 5.1 FPS  →  加速 {(196.1/1000)/dt:.1f}x")


if __name__ == "__main__":
    main()
