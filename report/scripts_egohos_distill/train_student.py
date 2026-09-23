"""train_student.py — ViT-Tiny 学生（EgoHOS 蒸馏）+ 辅助头（含类别权重 / F1 指标 / 特征蒸馏）

数据（prepare_distill_data.py 生成）:
    <data>/frames/<name>.jpg
    <data>/mask_hands/<name>.png     手 0/1/2      ← 主任务
    <data>/mask_contact/<name>.png   接触 0/1      ← 辅助头
    <data>/mask_object/<name>.png    物体 0/3(→1)  ← 辅助头
    <data>/meta.json                 split
教师特征（extract_teacher_feats.py 生成，可选）:
    <data>/teacher_feat.npz          names + stage3(N,512,14,14)

关键改进（v2）:
  1. **类别权重**：接触/物体用像素级 pos_weight 平衡（原来退化 → 全预测单类）
  2. **F1/IoU 全局指标**：按整个验证集累计 TP/FP/FN（原来按 batch 平均 mIoU 会误导）
  3. **特征蒸馏**：学生特征 1×1 投影到教师通道 → 余弦相似度损失（跨架构蒸馏）
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------- 数据 ----------------
class SegData(Dataset):
    def __init__(self, data: Path, split: str, size=(224, 224), aug=False, teacher=None):
        with open(data / "meta.json", encoding="utf-8") as f:
            self.items = [m for m in json.load(f) if m["split"] == split]
        self.root, self.size, self.aug = data, size, aug
        self.teacher = teacher                 # dict: name -> ndarray(512,14,14)
        if not self.items:
            raise RuntimeError(f"split={split} 无样本")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        m = self.items[i]
        n = m["name"]
        img = Image.open(self.root / "frames" / f"{n}.jpg").convert("RGB").resize(
            (self.size[1], self.size[0]), Image.BILINEAR)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255)

        def load(sub, remap=None):
            p = self.root / sub / f"{n}.png"
            if not p.exists():
                return torch.zeros(self.size, dtype=torch.long)
            a = np.array(Image.open(p).resize((self.size[1], self.size[0]), Image.NEAREST))
            if remap:
                out = np.zeros_like(a)
                for k, v in remap.items():
                    out[a == k] = v
                a = out
            return torch.from_numpy(a.astype(np.int64))

        yh = load("mask_hands")
        yc = load("mask_contact").clamp(0, 1)
        yo = load("mask_object", remap={3: 1}).clamp(0, 1)

        if self.aug:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, [2]); yh = torch.flip(yh, [1])
                yc = torch.flip(yc, [1]); yo = torch.flip(yo, [1])
            x = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
        x = (x - MEAN) / STD

        tf = {}
        if self.teacher is not None:
            for _k, _d in self.teacher.items():
                if n in _d:
                    tf[_k] = torch.from_numpy(_d[n].astype(np.float32))
        return x, yh, yc, yo, tf


# ---------------- 模型 ----------------
class Student(nn.Module):
    def __init__(self, backbone="vit_tiny_patch16_224", n_hand=3, use_aux=True,
                 size=(224, 224), patch=16, t_ch=None, distill=False):
        super().__init__()
        import timm
        try:
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0,
                                              features_only=False, dynamic_img_size=True)
        except Exception:
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0,
                                              features_only=False)
        self.patch, self.size = patch, size
        with torch.no_grad():
            f = self.backbone.forward_features(torch.zeros(1, 3, *size))
        self.is_token = f.dim() == 3
        if self.is_token:
            self.dim = f.shape[-1]
            self.n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        else:
            self.dim = f.shape[1]
            self.n_prefix = 0
        self.use_aux, self.distill = use_aux, distill
        self.t_ch = t_ch or {}

        def head(n_cls):
            return nn.Sequential(
                nn.Conv2d(self.dim, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.Conv2d(64, n_cls, 1))

        self.head_hand = head(n_hand)
        if use_aux:
            self.head_contact = head(1)
            self.head_object = head(1)
        if distill:
            # 每个教师层一个投影头（支持多尺度联合蒸馏）
            self.feat_proj = nn.ModuleDict({k: nn.Conv2d(self.dim, c, 1)
                                            for k, c in self.t_ch.items()})

    def features(self, x):
        f = self.backbone.forward_features(x)
        if self.is_token:
            B, N, C = f.shape
            t = f[:, self.n_prefix:, :]
            gh, gw = x.shape[-2] // self.patch, x.shape[-1] // self.patch
            f = t.transpose(1, 2).reshape(B, C, gh, gw)
        return f

    def forward(self, x):
        f = self.features(x)
        H, W = x.shape[-2], x.shape[-1]
        out = {"feat": f,
               "hand": F.interpolate(self.head_hand(f), (H, W), mode="bilinear", align_corners=False)}
        if self.use_aux:
            out["contact"] = F.interpolate(self.head_contact(f), (H, W), mode="bilinear", align_corners=False)
            out["object"] = F.interpolate(self.head_object(f), (H, W), mode="bilinear", align_corners=False)
        if self.distill:
            out["feat_proj"] = {k: m(f) for k, m in self.feat_proj.items()}
        return out


# ---------------- 像素级类别权重 ----------------
def pixel_pos_weight(data: Path, split: str, sub: str, remap=None):
    with open(data / "meta.json", encoding="utf-8") as f:
        items = [m for m in json.load(f) if m["split"] == split]
    pos = neg = 0
    for m in items:
        p = data / sub / f"{m['name']}.png"
        if not p.exists():
            continue
        a = np.array(Image.open(p))
        if remap:
            b = np.zeros_like(a)
            for k, v in remap.items():
                b[a == k] = v
            a = b
        a = (a > 0)
        pos += int(a.sum()); neg += int((~a).sum())
    w = neg / max(1, pos)
    w = min(w, 20.0)          # ★ 裁剪：避免极端权重（如 181）压倒主任务
    print(f"  [{sub}] pos_px={pos} neg_px={neg} → pos_weight={w:.3f} (已裁剪至 ≤20)")
    return torch.tensor([max(w, 1e-4)], dtype=torch.float32)


def dice_loss(logits, target, eps=1.0):
    """Dice 损失：对极稀疏前景（接触仅 0.55% 像素）比加权 BCE 更稳"""
    p = logits.sigmoid()
    num = 2 * (p * target).sum() + eps
    den = p.sum() + target.sum() + eps
    return 1 - num / den


# ---------------- 评测（全局 TP/FP/FN → F1/IoU）----------------
@torch.no_grad()
def evaluate(net, loader, dev, thr=0.5):
    net.eval()
    st = {}
    def acc(k, pred, gt):
        d = st.setdefault(k, dict(tp=0, fp=0, fn=0, tn=0))
        d["tp"] += int((pred & gt).sum()); d["fp"] += int((pred & ~gt).sum())
        d["fn"] += int((~pred & gt).sum()); d["tn"] += int((~pred & ~gt).sum())
    for x, yh, yc, yo, _ in loader:
        x = x.to(dev)
        out = net(x)
        acc("hand", out["hand"].argmax(1).cpu().bool(), yh.bool())
        if "contact" in out:
            acc("contact", (out["contact"].sigmoid() > thr).cpu()[:, 0], yc.bool())
            acc("object", (out["object"].sigmoid() > thr).cpu()[:, 0], yo.bool())
    res = {}
    for k, d in st.items():
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        res[k] = dict(f1=2 * prec * rec / max(1e-9, prec + rec),
                      iou=tp / max(1, tp + fp + fn), precision=prec, recall=rec)
    return res


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="vit_tiny_patch16_224")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--size", type=int, nargs=2, default=[224, 224])
    ap.add_argument("--use-aux-heads", dest="use_aux_heads", action="store_true", default=True)
    ap.add_argument("--no-aux-heads", dest="use_aux_heads", action="store_false")
    ap.add_argument("--lambda-contact", type=float, default=1.0)
    ap.add_argument("--lambda-object", type=float, default=1.0)
    ap.add_argument("--distill-feat", action="store_true", help="启用教师特征蒸馏")
    ap.add_argument("--lambda-feat", type=float, default=1.0)
    ap.add_argument("--teacher-key", default="stage3",
                    choices=["stage2", "stage3", "stage4"], help="单层蒸馏")
    ap.add_argument("--teacher-keys", nargs="+", default=None,
                    choices=["stage2", "stage3", "stage4"],
                    help="多层联合蒸馏（多尺度），如 --teacher-keys stage3 stage4")
    ap.add_argument("--loss-balance", default="manual", choices=["manual", "uncertainty"],
                    help="manual=手调 λ；uncertainty=可学习不确定性加权（Kendall 2018）")
    ap.add_argument("--gpu", default="1")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    data, size = Path(args.data), tuple(args.size)

    # 教师特征
    teacher = None
    if args.distill_feat:
        fz = data / "teacher_feat.npz"
        if not fz.exists():
            raise SystemExit(f"缺少教师特征: {fz}（先跑 extract_teacher_feats.py）")
        z = np.load(fz, allow_pickle=True)
        keys = args.teacher_keys or [args.teacher_key]
        for k in keys:
            if k not in z:
                raise SystemExit(f"{fz} 里没有 {k}（有的是 {list(z.keys())}）")
        teacher = {k: {str(n): s for n, s in zip(z["names"], z[k])} for k in keys}
        print("教师特征: " + ", ".join(f"{k}{z[k].shape[1:]}" for k in keys)
              + f"  {len(teacher[keys[0]])} 帧")

    tr = SegData(data, "train", size, aug=True, teacher=teacher)
    va = SegData(data, "val", size, aug=False, teacher=teacher)
    print(f"backbone={args.backbone} | aux={args.use_aux_heads} | distill={args.distill_feat}")
    print(f"train={len(tr)} val={len(va)} size={size}")
    print("类别权重（像素级）:")
    pw_c = pixel_pos_weight(data, "train", "mask_contact")
    pw_o = pixel_pos_weight(data, "train", "mask_object", remap={3: 1})

    ld_tr = DataLoader(tr, args.bs, shuffle=True, num_workers=args.workers, drop_last=True)
    ld_va = DataLoader(va, args.bs, shuffle=False, num_workers=args.workers)

    T_CH = {"stage2": 256, "stage3": 512, "stage4": 1024}
    t_keys = (args.teacher_keys or [args.teacher_key]) if args.distill_feat else []
    net = Student(args.backbone, use_aux=args.use_aux_heads, size=size,
                  distill=args.distill_feat,
                  t_ch={k: T_CH[k] for k in t_keys}).to(dev)
    print(f"参数量 {sum(p.numel() for p in net.parameters())/1e6:.2f}M")

    # 不确定性加权（Kendall 2018）: L = Σ exp(-s_i)·L_i + s_i
    TERM_IDX = {"hand": 0, "contact": 1, "object": 2, "feat": 3}
    log_sigma = nn.Parameter(torch.zeros(4, device=dev))
    params = list(net.parameters())
    if args.loss_balance == "uncertainty":
        params.append(log_sigma)
        print("损失平衡: uncertainty（可学习）")
    else:
        print(f"损失平衡: manual  λ_contact={args.lambda_contact} "
              f"λ_object={args.lambda_object} λ_feat={args.lambda_feat}")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ce = nn.CrossEntropyLoss()
    bce_c = nn.BCEWithLogitsLoss(pos_weight=pw_c.to(dev))
    bce_o = nn.BCEWithLogitsLoss(pos_weight=pw_o.to(dev))

    log, best = [], -1
    for ep in range(1, args.epochs + 1):
        net.train()
        t0, run, seen = time.time(), 0.0, 0
        for x, yh, yc, yo, tf in ld_tr:
            x, yh = x.to(dev), yh.to(dev)
            out = net(x)
            terms = {"hand": ce(out["hand"], yh)}
            if args.use_aux_heads:
                yc1 = yc.to(dev).unsqueeze(1).float()
                yo1 = yo.to(dev).unsqueeze(1).float()
                terms["contact"] = bce_c(out["contact"], yc1) + dice_loss(out["contact"], yc1)
                terms["object"] = bce_o(out["object"], yo1) + dice_loss(out["object"], yo1)
            if args.distill_feat:
                lf, nk = 0.0, 0
                for k in t_keys:
                    if k not in tf:
                        continue
                    t = tf[k].to(dev)
                    if t.dim() != 4 or t.shape[1] != T_CH[k]:
                        continue
                    sp = out["feat_proj"][k]
                    tp_ = t
                    if sp.shape[-2:] != tp_.shape[-2:]:
                        tp_ = F.interpolate(tp_, size=sp.shape[-2:], mode="bilinear",
                                            align_corners=False)
                    lf = lf + (1 - F.cosine_similarity(sp, tp_, dim=1)).mean()
                    nk += 1
                if nk:
                    terms["feat"] = lf / nk

            if args.loss_balance == "uncertainty":
                loss = torch.zeros((), device=dev)
                for k, L in terms.items():
                    s = log_sigma[TERM_IDX[k]]
                    loss = loss + torch.exp(-s) * L + s
            else:
                lam = {"hand": 1.0, "contact": args.lambda_contact,
                       "object": args.lambda_object, "feat": args.lambda_feat}
                loss = sum(lam[k] * L for k, L in terms.items())
            parts = {k: v.item() for k, v in terms.items()}
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * x.size(0); seen += x.size(0)
        sched.step()
        m = evaluate(net, ld_va, dev)
        msg = " ".join(f"{k} F1={v['f1']:.3f}" for k, v in m.items())
        print(f"[{ep:>3}/{args.epochs}] loss={run/max(1,seen):.4f} | {msg} ({time.time()-t0:.0f}s)", flush=True)
        log.append(dict(epoch=ep, loss=run / max(1, seen), metrics=m,
                        parts=parts, log_sigma=log_sigma.detach().cpu().tolist()))
        score = m["hand"]["f1"]
        if score > best:
            best = score
            torch.save({"model": net.state_dict(), "args": vars(args), "metrics": m}, outdir / "best.pt")

    with open(outdir / "log.json", "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), best_hand_f1=best, log=log), f, ensure_ascii=False, indent=2)
    print(f"\n最佳 hand F1 = {best:.4f} → {outdir/'best.pt'}")


if __name__ == "__main__":
    main()
