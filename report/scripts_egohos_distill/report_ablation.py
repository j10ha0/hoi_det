"""report_ablation.py — 汇总「有/无辅助头」消融结果 → 表格 + 曲线图

读取: runs/<name>/log.json
输出: results/ablation_aux_heads.md  +  results/ablation_curves.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RUNS = {
    "A_with_aux": "runs/student_aux",
    "B_no_aux": "runs/student_noaux",
}
OUT = Path("results")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, rows = {}, []
    for name, d in RUNS.items():
        p = Path(d) / "log.json"
        if not p.exists():
            print(f"[跳过] {p} 不存在")
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        log = j["log"]
        best = max(log, key=lambda r: r["mean_iou"]["hand"])
        last = log[-1]
        data[name] = log
        rows.append(dict(
            组=name,
            参数量_M=round(j.get("params_M", 0), 2),
            最佳hand_mIoU=round(best["mean_iou"]["hand"], 4),
            该轮=best["epoch"],
            末轮hand_mIoU=round(last["mean_iou"]["hand"], 4),
            接触_mIoU=(round(best["mean_iou"]["contact"], 4) if "contact" in best["mean_iou"] else "—"),
            物体_mIoU=(round(best["mean_iou"]["object"], 4) if "object" in best["mean_iou"] else "—"),
            末轮loss=round(last["loss"], 4),
        ))

    if not rows:
        print("没有可汇总的结果")
        return

    # markdown 表
    keys = list(rows[0].keys())
    lines = ["# 消融实验：辅助头是否有用（ViT-Tiny 学生 × EgoHOS 伪标签）", "",
             f"- 数据：EgoHOS 伪标签，train {len(data.get('A_有辅助头', [])) and ''}543 帧 / val 320 帧",
             "- 主任务：手分割（3 类）；辅助头：接触（二分类）+ 物体（二分类）", "",
             "| " + " | ".join(keys) + " |",
             "|" + "|".join(["---"] * len(keys)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    lines += ["", "## 结论", ""]
    if len(rows) == 2:
        a = rows[0]["最佳hand_mIoU"]
        b = rows[1]["最佳hand_mIoU"]
        diff = a - b
        lines.append(f"- 有辅助头 hand mIoU = **{a}**，无辅助头 = **{b}**，差值 **{diff:+.4f}**")
        if diff > 0.02:
            lines.append("- ✅ 辅助头**明显有帮助**（复现了 PocketVLA 的结论方向）")
        elif diff > 0.005:
            lines.append("- ⚠️ 辅助头**略有帮助**（幅度较小）")
        else:
            lines.append("- ❌ 辅助头**帮助不明显**（与 PocketVLA 结论不同，值得讨论）")
    (OUT / "ablation_aux_heads.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # 曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, log in data.items():
            ep = [r["epoch"] for r in log]
            axes[0].plot(ep, [r["mean_iou"]["hand"] for r in log], marker="o", ms=3, label=name)
            axes[1].plot(ep, [r["loss"] for r in log], marker="o", ms=3, label=name)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("hand mIoU")
        axes[0].set_title("Main task: hand segmentation mIoU"); axes[0].legend(); axes[0].grid(alpha=.3)
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("train loss")
        axes[1].set_title("Training loss"); axes[1].legend(); axes[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(OUT / "ablation_curves.png", dpi=140)
        print(f"\n[图] {OUT/'ablation_curves.png'}")
    except Exception as e:
        print(f"(画图失败: {e})")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    main()
