"""report_v2.py — 汇总 v2 三组实验：A(辅助头) / B(无辅助头) / C(辅助头+特征蒸馏)

读取: runs_v2/*/log.json（v2 训练脚本输出：metrics 含 F1/IoU/precision/recall）
输出: results_v2/ablation_v2.md + results_v2/curves_v2.png
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

RUNS = {
    "A_aux": "runs_v2/A_aux",
    "B_noaux": "runs_v2/B_noaux",
    "C_aux+featdistill": "runs_v2/C_aux_distill",
}
OUT = Path("results_v2")


def main():
    os.chdir(Path(__file__).parent)
    OUT.mkdir(parents=True, exist_ok=True)
    data, rows = {}, []
    for name, d in RUNS.items():
        p = Path(d) / "log.json"
        if not p.exists():
            print(f"[跳过] {p} 不存在（可能还在训练）")
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        log = j["log"]

        def best_of(key, metric="f1"):
            vs = [r["metrics"][key][metric] for r in log if key in r["metrics"]]
            return max(vs) if vs else None

        data[name] = log
        rows.append(dict(
            组=name,
            参数量_M=round(sum(1 for _ in []) or 0, 2),   # 占位，下面补
            最佳hand_F1=round(best_of("hand") or 0, 4),
            末轮hand_F1=round(log[-1]["metrics"]["hand"]["f1"], 4),
            最佳contact_F1=(round(best_of("contact"), 4) if best_of("contact") is not None else "—"),
            最佳object_F1=(round(best_of("object"), 4) if best_of("object") is not None else "—"),
            contact_IoU=(round(best_of("contact", "iou"), 4) if best_of("contact") is not None else "—"),
            object_IoU=(round(best_of("object", "iou"), 4) if best_of("object") is not None else "—"),
            epochs=len(log),
        ))

    if not rows:
        print("无结果")
        return

    keys = list(rows[0].keys())[1:]          # 去掉参数量占位
    lines = ["# 实验记录 002：辅助头 + 特征蒸馏（v2，修正类别权重与指标）", "",
             "- 数据：EgoHOS 伪标签 863 帧（train 543 / val 320）",
             "- 主任务：手分割 F1；辅助头：接触/物体（Dice + 裁剪 pos_weight=20）",
             "- C 组：额外加 Swin-B stage3 特征的余弦蒸馏（λ=1.0）",
             "",
             "| 组 | " + " | ".join(keys) + " |",
             "|" + "|".join(["---"] * (len(keys) + 1)) + "|"]
    for r in rows:
        lines.append("| " + r["组"] + " | " + " | ".join(str(r[k]) for k in keys) + " |")

    lines += ["", "## 结论", ""]
    a = next((r for r in rows if r["组"].startswith("A")), None)
    b = next((r for r in rows if r["组"].startswith("B")), None)
    c = next((r for r in rows if r["组"].startswith("C")), None)
    if a and b:
        d = a["最佳hand_F1"] - b["最佳hand_F1"]
        lines.append(f"- **辅助头效果**：A(有) {a['最佳hand_F1']} vs B(无) {b['最佳hand_F1']}，差值 **{d:+.4f}**")
    if a and c:
        d = c["最佳hand_F1"] - a["最佳hand_F1"]
        lines.append(f"- **特征蒸馏效果**：C(有) {c['最佳hand_F1']} vs A(无) {a['最佳hand_F1']}，差值 **{d:+.4f}**")
    if c and b:
        d = c["最佳hand_F1"] - b["最佳hand_F1"]
        lines.append(f"- **完整方案 vs 基线**：C vs B，差值 **{d:+.4f}**")
    lines.append("")
    lines.append("> ⚠️ 数据仅 543 训练帧，绝对数值不代表最终水平；辅助头 F1 仍偏低说明伪标签稀疏类难学。")

    (OUT / "ablation_v2.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for name, log in data.items():
            ep = [r["epoch"] for r in log]
            axes[0].plot(ep, [r["metrics"]["hand"]["f1"] for r in log], marker="o", ms=3, label=name)
            axes[1].plot(ep, [r["loss"] for r in log], marker="o", ms=3, label=name)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("hand F1")
        axes[0].set_title("Main task: hand segmentation F1"); axes[0].legend(); axes[0].grid(alpha=.3)
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("train loss")
        axes[1].set_title("Training loss"); axes[1].legend(); axes[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT / "curves_v2.png", dpi=140)
        print(f"\n[图] {OUT/'curves_v2.png'}")
    except Exception as e:
        print(f"(画图失败: {e})")


if __name__ == "__main__":
    main()
