"""report_v5.py — stage4 的 λ 细化 + 多尺度联合蒸馏

对照: A_aux_基线(0.7456, EXP_003), C4_λ0.2(0.7729, EXP_004)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CAND = {
    "A_基线(无蒸馏)": "runs_v3/A_aux",
    "C4_λ0.2(EXP004最佳)": "runs_v4/C4_l0.2",
    "C4_λ0.1": "runs_v5/C4_l0.1",
    "C4_λ0.3": "runs_v5/C4_l0.3",
    "C4_λ0.5": "runs_v5/C4_l0.5",
    "MS_stage3+4_λ0.2": "runs_v5/MS_34_l0.2",
}
OUT = Path("results_v5")


def main():
    os.chdir(Path(__file__).parent)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, logs = [], {}
    base = None
    for name, d in CAND.items():
        p = Path(d) / "log.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        log = j["log"]
        logs[name] = log

        def best(k, metric="f1"):
            vs = [r["metrics"][k][metric] for r in log if k in r["metrics"]]
            return max(vs) if vs else None

        hb = best("hand") or 0
        if name.startswith("A_"):
            base = hb
        rows.append(dict(组=name, 最佳hand_F1=round(hb, 4),
                         末轮=round(log[-1]["metrics"]["hand"]["f1"], 4),
                         接触F1=round(best("contact"), 4) if best("contact") is not None else "—",
                         物体F1=round(best("object"), 4) if best("object") is not None else "—"))

    lines = ["# 实验记录 005：stage4 的 λ 细化 + 多尺度蒸馏", "",
             "- 骨架 ViT-Small，辅助头 λ=0.05，只变蒸馏配置", ""]
    keys = ["最佳hand_F1", "末轮", "接触F1", "物体F1"]
    lines += ["| 组 | " + " | ".join(keys) + " | Δ vs 基线 |",
              "|" + "|".join(["---"] * (len(keys) + 2)) + "|"]
    for r in rows:
        d = (r["最佳hand_F1"] - base) if base else None
        lines.append("| " + r["组"] + " | " + " | ".join(str(r[k]) for k in keys)
                     + f" | {d:+.4f} |" if d is not None else " |")
    best_row = max(rows, key=lambda r: r["最佳hand_F1"]) if rows else None
    lines += ["", "## 结论", ""]
    if best_row:
        lines.append(f"- 🏆 本轮最佳：**{best_row['组']} = {best_row['最佳hand_F1']}**")
        if base:
            lines.append(f"- 相对基线（无蒸馏）提升 **{best_row['最佳hand_F1'] - base:+.4f}**")
    lines.append("")
    (OUT / "ablation_v5.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        for n, log in logs.items():
            ep = [r["epoch"] for r in log]
            axes[0].plot(ep, [r["metrics"]["hand"]["f1"] for r in log], marker="o", ms=3, label=n)
            axes[1].plot(ep, [r["loss"] for r in log], marker="o", ms=3, label=n)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("hand F1")
        axes[0].set_title("stage4 λ sweep + multi-scale: hand F1")
        axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("train loss")
        axes[1].set_title("train loss"); axes[1].legend(fontsize=7); axes[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT / "curves_v5.png", dpi=140)
        print(f"\n[图] {OUT/'curves_v5.png'}")
    except Exception as e:
        print(f"(画图失败: {e})")


if __name__ == "__main__":
    main()
