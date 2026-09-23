"""report_v4.py — λ_feat 扫描 + 教师层扫描 + 不确定性加权 vs A 基线

基线: runs_v3/A_aux（ViT-Small + 辅助头，无特征蒸馏）= 0.7456
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CAND = {
    "A_aux_基线(无蒸馏)": "runs_v3/A_aux",
    "C3_λ0.1(stage3)": "runs_v4/C3_l0.1",
    "C3_λ0.2(stage3)": "runs_v4/C3_l0.2",
    "C2_λ0.2(stage2)": "runs_v4/C2_l0.2",
    "C4_λ0.2(stage4)": "runs_v4/C4_l0.2",
    "D_不确定性加权": "runs_v4/D_uncertainty",
}
OUT = Path("results_v4")


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
                         物体F1=round(best("object"), 4) if best("object") is not None else "—",
                         epochs=len(log)))

    lines = ["# 实验记录 004：λ_feat 扫描 + 教师层 + 自动损失平衡", "",
             "- 骨架统一 ViT-Small（22M），辅助头 λ=0.05（除 D 用不确定性加权）",
             "- 基线 A = 辅助头、无特征蒸馏", ""]
    keys = ["最佳hand_F1", "末轮", "接触F1", "物体F1", "epochs"]
    lines += ["| 组 | " + " | ".join(keys) + " | Δ vs 基线 |",
              "|" + "|".join(["---"] * (len(keys) + 2)) + "|"]
    for r in rows:
        d = (r["最佳hand_F1"] - base) if base else None
        dd = f"{d:+.4f}" if d is not None else "—"
        lines.append("| " + r["组"] + " | " + " | ".join(str(r[k]) for k in keys) + f" | {dd} |")

    lines += ["", "## 结论", ""]
    for r in rows:
        if base and not r["组"].startswith("A_"):
            lines.append(f"- {r['组']}：{r['最佳hand_F1']}（{r['最佳hand_F1'] - base:+.4f} vs 基线）")
    lines += ["", "> 正 Δ = 特征蒸馏/自动平衡**优于**只用辅助头的基线。", ""]

    (OUT / "ablation_v4.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        for n, log in logs.items():
            ep = [r["epoch"] for r in log]
            ax[0].plot(ep, [r["metrics"]["hand"]["f1"] for r in log], marker="o", ms=3, label=n)
            ax[1].plot(ep, [r["loss"] for r in log], marker="o", ms=3, label=n)
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("hand F1")
        ax[0].set_title("ViT-Small + aux: hand F1"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("train loss")
        ax[1].set_title("train loss"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT / "curves_v4.png", dpi=140)
        print(f"\n[图] {OUT/'curves_v4.png'}")
    except Exception as e:
        print(f"(画图失败: {e})")


if __name__ == "__main__":
    main()
