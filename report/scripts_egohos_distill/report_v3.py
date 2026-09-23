"""report_v3.py — ViT-Small 三组对比（排除容量因素）+ 与 ViT-Tiny 对照

输出: results_v3/ablation_v3.md, results_v3/curves_v3.png
      results_v3/tiny_vs_small.md  （跨规模对照）
"""
from __future__ import annotations

import json
import os
from pathlib import Path

V3 = {
    "A_aux(small)": "runs_v3/A_aux",
    "B_noaux(small)": "runs_v3/B_noaux",
    "C_aux+featdistill(small)": "runs_v3/C_aux_distill",
}
V2 = {
    "A_aux(tiny)": "runs_v2/A_aux",
    "B_noaux(tiny)": "runs_v2/B_noaux",
    "C_aux+featdistill(tiny)": "runs_v2/C_aux_distill",
}
OUT = Path("results_v3")


def load(runs):
    data = {}
    for name, d in runs.items():
        p = Path(d) / "log.json"
        if p.exists():
            data[name] = json.loads(p.read_text(encoding="utf-8"))
    return data


def summarize(data):
    rows = []
    for name, j in data.items():
        log = j["log"]
        def best(key, metric="f1"):
            vs = [r["metrics"][key][metric] for r in log if key in r["metrics"]]
            return max(vs) if vs else None
        rows.append(dict(组=name,
                         参数量_M=round(j.get("params_M", 0) or 0, 2),
                         最佳hand_F1=round(best("hand") or 0, 4),
                         末轮hand_F1=round(log[-1]["metrics"]["hand"]["f1"], 4),
                         最佳contact_F1=(round(best("contact"), 4) if best("contact") is not None else "—"),
                         最佳object_F1=(round(best("object"), 4) if best("object") is not None else "—"),
                         epochs=len(log)))
    return rows


def md_table(rows, title):
    if not rows:
        return [f"## {title}", "", "(无数据)", ""]
    keys = [k for k in rows[0] if k != "组"]
    out = [f"## {title}", "", "| 组 | " + " | ".join(keys) + " |",
           "|" + "|".join(["---"] * (len(keys) + 1)) + "|"]
    for r in rows:
        out.append("| " + r["组"] + " | " + " | ".join(str(r[k]) for k in keys) + " |")
    out.append("")
    return out


def main():
    os.chdir(Path(__file__).parent)
    OUT.mkdir(parents=True, exist_ok=True)
    d3, d2 = load(V3), load(V2)
    rows3, rows2 = summarize(d3), summarize(d2)

    lines = ["# 实验记录 003：ViT-Small（排除容量因素）", "",
             "- 与 EXP_002 完全同设置，只把骨干 ViT-Tiny → ViT-Small（5.8M → 22M）",
             "- 数据：EgoHOS 伪标签 863 帧（train 543 / val 320）", ""]
    lines += md_table(rows3, "ViT-Small 三组结果")

    lines += ["## 与 ViT-Tiny 对照（跨规模）", ""]
    def get(rows, key):
        for r in rows:
            if r["组"].startswith(key):
                return r
        return None
    for tag in ["A", "B", "C"]:
        s, t = get(rows3, tag), get(rows2, tag)
        if s and t:
            lines.append(f"- **{tag}**：Tiny {t['最佳hand_F1']} → Small {s['最佳hand_F1']}"
                         f"（{s['最佳hand_F1'] - t['最佳hand_F1']:+.4f}）")
    lines.append("")
    if len(rows3) == 3:
        a, b, c = get(rows3, "A"), get(rows3, "B"), get(rows3, "C")
        lines += ["## 结论（ViT-Small）", "",
                  f"- 辅助头：A−B = **{a['最佳hand_F1'] - b['最佳hand_F1']:+.4f}**",
                  f"- 特征蒸馏：C−A = **{c['最佳hand_F1'] - a['最佳hand_F1']:+.4f}**",
                  f"- 完整 vs 基线：C−B = **{c['最佳hand_F1'] - b['最佳hand_F1']:+.4f}**", ""]
    (OUT / "ablation_v3.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for name, j in d3.items():
            log = j["log"]
            axes[0].plot([r["epoch"] for r in log], [r["metrics"]["hand"]["f1"] for r in log],
                         marker="o", ms=3, label=name)
            axes[1].plot([r["epoch"] for r in log], [r["loss"] for r in log],
                         marker="o", ms=3, label=name)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("hand F1")
        axes[0].set_title("ViT-Small: hand F1"); axes[0].legend(); axes[0].grid(alpha=.3)
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("train loss")
        axes[1].set_title("ViT-Small: train loss"); axes[1].legend(); axes[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(OUT / "curves_v3.png", dpi=140)
        print(f"\n[图] {OUT/'curves_v3.png'}")
    except Exception as e:
        print(f"(画图失败: {e})")


if __name__ == "__main__":
    main()
