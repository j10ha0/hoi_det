#!/bin/bash
# build_report.sh — 把分散的实验产物归集到 EgoHOS_distill/report/
set -e
cd /mnt/storage/siat_cjh/proj/EgoHOS_distill

mkdir -p report/logs report/figures report/results

# 1) 日志固化（/tmp 会被清理，必须拷贝）
cp -f /tmp/v2_A.log /tmp/v2_B.log /tmp/v2_C.log            report/logs/ 2>/dev/null || true
cp -f /tmp/v3_A.log /tmp/v3_B.log /tmp/v3_C.log            report/logs/ 2>/dev/null || true
cp -f /tmp/v4_C3l01.log /tmp/v4_C3l02.log /tmp/v4_C2l02.log /tmp/v4_C4l02.log /tmp/v4_D.log report/logs/ 2>/dev/null || true
cp -f /tmp/v5_C4l01.log /tmp/v5_C4l03.log /tmp/v5_C4l05.log /tmp/v5_MS.log report/logs/ 2>/dev/null || true
cp -f /tmp/bench.log      report/logs/exp000_egohos_benchmark.log 2>/dev/null || true
cp -f /tmp/extract2.log   report/logs/teacher_feat_extract.log 2>/dev/null || true
cp -f /tmp/infer.log      report/logs/egohos_infer_testvideo1.log 2>/dev/null || true
cp -f /tmp/prep.log       report/logs/prepare_distill_data.log 2>/dev/null || true

# 2) 汇总结论（各实验的 ablation markdown）
cp -f results/ablation_aux_heads.md   report/results/EXP_001_ablation.md 2>/dev/null || true
cp -f results_v2/ablation_v2.md       report/results/EXP_002_ablation.md 2>/dev/null || true
cp -f results_v3/ablation_v3.md       report/results/EXP_003_ablation.md 2>/dev/null || true
cp -f results_v4/ablation_v4.md       report/results/EXP_004_ablation.md 2>/dev/null || true
cp -f results_v5/ablation_v5.md       report/results/EXP_005_ablation.md 2>/dev/null || true

# 3) 曲线图
cp -f results/ablation_curves.png     report/figures/exp001_curves.png 2>/dev/null || true
cp -f results_v2/curves_v2.png        report/figures/exp002_curves.png 2>/dev/null || true
cp -f results_v3/curves_v3.png        report/figures/exp003_curves.png 2>/dev/null || true
cp -f results_v4/curves_v4.png        report/figures/exp004_curves.png 2>/dev/null || true
cp -f results_v5/curves_v5.png        report/figures/exp005_curves.png 2>/dev/null || true

# 4) 数据集统计
cp -f data/stats.json                 report/results/dataset_stats.json 2>/dev/null || true

echo "=== report/ 目录 ==="
find report -type f | sort
echo
echo "=== 统计 ==="
echo "日志:   $(ls report/logs | wc -l) 个"
echo "结论:   $(ls report/results | wc -l) 个"
echo "图:     $(ls report/figures | wc -l) 个"
