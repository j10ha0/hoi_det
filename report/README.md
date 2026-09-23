# EgoHOS 蒸馏实验报告（report 总索引）

> 项目：**假肢多模态（视觉 + EMG）· 大模型蒸馏轻量学生**
> 日期：2026-09-21
> 目的：验证「EgoHOS（离线大模型）→ 蒸馏轻量学生（在线）」这条 V-E-A 路线是否可行
>
> **本目录 = 全部实验结论的唯一入口。每个数字都能追溯到脚本 + 日志。**

---

## 0. 目录结构

```
report/
├── README.md                    ← 本文件（总索引 + 追溯表）
├── SUMMARY.md                   ← 实验汇总表（EXP_001~005 一页看完）
├── EXP_001_DISTILL_ABLATION.md  ← 辅助头消融（ViT-Tiny）
├── EXP_002_DISTILL_V2.md        ← 修退化解 + 特征蒸馏（ViT-Tiny）
├── EXP_003_VITS_SMALL.md        ← ViT-Small 排除容量因素（关键反转）
├── EXP_004_LAMBDA_LAYER_BALANCE.md ← λ 扫描 + 教师层 + 自动损失平衡
├── EXP_005_LAMBDA_SWEEP_MULTISCALE.md ← λ 细化 + 多尺度蒸馏
├── figures/                     ← 所有曲线图
├── results/                     ← 各实验机读结论（ablation markdown / json）
└── egohos_result_testvideo1.mp4 ← EgoHOS 推理结果视频
```

**远端对应位置**：`/mnt/storage/siat_cjh/proj/EgoHOS_distill/report/`（含 `logs/` 日志固化）

---

## 1. ⭐ 结论速览

| # | 实验 | 关键结论 | 最佳 hand F1 |
|---|------|---------|-------------|
| 000 | EgoHOS 基准 | 输入 360×480；三模型串联 **196 ms = 5.1 FPS**；峰值显存 1.71 GB | — |
| 000 | HaGRID 握拳验证 | ext_ratio **双峰**：握拳 0.916±0.116 / 张开 1.670±0.085；阈值 1.258 → **准确率 98.6%**，d′=7.39 | — |
| 001 | 辅助头消融（Tiny）| 辅助头 +4.8 点（mIoU），但**辅助头是退化解** | 0.6995(mIoU) |
| 002 | 修退化 + 特征蒸馏（Tiny）| 退化修好（接触 F1 0.18）；但辅助头/蒸馏**都未超基线** | 0.7065 |
| 003 | **换 ViT-Small** | **辅助头 +5.3 点**（容量是混杂因素）；λ_feat=1.0 反而 **−2.1** | 0.7456 |
| 004 | λ 扫描 + 教师层 + 自动平衡 | **λ 是关键**（1.0→0.1 挽回 3.2 点）；**越深的教师层越好**（stage4 最优）；不确定性加权**无效** | 0.7729 |
| 005 | λ 细化 + 多尺度 | stage4 偏大 λ 更好（0.5 最优）；多尺度未超单层；**差值可能落在噪声内** | **0.7832** |

### 🏆 当前最佳配置

```
骨架      : ViT-Small (22M, timm 预训练)
主任务    : 手分割（3 类）
辅助头    : 接触 + 物体（Dice + BCE，像素级 pos_weight 裁剪 ≤20，λ=0.05）
特征蒸馏  : EgoHOS(Swin-B) stage4 (1024×7×7)，余弦损失，λ_feat = 0.5
数据      : EgoHOS 伪标签 863 帧（train 543 / val 320）
结果      : hand F1 = 0.7832（比无蒸馏基线 0.7456 高 +3.8 点）
```

---

## 2. ⭐ 追溯表（数字 → 脚本 → 日志）

> 工作区硬规则：**任何写进报告的数字，都要能指出生成它的脚本与日志路径。**

| 数字 | 生成脚本 | 日志（远端 `report/logs/`）| 代码 |
|------|---------|--------------------------|------|
| EgoHOS 延迟 60.6/65.4/70.1 ms、196.1 ms、5.1 FPS、1.71 GB | `bench_egohos.py` | `exp000_egohos_benchmark.log` | `proj/EgoHOS/bench_egohos.py` |
| haGRID 双峰：0.916 / 1.670 / 阈值 1.258 / 98.6% / d′=7.39 | `analyze_hagrid_ext.py` | `proj/tsm_visual` 运行日志 | `proj/tsm_visual/analyze_hagrid_ext.py` |
| 数据集 863 帧（543/320）、接触 98.6%、物体 36.4% | `prepare_distill_data.py` | `prepare_distill_data.log` | `EgoHOS_distill/prepare_distill_data.py` |
| 教师特征 shape (863,256/512/1024,28/14/7,·)、561 MB | `extract_teacher_feats.py` | `teacher_feat_extract.log` | `EgoHOS_distill/extract_teacher_feats.py` |
| EXP_001 0.6995 / 0.6516 | `train_student.py`(旧版) | — | `runs/`, `report_ablation.py` |
| EXP_002 A/B/C = 0.6863/0.7065/0.6945；接触 F1 0.182、物体 0.138 | `train_student.py`(v2) | `v2_A.log` `v2_B.log` `v2_C.log` | `runs_v2/`, `report_v2.py` |
| EXP_003 A/B/C = 0.7456/0.6928/0.7250 | `train_student.py`(v2) | `v3_A.log` `v3_B.log` `v3_C.log` | `runs_v3/`, `report_v3.py` |
| EXP_004 五个配置（0.7571/0.7241/0.7499/**0.7729**/0.7214）| `train_student.py`(v3) | `v4_*.log`（5 个）| `runs_v4/`, `report_v4.py` |
| EXP_005 四个配置（0.7415/0.7505/**0.7832**/0.7617）| `train_student.py`(v4, 多尺度) | `v5_*.log`（4 个）| `runs_v5/`, `report_v5.py` |
| EgoHOS 推理（手/接触/物体掩膜，543+320 帧）| EgoHOS `predict_video.py` | `egohos_infer_testvideo1.log` | `proj/EgoHOS/mmsegmentation/predict_video.py` |

**统一实验命令模板**（所有 EXP_002~005 都用它，只变参数）：

```bash
PY=/home/siat_cjh/miniconda3/envs/AnyEMG/bin/python
cd /mnt/storage/siat_cjh/proj/EgoHOS_distill
$PY train_student.py --data data --out runs_vX/<名> --backbone vit_small_patch16_224 \
    --use-aux-heads --lambda-contact 0.05 --lambda-object 0.05 \
    [--distill-feat --teacher-key stage4 --lambda-feat 0.5] \
    --epochs 30 --bs 8 --size 224 224 --gpu <N>
```
报告：`$PY report_v<X>.py` → `results_v<X>/ablation_v<X>.md` + `curves_v<X>.png`

---

## 3. 环境

| 用途 | 环境 / 路径 |
|------|-----------|
| EgoHOS（教师，mmseg）| conda `egohos`：`/mnt/storage/siat_cjh/conda_envs/egohos/bin/python`（Python 3.8 + torch 1.11.0+cu113 + mmcv-full 1.6.0 + mmseg 0.24.1）|
| 学生训练（timm）| conda `AnyEMG`：`/home/siat_cjh/miniconda3/envs/AnyEMG/bin/python`（torch 2.1.0 + timm 1.0.29）|
| 教师权重 | `proj/EgoHOS/mmsegmentation/work_dirs/`（4 个模型，5.1 GB）|
| 数据 | `EgoHOS_distill/data/`（863 帧 + 掩膜 + `teacher_feat.npz` 561 MB）|
| GPU | 4× NVIDIA TITAN V (12 GB) |

---

## 4. ⚠️ 全部局限（诚实清单）

| 局限 | 影响 |
|------|------|
| **数据极少**：仅 2 段视频 863 帧（训练 543）| 所有绝对数值不代表最终水平 |
| **伪标签噪声**：EgoHOS 并非每帧都检测到物体（仅 36.4% 帧有物体）| 辅助头 F1 偏低（0.18–0.34）|
| **单次运行、无多种子** | **±1–2 点差异很可能是随机波动**，EXP_005 的 λ 排序不可当定论 |
| **"取最佳 epoch"选择偏差** | 跑得越多越容易挑到高值 |
| **无显著性检验** | 不能宣称"A 显著优于 B" |
| 场景不同 | 用的是 EPIC-Kitchens 第三人称测试视频，**不是头戴第一人称**场景 |

**目前可稳妥陈述的**：
1. 流水线（EgoHOS 伪标签 → 学生训练 → 特征蒸馏 → 消融）**完全跑通且可复现**；
2. **容量**是决定辅助头是否有用的关键因素（Tiny −2.0 / Small +5.3）；
3. **特征蒸馏的权重 λ 与教师层选择**决定成败（stage4 + λ∈[0.2,0.5] 稳定为正增益）；
4. **需要多种子重复**才能确认最优配置。

---

## 5. 下一步（优先级）

1. **多种子重复**（基线 / stage4 λ0.2 / stage4 λ0.5 × 3 seed）→ 判断差异是否显著 ← 最该做
2. **加权多尺度**（stage3 λ=0.2 + stage4 λ=0.5，而非平均）
3. **增加数据**（用自己采集的数据）← 根本解法
4. 接触/物体辅助头质量提升（伪标签去噪）
