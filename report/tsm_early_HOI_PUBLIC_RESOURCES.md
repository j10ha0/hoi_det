# 单目 RGB 手-物检测：公开数据集与开源仓库清单

> 结论：**单目 RGB 完全可做，公开数据集和仓库都很成熟**——不用从零做
> 检索时间：2026-09-15　验证：关键仓库均已 `git ls-remote` 确认可访问

---

## 一、公开数据集

| 数据集 | 内容 | 规模 | 出处 |
|--------|------|------|------|
| **100DOH**（100 Days of Hands）| **手-物接触检测**（手是否接触物体）| ~100K 手 / 131 天视频 | CVPR 2020（255 引）|
| **ContactPose** | 手-物**接触**（含抓握姿态）| 2.9M 帧 / 25 物体 | Meta Research |
| **HOT3D** | **第一人称**手物 3D 追踪 | 833 分钟 | Meta（2024）|
| **Ego4D** | 第一人称日常活动 | 3670 小时 | Meta + 多校 |
| **Epic-Kitchens** | 第一人称厨房操作 | 100 小时 | 布里斯托 |
| **DexYCB** | 手抓取物体（3D 姿态）| 582K 帧 | NVIDIA |
| **HOI4D** | 4D 手-物交互 | 240 万帧 | — |

**与你任务最相关**：**100DOH**（手-物接触）、**ContactPose**（接触+抓握）、**HOT3D**（第一人称 3D）

---

## 二、开源仓库（均已验证可访问 ✓）

### 2.1 ⭐ 直接可用（手检测 / 接触检测）
| 仓库 | ⭐ | 用途 |
|------|-----|------|
| **`ddshan/hand_detector.d2`** | 85 | **100DOH 训练的手检测器**（detectron2）→ 输出手框 + 是否接触物体 |
| `dulre159/yolo_hand_object_detector` | 0 | YOLOv3 版手-物检测（100DOH 训练）|
| `hwjiang1510/GraspTTA` | 130 | 手-物接触一致性（ICCV 2023）|
| `lixiny/CPF` | 141 | 接触势场（ICCV 2021）|
| `takumayagi/hand_object_contact_prediction` | 17 | 手-物**接触预测**（代码+数据）|

### 2.2 HOI 检测（手-物交互）
| 仓库 | ⭐ | 用途 |
|------|-----|------|
| `AhmadDarKhalil/HOI-DETR` | 201 | HOI 检测（DETR 系列）|
| **`Sid2697/HOPformer`** | 43 | **时序 HOI 检测**（视频流，ECCV 2026）← 适合"到达检测" |
| `fpv-iplab/HOI-Synth` | 17 | 合成数据 HOI |
| `IRMVLab/Diff-IP2D` | 23 | 手-物交互（IROS 2025）|

### 2.3 第一人称 / 自我中心
| 仓库 | ⭐ | 用途 |
|------|-----|------|
| `facebookresearch/hot3d` | 285 | **第一人称手物 3D 追踪**（Meta）|
| `owenzlz/EgoHOS` | 149 | 第一人称手-物分割 |
| `Sid2697/HOI-Ref` | 30 | 手-物交互引用 |
| `yuggiehk/CaRe-Ego` | 23 | 接触感知 |
| `zhoubohan0/MEgoHand` | 12 | 多模态第一人称手物（NeurIPS 2025）|
| `phai-lab/EgoHOI` | 17 | 第一人称世界模型 |

### 2.4 数据集仓库
| 仓库 | ⭐ | 用途 |
|------|-----|------|
| `facebookresearch/ContactPose` | 432 | 手-物接触数据集 + 工具 |
| `Pradeepsaman/Egocentric-100K-dataset` | 1 | 第一人称 100K 数据集 |

---

## 三、⭐ 推荐给你的三条技术路线

### 路线 A：最快（1–2 天）—— MediaPipe + 尺度归一化
```
手: MediaPipe Hands（掌心/包围盒）
物: YOLO 或固定框
距离: 尺度归一化(手宽) + IoU  ← 我已写好演示
```
✅ 零成本、最快 ⚠️ 距离是相对的

### 路线 B：借力现成模型（1–2 周）—— 100DOH 手检测器
```
手: ddshan/hand_detector.d2（100DOH 训练）
    → 直接输出「手框 + 是否接触物体」
物: YOLO
```
✅ **不用自己定义"距离"**——模型直接给"接触/未接触" ⚠️ 需装 detectron2

### 路线 C：最学术（2–4 周）—— 时序 HOI 检测
```
用 HOPformer 类模型（视频流 HOI 检测）
→ 直接输出时序的手-物交互状态
```
✅ 故事最硬（有 baseline 对比）⚠️ 需要更多调试

---

## 四、⭐ 最关键的认知

**"手是否接触物体"在学术界叫 Hand-Object Contact Detection，是成熟子领域**：
- 数据集：100DOH（100K 手）、ContactPose
- 方法：从"算距离"演进到"**预测接触状态**"（分类问题，更鲁棒）
- **你不必自己定义距离阈值** → 可直接用现成模型的"接触概率"

**对你的价值**：
| 你的目标 | 用现成方案的哪部分 |
|---------|------------------|
| 手到达检测 | 100DOH 手检测器 / HOPformer |
| 物体形状判断 | 自己训 5 类分类器（简单）|
| 与 EMG 融合 | 你的原创部分 ✅ |

## 五、建议

1. **先跑路线 A**（2 天出 baseline，我已给代码）
2. **同时下载 100DOH 手检测器**试试（`ddshan/hand_detector.d2`）
3. 对比两者 → 选一个作为你的"视觉分支"
4. **物体形状分类自己训**（这是你最可控的部分）
5. **与 EMG 融合是你的原创贡献**

---

## 六、文件

| 文件 | 说明 |
|------|------|
| `HOI_PUBLIC_RESOURCES.md` | 本文件（数据集+仓库清单）|
| `VISION_REACH_OBJECT_PLAN.md` | 纯视觉方案（手到达+物体形状）|
| `distance_estimation_demo.py` | 距离估计 4 种方法演示 |
| `VISION_CONTROL_STRATEGIES.md` | 控制策略文献调研 |
