# hoi_det — 视觉驱动的假肢抓握控制（视觉感知 + EMG 门控）

> 多模态假肢/机械手抓握控制系统。**EMG 只做"发力/静息"门控，视觉承担核心决策**。

---

## 1. 任务定义

```
EMG 张手      → 启动视觉对准循环
EMG 握手      → 停止，用当前预测的抓握类型执行抓取
EMG 持续握紧  → 维持抓握
EMG 静息      → 松开

视觉要解决三件事：
  ① 检测「离手最近」的物体 → 物体中心
  ② 预测「该用哪种抓握类型」（4 类）
  ③ 预测「腕部该转多少」让掌心对准物体中心
```

**关键设计原则：把"怎么对准"和"抓什么"解耦**

| 问题 | 用什么 | 性质 |
|------|--------|------|
| **怎么对准？** | **物体中心** + 掌心法向（几何投影）| 不用学 |
| **抓什么？** | 4 类抓握分类器 | 要学 |
| 对不准怎么办？ | 残余误差大 → 移动手臂 | 控制 |

---

## 2. 当前方案（已确定）

### 感知（用现成的，不训）
| 组件 | 模型 | 说明 |
|------|------|------|
| 手部 | **WiLoR**（ViT-H）/ MediaPipe | 掌心位置 + 掌心法向（21 个 3D 关键点叉乘）|
| 物体 | **YOLO**（COCO）| 检测框；**类别错无所谓，只要框中心** |
| 「离手最近」| **规则** | 框中心距掌心最近 |

### 对准（几何，不用学）
```python
u   = normalize(p_obj − p_palm)              # 掌心应指向物体中心
ang = angle(掌心法向 n, u)                    # 偏差多大
Δ   = signed(n → u)                          # 往哪转、转多少
aligned ⟺ ang < ε                            # 判定阈值（建议 40°）
```

### 抓握分类（自己训，小模型）
- **4 类**：`五指抓握(Power)` / `三指抓握(Tripod)` / `两指勾(Hook, 提袋)` / `侧捏(Lateral pinch)`
- **用两个网络**（解耦）：感知用现成的 + 抓握分类自己训
- 骨干：MobileNetV3-Small (2.5M) / ViT-Tiny (5.8M)
- 数据：每类每物体 ≥ 20 次；**训练/测试按物体划分**（object-held-out）

> ⚠️ 见 `report/ONE_OR_TWO_NETWORKS.md`：两个必须避开的坑
> ① 别把手部姿态喂进分类器（标签泄漏）
> ② 训练/测试必须按**物体**划分，否则准确率虚高

---

## 3. 目录结构

```
hoi_det/
├── report/                     ⭐ 全部设计文档（43 个 md）+ 可追溯材料
│   ├── ONE_OR_TWO_NETWORKS.md     网络架构决策（一个 vs 两个）
│   ├── GRASP_TYPE_DECISION.md     抓握类型判定方案
│   ├── OBJECT_CENTER_VS_AXIS.md   ⭐ 为什么只对准物体中心（不算主轴）
│   ├── SMALL_MODEL_VS_JEV.md      小视觉模型 vs Jev 决策模型
│   ├── VISUAL_SERVOING_DESIGN.md  视觉伺服设计
│   ├── EMG_TRIGGERED_VISION_FRAMEWORK.md  EMG 门控框架
│   ├── figures/                   实验曲线图
│   ├── logs_egohos_distill/       （历史）EgoHOS 蒸馏日志
│   ├── results_egohos_distill/    （历史）蒸馏结果
│   └── scripts_egohos_distill/    （历史）蒸馏脚本
├── egotactile/                 EgoTactile 数据集（bare_hand 子集）
│   ├── *.json                      压力标注（逐帧 162 传感器）
│   ├── frames/  grasp_frames/      抽帧（普通 / 抓握中）
│   ├── nc_frames4/                 手未接触帧（用压力标注选出）
│   └── vis*  overview*.jpg         实验结果可视化
├── ego_frames/ gt_frames/ scan_frames/   EgoDex 抽帧（参考用）
├── dexycb_seq/                 DexYCB 抽帧（参考用，已判定视角不适合）
└── *.py                        28 个分析脚本
```

### 主要脚本

| 脚本 | 作用 |
|------|------|
| `detect_coco.py` | COCO YOLO 检测物体（标准视觉检测，无需提示）|
| `eval_coco_classes.py` | 逐类评估检测器能否说出"具体是什么" |
| `align_by_detector.py` | ⭐ **用检测框中心做对准**（当前主方案）|
| `align_center_demo2.py` | 对准物体中心（含符号自动标定）|
| `grasp_cluster.py` | ⭐ **手型几何特征 + 聚类**（用作抓握标签源）|
| `pick_grasp_frames.py` | 用压力标注挑「抓握中」的帧 |
| `pick_noncontact3.py` | 用压力标注挑「手未接触」的帧 |
| `wilor_demo_vis.py` | WiLoR 3D 手部重建 + 掌心法向可视化 |
| `grasp_cluster.py` | 14 维手型特征（手指弯曲角/指尖距/对合/张开度）|

---

## 4. 环境

| 环境 | 路径 | 用途 |
|------|------|------|
| `wilor` | `/mnt/storage/siat_cjh/conda_envs/wilor` | WiLoR / YOLO / FastSAM / sklearn（**本项目主环境**）|
| `AnyEMG` | `/home/siat_cjh/miniconda3/envs/AnyEMG` | EMG 主线（不在本仓库范围内）|

```bash
# 典型用法
CUDA_VISIBLE_DEVICES=0 /mnt/storage/siat_cjh/conda_envs/wilor/bin/python align_by_detector.py
```

**依赖**：torch 2.1.0+cu118, ultralytics, wilor_mini, scikit-learn, opencv-python, numpy<2

---

## 5. 数据来源

| 数据集 | 用途 | 获取 |
|--------|------|------|
| **EgoTactile** | ⭐ **主数据集**（ego 视角、绿幕、单物体、25 类、带压力标注）| [HuggingFace](https://huggingface.co/datasets/HustleHard/EgoTactile) |
| EgoDex | 参考（有真值关节，但无物体标注）| Apple Vision Pro 采集 |
| DexYCB | ❌ 已排除（第三人称，视角不符）| — |

**EgoTactile 下载**（视频与权重不入库）：
```bash
curl -L -o Apple.mp4 \
  "https://huggingface.co/datasets/HustleHard/EgoTactile/resolve/main/Raw_data/bare_hand/p001/Apple/repeat0000/video.mp4"
```

**模型权重**（均不入库，首次运行自动下载或手动放置）：
- `yolo11s.pt` — COCO 检测
- `FastSAM-s.pt` — 分割
- WiLoR 权重 — 自动下载到 `wilor_mini/pretrained_models/`

---

## 6. 实验结果摘要

| 实验 | 结果 | 出处 |
|------|------|------|
| **对准（检测框中心）** | 符号标定 +1 vs −1：**137.1° vs 42.9°**（差异 94°，极清晰）| `align_by_detector.py` |
| 对准角度分布 | 23°–76°，中位 **42.9°** | 同上 |
| **COCO 类别识别** | 15 类中仅 **6 类**类别正确（罐头/哑铃/夹子检不出）| `eval_coco_classes.py` |
| 检测框可用性 | **11/13 帧**框位置正确（类别错不影响中心）| `detect_coco.py` |
| **手型聚类** | 轮廓系数最佳 k=5，但**有单样本簇** → 实用取 **k=3** | `grasp_cluster.py` |
| 手型聚类分布 | 一个簇占 **63%**（EgoTactile 物体多为瓶罐）| 同上 |
| WiLoR 腕旋精度 | 右手误差中位 **22.6°**（左 47.2°）| `compare_wilor_gt.py` |

> ⚠️ **可追溯性**：以上数字均可由对应脚本生成；历史 EgoHOS 蒸馏的日志与结果保存在
> `report/logs_egohos_distill/` 与 `report/results_egohos_distill/`。

---

## 7. 未完成 / 下一步

- [ ] 微调**专用检测器**（覆盖 25 类，用绿幕+目录名**自动标注**）
- [ ] 训 **4 类抓握分类器**（MobileNetV3/ViT-Tiny），按物体划分验证
- [ ] 自动标注流水线（几何规则 + VLM 交叉验证 + 人工裁决分歧）
- [ ] 采集自己的数据（4 类 × 设计物体）
- [ ] 与 EMG 门控联调

---

## 8. 引用

- **EgoTactile** — *Learning Grasp Pressure for Everyday Objects from Egocentric Video*, ICML 2026 spotlight. [arXiv:2606.09243](https://arxiv.org/abs/2606.09243) ｜ [项目页](https://egotactile.github.io/) ｜ 数据 CC BY-NC 4.0
- **WiLoR** — *Reconstructing Hands in 3D with Transformers*, CVPR 2025
- **EgoHOS** — *Fine-Grained Egocentric Hand-Object Segmentation*, ECCV 2022（历史，已归档）
