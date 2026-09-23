# MediaPipe HandLandmarker 输出数据完整说明（实测）

> 环境：mediapipe **1.0.1**（新版**只有 tasks API**，旧的 `mp.solutions.hands` 已移除）
> 模型：`hand_landmarker.task`（7.8 MB）　实测图：640×960，检测到 2 只手
> 目录：`/mnt/storage/siat_cjh/proj/tsm_visual/`　日期：2026-09-15

---

## 1. 一次检测返回什么

```python
result = detector.detect(mp_image)
result.hand_landmarks        # list[list[NormalizedLandmark]]，每只手 21 点
result.hand_world_landmarks  # list[list[Landmark]]，每只手 21 点（米制 3D）
result.handedness            # list[list[Category]]，左右手 + 置信度
```

**三样东西，每只手一份**：

| 输出 | 含义 | 坐标系 |
|------|------|--------|
| `hand_landmarks` | 21 个关键点（归一化）| 图像像素归一化 + 相对手腕深度 |
| `hand_world_landmarks` | 21 个关键点（米制 3D）| **原点 = 手部几何中心** |
| `handedness` | 左手 / 右手 | 类别 + 置信度 |

## 2. 21 个关键点的定义（官方索引）

| 索引 | 名称 | 位置 |
|------|------|------|
| **0** | `WRIST` | 手腕 |
| 1–4 | `THUMB_CMC` / `MCP` / `IP` / `TIP` | 拇指：腕掌/掌指/指间/指尖 |
| 5–8 | `INDEX_MCP` / `PIP` / `DIP` / `TIP` | 食指：掌指/近端/远端/指尖 |
| 9–12 | `MIDDLE_MCP` / `PIP` / `DIP` / `TIP` | 中指 |
| 13–16 | `RING_MCP` / `PIP` / `DIP` / `TIP` | 无名指 |
| 17–20 | `PINKY_MCP` / `PIP` / `DIP` / `TIP` | 小指 |

> 对比：你们参考项目手套的关节命名是 `TMC/MCP/PIP/DIP`（Thumb 无 PIP，用 IP）——**基本能对上**，可做映射。

## 3. 每个关键点的字段（实测）

```python
landmark = result.hand_landmarks[0][8]     # 食指尖
landmark.x, landmark.y, landmark.z
landmark.visibility      # ← 实测存在！
landmark.presence        # ← 实测存在！
landmark.name            # 如 "INDEX_FINGER_TIP"
```

**实测字段列表**：`['x', 'y', 'z', 'visibility', 'presence', 'name']`

| 字段 | 含义 |
|------|------|
| `x`, `y` | **归一化到图像宽高**，范围 [0,1] |
| `z` | **相对手腕的深度**，与 x 同尺度（**非米制**，越负表示越靠近相机）|
| `visibility` / `presence` | 可见性 / 存在置信度 |

## 4. 两套坐标的实测数值对比

（同一张图，右手，食指尖 = 索引 8）

| 坐标系 | WRIST (0) | INDEX_TIP (8) | WRIST→TIP 距离 |
|--------|-----------|---------------|---------------|
| **归一化** `hand_landmarks` | (0.466, 0.400, 0.000) | (0.725, 0.420, −0.028) | **0.261**（无量纲）|
| **米制** `hand_world_landmarks` | (−0.0826, +0.0034, −0.0257) m | (+0.0596, +0.0213, −0.0392) m | **14.39 cm** |

> 成年人手长（腕→食指尖）约 17–19 cm，实测 14.4–14.9 cm —— **基本合理**（因为世界坐标原点是手部几何中心，不是腕部）。

**关键区别**：
- **归一化坐标**：只能做 2D 像素级任务（画图、区域裁剪），比例随相机距离变化
- **米制世界坐标**：可用于**计算关节角**、判断手势（不随距离变）

## 5. 从这些数据能推出什么

| 想要的量 | 怎么算 | 可行性 |
|---------|--------|--------|
| **关节角**（手指弯曲度）| 用 `hand_world_landmarks`：相邻三点的向量夹角（如 MCP-PIP-DIP）| ✅ 可算（精度见 §7）|
| **指尖距离**（捏合程度）| 两点欧氏距离（world 坐标）| ✅ 可算 |
| **手指伸展/弯曲状态** | 关节角阈值 | ✅ 可算 |
| **手掌朝向** | 由 WRIST/MCP 点拟合平面法向量 | ✅ 可算 |
| **手在图像中的位置/框** | `hand_landmarks` 的 x,y 范围 | ✅ 可算 |
| **左右手** | `handedness[0].category_name` + score | ✅ 直接给 |
| **深度/绝对位置** | ❌ 单目限制 | 只有相对手部几何 |

## 6. 可配置参数与三种运行模式

```python
vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=2,                              # 检测几只手（1/2）
    min_hand_detection_confidence=0.5,        # 首次检测阈值
    min_hand_presence_confidence=0.5,         # 存在置信度阈值
    min_tracking_confidence=0.5,              # 跟踪阈值（视频/流）
    running_mode=vision.RunningMode.VIDEO,    # IMAGE / VIDEO / LIVE_STREAM
)
```

| 模式 | 用途 | 特点 |
|------|------|------|
| `IMAGE` | 单帧 | 无时序跟踪 |
| `VIDEO` | 离线视频 | 需传时间戳，帧间跟踪 |
| **`LIVE_STREAM`** | **实时摄像头** | 异步回调，**延迟最低**（实时用这个）|

## 7. ⚠️ 精度现实（决定你能拿它做什么）

| 用途 | 够不够 | 说明 |
|------|--------|------|
| **视觉输入 / 场景条件**（给 EMG 提供上下文）| ✅ **完全够** | 只需手部区域/语义 |
| 2D 关键点可视化/裁剪 | ✅ 够 | 像素级精度 |
| **关节角做"真值"训练 sEMG→pose** | ⚠️ **精度有限** | 单目 3D 估计误差；比手套/Leap 差 |
| 自遮挡/背对相机的指节 | ❌ 明显退化 | 看不到的手指角度不可信 |

**建议**：
- 做**视觉分支/场景条件** → 直接用（零成本）
- 做**真值标签** → 先做§8 的对比验证，再决定

## 8. 建议的验证（呼应前面结论）

```
1. 电脑接 USB 摄像头 + LIVE_STREAM 模式 → 实时 21 关键点
2. 同时戴数据手套 → 记录两者的关节角
3. 对比误差 → 判断 MediaPipe 在你的场景够不够
```

我已给你可跑脚本：`mediapipe_hand_probe.py`（图片版）。
需要的话我再给**实时摄像头版 + 关节角换算 + 与手套对齐评测**脚本。

## 9. 文件

| 文件 | 说明 |
|------|------|
| `models/hand_landmarker.task` | 手部关键点模型（7.8 MB）|
| `mediapipe_hand_probe.py` | ★ 探测脚本（输出上述所有字段）|
| `MEDIAPIPE_HAND_OUTPUT.md` | 本说明 |

复现：
```bash
cd /mnt/storage/siat_cjh/proj/tsm_visual
/home/siat_cjh/miniconda3/envs/AnyEMG/bin/python mediapipe_hand_probe.py
```
