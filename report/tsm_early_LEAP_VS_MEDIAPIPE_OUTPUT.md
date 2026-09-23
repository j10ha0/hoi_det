# Leap Motion Controller 2 vs MediaPipe：输出数据对比（基于 SDK 源码实测）

> Leap 数据来源：`Fingers_Gestures_Recognition/LeapSDK/include/LeapC.h`（该论文官方 SDK，逐字段提取）
> MediaPipe 数据来源：本机实测（`mediapipe_hand_probe.py`，mediapipe 1.0.1）
> 日期：2026-09-15

---

## 1. Leap Motion Controller 2 输出什么

**一帧 = 若干 `LEAP_HAND`**，每只手包含以下层级：

```
LEAP_HAND
├── id                 手部 ID（跨帧跟踪，丢失重获会换新 ID）
├── type               左手 / 右手
├── confidence         ⚠️ 文档注明「Not currently used (always 1.0)」——实际恒为 1
├── visible_time       已跟踪时长（微秒）
├── pinch_distance     食指与拇指的距离（mm）
├── grab_angle         手指相对手掌的平均角度（度）
├── pinch_strength     ✅ 归一化捏合强度（0=未捏，1=完全捏合）
├── grab_strength      ✅ 归一化抓握强度（0=未抓，1=完全抓）
├── palm               LEAP_PALM（手掌）
├── digits[5]          thumb / index / middle / ring / pinky（每指 LEAP_DIGIT）
└── arm                LEAP_BONE（肘→腕，单骨节）
```

### 1.1 LEAP_PALM（手掌）
| 字段 | 含义 | 单位 |
|------|------|------|
| `position` | 手掌中心位置（**相对相机原点**）| **mm** |
| `stabilized_position` | 平滑稳定后的位置（有滞后）| mm |
| `velocity` | 手掌速度 | mm/s |
| `normal` | 手掌法向量 | 单位向量 |
| `width` | 手掌宽度 | mm |
| `direction` | 指向手指的单位方向向量 | 单位向量 |
| `orientation` | 手掌朝向 | 四元数 |

### 1.2 LEAP_DIGIT（每根手指）
| 字段 | 含义 |
|------|------|
| `finger_id` | 手指 ID |
| `bones[4]` | **metacarpal / proximal / intermediate / distal**（4 个骨节）|
| `is_extended` | ✅ 手指是否伸展 |

> ⚠️ SDK 注释：**拇指的 metacarpal 是"零长度"骨节**（编程方便，解剖上不准确）

### 1.3 LEAP_BONE（每个骨节）
| 字段 | 含义 | 单位 |
|------|------|------|
| `prev_joint` | 骨节起点（靠近心脏）| mm |
| `next_joint` | 骨节终点（远离心脏）| mm |
| `width` | 骨骼周围肉的平均宽度 | mm |
| `rotation` | **世界空间旋转四元数** | (x,y,z,w) |

### 1.4 坐标系
- **右手坐标系，原点 = 追踪相机**
- **单位：毫米（mm）**
- 固定坐标（不随图像变化）

---

## 2. MediaPipe HandLandmarker 输出什么（实测）

```
result.hand_landmarks        # 21 点，归一化图像坐标 (x,y,z + visibility/presence)
result.hand_world_landmarks  # 21 点，米制 3D（原点=手部几何中心）
result.handedness            # 左右手 + 置信度
```
- 21 点：WRIST + 5 指 × (MCP/PIP/DIP/TIP)，拇指为 CMC/MCP/IP/TIP
- 单位：归一化 [0,1] 或 **米（world）**

---

## 3. 直接对比

| 维度 | **Leap Motion Controller 2** | MediaPipe（单目 RGB）|
|------|---------------------------|---------------------|
| **传感器** | 专用红外立体（双目）| 普通 RGB 单目 |
| **输出类型** | 手部**3D 骨架**（关节位置 + 旋转四元数）| 21 关键点 |
| **关节粒度** | 每指 **4 骨节**（metacarpal/proximal/intermediate/distal）| 每指 MCP/PIP/DIP/TIP（拇指 CMC/MCP/IP）|
| **单位 / 坐标系** | **mm，相机坐标系（固定）** | 归一化图像 / 米（手中心为原点）|
| **手掌位置** | ✅ 直接给（mm + 速度 + 法向 + 朝向）| ❌ 只能从关键点推 |
| **抓握/捏合强度** | ✅ **直接给** `grab_strength` / `pinch_strength` | ❌ 需自己算 |
| **手指伸展** | ✅ `is_extended` | ❌ 需自己算（阈值）|
| **关节角**（Flex/Adb）| ❌ **不直接给**，需从骨节位置/四元数换算 | ❌ 不直接给，需从关键点换算 |
| **手部速度** | ✅ `palm.velocity` | ❌ |
| **左右手** | ✅ `type` | ✅ `handedness`（带置信度）|
| **置信度** | ⚠️ `confidence` **恒为 1.0（未使用）** | ✅ `visibility`/`presence` |
| **精度** | **高**（专用红外立体 + 深度）| 中（单目 3D 估计）|
| **遮挡鲁棒性** | 较好 | 明显退化 |
| **FOV / 距离** | 160°×160°，10–110 cm | 取决于相机镜头 |
| **成本** | ¥1500–3000（供货不稳）| **¥0**（现有相机）|
| **实时刻录** | ✅ ~100+ FPS | ✅ CPU 30+ FPS |

---

## 4. ⭐ 最关键的一点：**两者都不直接输出"关节角"**

**Leap 给的是**：每根骨节的 `prev_joint` / `next_joint`（3D 点）+ `rotation`（四元数）
**MediaPipe 给的是**：21 个 3D 关键点

论文里的 **R16/R21 关节角**（`{finger}_{joint}_{Flex/Adb}`）在两者上**都需要二次计算**：

```
关节角 = 相邻两段骨节向量的夹角
  Leap:       angle(bone[i].next_joint - bone[i].prev_joint,
                    bone[i+1].next_joint - bone[i+1].prev_joint)
  MediaPipe:  angle(P[i+1]-P[i], P[i+2]-P[i+1])
             （用 hand_world_landmarks 的米制坐标）
```

**差异在于**：
- Leap 的骨节位置是**红外立体直接测得**（含深度）→ 角度更准
- MediaPipe 的 3D 是**单目估计**（有深度歧义）→ 角度误差更大

---

## 5. 选型建议

| 你的目的 | 建议 |
|---------|------|
| **验证"视觉这条路"、做场景条件** | **MediaPipe（¥0）**，直接开跑 |
| 要**高精度手指关节角真值** | **数据手套**（你已有，比 Leap 更直接）|
| 要"无接触 + 高精度 + 手掌速度/朝向" | Leap Motion 2（但供货/驱动是坑）|
| 要"抓握/捏合强度"这类高层特征 | Leap 直接给（MediaPipe 要自己算）|

## 6. 一个实用洞察

Leap 的 `grab_strength` / `pinch_strength` / `is_extended` 是**已经算好的高层语义**，而 MediaPipe 只给裸关键点。所以：

- **如果你想快速得到"手势状态"标签** → Leap 省事
- **如果你要做自己的关节角定义**（如论文的 R16/R21）→ 两者都要自己算，**但你会有一套统一的换算代码，两个来源都能用**

## 7. 文件

| 文件 | 说明 |
|------|------|
| `MEDIAPIPE_HAND_OUTPUT.md` | MediaPipe 输出说明（实测）|
| `mediapipe_hand_probe.py` | MediaPipe 探测脚本 |
| 本文件 | Leap vs MediaPipe 对比 |
| `Fingers_Gestures_Recognition/LeapSDK/include/LeapC.h` | Leap 官方 SDK 头文件（字段来源）|
