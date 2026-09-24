# 范围调整：手部标定 + 实时显示（大疆相机可行性）

> 老师意见：**不做视觉伺服控制**，只做**手部标定**；能**实时在电脑上显示手部状态**即可。
> 你的问题：**用大疆相机能实现实时在电脑上显示吗？**
>
> **答案：能。** 大疆多个型号支持标准 UVC 协议，插 USB 就能被电脑当摄像头读。

---

## 一、⭐ 大疆相机：官方确认支持 UVC（当作电脑摄像头）

依据 [DJI 官方支持文档](https://support.dji.com/help/content?customId=en-us03400006962&spaceId=34&re=US&lang=en)：

> **USB Video Class (UVC)** 是 USB 视频采集设备的标准协议。
> **Osmo Action 6 / Osmo Action 5 Pro / Osmo Pocket 4 / Osmo Pocket 3 /
> Osmo Action 4 / Osmo Action 3 / DJI Action 2** 都可以在电脑上作为摄像头工作。

| 型号 | UVC 支持 |
|------|---------|
| Osmo Action 6 | ✅ |
| Osmo Action 5 Pro | ✅ |
| Osmo Pocket 4 | ✅ |
| Osmo Pocket 3 | ✅ |
| Osmo Action 4 | ✅ |
| Osmo Action 3 | ✅ |
| DJI Action 2 | ✅ |
| 老款 Osmo Action(1) | ⚠️ 需确认 |
| 大疆无人机（Mini/Air/Mavic）| ❌ 不是 UVC，需 RC + app |

**使用方式**（官方）：
1. USB-C 连接电脑
2. 相机屏幕上选 **"网络摄像头 / Webcam"** 模式
3. 电脑上识别为 **"UVC Camera"** 设备

**→ 既然是标准 UVC，OpenCV 可以直接读**：
```python
import cv2
cap = cv2.VideoCapture(0)        # 或试 1,2,3... 直到拿到 UVC Camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
while True:
    ok, frame = cap.read()
    ...
```

**支持平台**（官方）：Windows（VooV / OBS / PotPlayer）、macOS（Photo Booth）

---

## 二、⚠️ 关键坑：大疆是**超广角/鱼眼**，会毁掉手部估计精度

DJI Osmo Action 系列 FOV 约 **155°** → **强鱼眼畸变**。
而 **MediaPipe / WiLoR 都假设"正常透视相机"**（小畸变），鱼眼输入会显著降低精度。

**解决（两个都必须做）**：
| # | 做法 |
|---|------|
| **1** ⭐ | **相机里切到「标准 FOV / Standard」视角**（DJI 菜单里有 Wide / Standard 选项），关掉超广角 |
| **2** | **自己标定内参 + 畸变系数**（棋盘格 / `cv2.calibrateCamera`），做 `cv2.undistort` 后再喂给手部模型 |

**只做 1 通常够用；要论文级精度就再做 2。**

---

## 三、实时性：模型必须换

| 模型 | 速度 | 适合实时显示？ |
|------|------|--------------|
| **MediaPipe Hands** | **30–60+ FPS**（CPU 就够）| ✅ **首选** |
| WiLoR (ViT-H, 2.4 GB) | ~2–3 FPS（TITAN V）| ❌ 不能实时 |

### 推荐架构（实时显示手部状态）

```
DJI (UVC) ──USB-C──> PC
   │
   ├─ OpenCV 读帧（1080p30）
   ├─ (可选) undistort 去鱼眼畸变
   ├─ MediaPipe Hands → 21 关键点 + 左右手 + 置信度
   │                      + world landmarks（3D，相对手部）
   ├─ 计算手部状态：掌心位置 / 掌心法向 / 手长轴 / 关节角 / 张开度
   ├─ 实时可视化：骨架 + 坐标轴 + 数值 / 曲线
   └─ (可选) WiLoR 异步精修：每 1–2 秒跑一次，覆盖显示
```

**关键**：MediaPipe 负责**实时**，WiLoR 负责**精度**（异步，不阻塞显示）。
如果你只要"实时显示手部状态"，**MediaPipe 单独就够了**。

---

## 四、延迟与硬件选择

| 方案 | 延迟 | 视角 | 深度 | 备注 |
|------|------|------|------|------|
| **DJI Osmo（UVC）** | ~50–150 ms | 超广角 | ❌ | 防抖好、自带电池、可头戴 |
| **USB 网络摄像头** | ~30–80 ms | 窄 | ❌ | 便宜、标准透视、低延迟 |
| 手机（DroidCam/Iriun）| 不稳 | 中 | ❌ | 临时可用 |
| HDMI 采集卡 + 微单 | ~30–60 ms | 可换镜头 | ❌ | 画质最高，贵且重 |
| ⭐ **Intel RealSense D435i/D455** | ~30 ms | 窄 | ✅ **有深度** | **内参已标定 + 直接给 3D** |
| DJI 无线图传 | +100–300 ms | — | ❌ | **不适合实时** |

**⭐ 如果做「3D 手部标定」，RealSense 是更优解**：
- 内参/畸变已标定好，不用自己标
- 直接给**深度** → 3D 手部关键点不用单目估计
- 正好解决之前讨论的"物体深度"问题
- 缺点：FOV 窄（约 87°×58°），需要手在视野内

---

## 五、需要你确认的 4 件事

| # | 问题 | 为什么重要 |
|---|------|-----------|
| **1** | **大疆具体型号？** | 决定是否支持 UVC（上面表格）|
| **2** | ⭐ **"手部标定"具体指什么？** | 见下表，三者差别很大 |
| **3** | **需要 2D 还是 3D 手部状态？** | 决定是否要深度 / WiLoR |
| **4** | **有线 USB 还是无线？** | 无线会加 100–300 ms，影响实时性 |

### "手部标定"的三种可能含义

| 解释 | 内容 | 实现难度 |
|------|------|---------|
| **A. 手部姿态估计** | 实时估计并显示手的位置/朝向/关节角 | ✅ 容易（MediaPipe）|
| **B. 相机↔手/假肢坐标系标定** | 求相机坐标系到假肢坐标系的变换（外参）| 🔶 中（需参考物/标定板）|
| **C. 手部模型参数标定** | per-user 标定（如 MANO shape 参数、手长）| 🔶 中 |

**从"实时显示手部状态"推测，你要的是 A**。但请确认。

---

## 六、一句话回答你

> **能。** 大疆 **Osmo Action 3/4/5/6、Osmo Pocket 3/4、DJI Action 2** 都官方支持
> **UVC 协议**，USB-C 连电脑后直接识别为 "UVC Camera" → **OpenCV 可直接读** → 实时处理没问题。
>
> **但两个必须注意**：
> ① ⚠️ **大疆是超广角/鱼眼** → 会显著降低手部估计精度 → **相机里切"标准 FOV"**，必要时自己标定去畸变
> ② ⚠️ **WiLoR 太慢（2–3 FPS）不能实时** → 实时显示要用 **MediaPipe Hands（30–60 FPS）**
>
> **如果做 3D 标定，RealSense 更优**（内参已标定 + 直接给深度）。
