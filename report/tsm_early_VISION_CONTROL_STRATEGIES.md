# 视觉 + EMG 假肢/机械手控制策略调研

> 目的：回答"视觉检测手-物接近 + EMG 触发抓握"这类控制策略，学术界是怎么做的、哪种有实际意义
> 检索源：OpenAlex API（web 搜索额度不可用）　日期：2026-09-15

---

## 1. 六种控制范式（从"人在环"到"全自主"）

| # | 范式 | 决策权 | 视觉的作用 | 代表工作 |
|---|------|--------|-----------|---------|
| **1** | **直接 EMG 控制**（传统）| 人 100% | 无（纯肌电）| 传统 myoelectric |
| **2** | **接近/触发式** ⭐ | 人触发 + 机器执行 | 检测手-物接近/重合 | 你说的方案 |
| **3** | **半自主**（Semi-autonomous）⭐ | 机器提议 + 人可否决 | 自主选抓握方式 | **2022 RAS**（-25.9% 负荷）|
| **4** | **共享控制**（Shared control）| 人机混合 | 视觉贡献控制信号 | 2018 综述（391 引）|
| **5** | **意图预测**（Intent inference）| 机器预测 + 人确认 | 提前预测抓握类型 | Zandigohar 2024 |
| **6** | **全自主**（Autonomous）| 机器 100% | 接近传感器 + DL 识别 | 2024 TMRB |

**你说的方案属于 2/3**（接近触发 → EMG 确认 → 执行），这正是当前**最有实际意义且最容易落地**的一档。

---

## 2. 关键论文的具体做法（摘要原文提取）

### ① Semi-autonomous control of prosthetic hands（2022, Robotics and Autonomous Systems）
> DOI: 10.1016/j.robot.2022.104123

**做法**：
- **单个 EMG 通道** + **手内嵌多模态传感器**（object perception）
- 系统**自主选择并执行抓握**，用户保持在环中，可随时**触发 / 接受 / 拒绝**
- 20 人用户研究，对比传统 EMG 控制

**结果**：**认知负荷降低 25.9%、体力需求降低 60%**

**启示**：**单通道 EMG 就够触发**——你不需要几十个通道；关键是"接近感知 + 自主执行 + 人可否决"。

### ② Explorations of Autonomous Prosthetic Grasping via Proximity Vision（2024, IEEE TMRB）
> DOI: 10.1109/tmrb.2024.3377530

**做法**：
- **雷达传感器 + 低分辨率 ToF 相机**（不是普通高分辨率相机！）
- 用深度学习在 **HANDdata**（人-物交互数据集，聚焦 reach-to-grasp）上训练
- 覆盖**静态与动态**场景

**结论**：**接近传感器（雷达/ToF）可作为相机的替代或补充**——低分辨率、低功耗就够用

**启示**：做"接近检测"**不需要高端相机**，甚至雷达就够。你用手边相机或头戴相机都绰绰有余。

### ③ Cognitive vision system for control of dexterous prosthetic hands（2010, 149 引）
- 用**认知视觉系统**（物体识别）驱动灵巧假肢控制

### ④ A Review of Intent Detection, Arbitration, and Communication Aspects of Shared Control（2018, 391 引）
- **综述**：把共享控制拆成三要素——**意图检测 / 仲裁 / 沟通**
- 是你写 Related Work 时的核心引用

---

## 3. 控制策略的四个关键设计要素（决定你的方案做得好不好）

### ① 触发条件（Trigger）
| 设计 | 说明 | 风险 |
|------|------|------|
| 仅距离阈值 | 手-物距离 < d | **易误触发**（手路过物体）|
| 距离 + 停留时间 | 距离小且持续 > t | 稍好，但慢 |
| **距离 + EMG 确认** ⭐ | 接近 且 肌肉发力 | **最稳**（双条件）|
| 距离 + 凝视 | 接近 且 看着它 | 需要眼动仪 |

**推荐**：**距离 + EMG 双条件**（视觉负责"该抓了"，EMG 负责"现在抓"）

### ② 仲裁（Arbitration）—— 人 vs 机器谁说了算
- **人优先级**：用户随时可覆盖（安全、可控）
- **机器优先级**：机器自主，人只能否决
- **混合**：按置信度加权（综述的核心议题）

**推荐**：**人可否决**（用户握拳/放松即可中断）

### ③ 反馈（Communication）
用户需要知道"系统要做什么"：
- 视觉（屏幕/LED）
- 听觉（提示音）
- **触觉（振动）** ← 假肢常用

**推荐**：至少加一个"即将抓握"的提示（否则用户不知道机器意图）

### ④ 安全与失效处理
- 误触发后如何退出？
- 视觉失效（遮挡/暗光）时怎么办？→ **回退到纯 EMG 模式**（这正好是你多模态的另一半价值）

---

## 4. ⭐ 给你的建议：做一个"接近触发 + EMG 确认"的半自主抓握

### 为什么这条最简单且有实际意义
| 理由 | 说明 |
|------|------|
| **有文献支撑** | 2022 半自主论文证明能降 25.9% 负荷、60% 体力 |
| **技术门槛低** | 手-物检测用 MediaPipe + 物体检测即可；EMG 单通道够 |
| **你已有条件** | 头戴/普通相机 + 6ch EMG + 数据手套（真值）|
| **解决真问题** | 减少认知负荷、防误触发、加快响应 |
| **可扩展** | 后续可加"抓握类型选择"（视觉给 affordance）|

### 具体控制状态机（可直接实现）
```
[IDLE] 手远离物体
   ↓ 视觉: 手-物距离 < d1
[APPROACH] 接近中
   ↓ 视觉: 距离 < d2 且 EMG 幅值 > 阈值（意图确认）
[TRIGGER] 触发抓握
   ↓ 执行: 视觉建议的抓握类型（或固定抓握）
[GRASP] 抓握保持
   ↓ EMG 放松 / 距离增大
[RELEASE] 释放 → 回 IDLE

异常分支:
   - 视觉失效（遮挡/暗光）→ 回退纯 EMG 控制 + 提示
   - 用户主动中断（急握）→ 立即释放
   - 距离抖动 → 加时间去抖（连续 N 帧才触发）
```

### 关键参数（需实验标定）
| 参数 | 说明 |
|------|------|
| `d1`（接近阈值）| 开始"准备"的距离 |
| `d2`（触发阈值）| 允许触发的距离 |
| EMG 阈值 | 意图确认的肌电幅值（建议 MVC 归一化）|
| 去抖帧数 | 连续多少帧满足才触发（防抖）|

---

## 5. 你能做的"有意义"的方向（按难度排）

| 方向 | 内容 | 难度 | 意义 |
|------|------|------|------|
| **A. 接近触发 + EMG 确认** ⭐ | 上述状态机 | 低 | 直接降负荷/防误触发 |
| **B. A + 抓握类型选择** | 视觉识别物体 → 建议抓握方式 | 中 | 更自然（免去手动切换）|
| **C. 意图提前预测** | reach 前期就预测抓握（Zandigohar 路线）| 中高 | 更快响应 |
| **D. 视觉失效时的 EMG 兜底** | 遮挡/暗光下融合更稳 | 中 | **这是多模态的核心卖点** |
| **E. 全自主抓握** | 雷达/ToF + DL 自动抓 | 高 | 最激进 |

**建议从 A 起步**（1–2 个月可出结果），把 **D 作为论文的核心论证**（视觉退化时 EMG 撑住），这与前面讨论的"融合有效性"完全一致。

---

## 6. 核心参考文献（可直接引用）

| 年份 | 标题 | 出处 | 引用 |
|------|------|------|------|
| 2022 | Semi-autonomous control of prosthetic hands based on multimodal sensing, human grasp demonstration and user intention | Robotics and Autonomous Systems | 58 |
| 2024 | Explorations of Autonomous Prosthetic Grasping via Proximity Vision and Deep Learning | IEEE TMRB | 7 |
| 2018 | A Review of Intent Detection, Arbitration, and Communication Aspects of Shared Control | (综述) | 391 |
| 2010 | Cognitive vision system for control of dexterous prosthetic hands | (实验评估) | 149 |
| 2017 | Deep learning-based artificial vision for grasp classification in myoelectric hands | | 183 |
| 2023 | A Review of Myoelectric Control for Prosthetic Hand Manipulation | | 95 |
| 2020 | Gaze, visual, myoelectric, and inertial data of grasps for intelligent prosthetics | | 41 |

> DOI 已在上文给出；`A Review of Shared Control`(2018) 与 `Semi-autonomous`(2022) 建议优先精读。

---

## 7. 一句话结论

**你老师说的"检测手位置和物体距离、重合时配合 EMG 完成抓握"，在文献里叫「接近触发的半自主控制」（proximity-triggered semi-autonomous control）——有成熟论文支撑（2022 那篇证明了 -25.9% 认知负荷 / -60% 体力），技术门槛低（普通相机 + 单通道 EMG 就够），而且天然引出你的多模态卖点（视觉失效时 EMG 兜底）。**

**建议：以「接近触发 + EMG 确认」的状态机为骨架，把「视觉退化下的鲁棒性」作为论文核心论证。**
