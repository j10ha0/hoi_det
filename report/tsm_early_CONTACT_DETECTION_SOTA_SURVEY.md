# 手-物接触检测：公开方案对比（训练目标 + 输出）

> 目标：找到"手-物接触检测"的公开方案，搞清它们的**训练目标**和**输出**
> 日期：2026-09-15　已 clone 验证：3 个仓库

---

## 一、⭐ 三个仓库对比（结论先行）

| 仓库 | 任务 | 训练目标 | 输出 | 训练代码 | 对你 |
|------|------|---------|------|---------|------|
| **`takumayagi/hand_object_contact_prediction`**（BMVC 2021）| **手-物接触预测** | **BCE 二分类** | **接触概率** | ✅ **有** | ⭐⭐⭐⭐⭐ |
| `facebookresearch/ContactOpt`（CVPR 2021）| 接触**优化手姿态** | 接触场优化 | 优化后的手姿态 | ✅ 有 | ⭐ 不适用 |
| `hwjiang1510/GraspTTA`（ICCV 2021）| 抓握**生成** | — | 生成的抓握 | ❌ **未开源** | ⭐ 不适用 |
| ~~`ddshan/hand_detector.d2`~~ | **只检测手** | 目标检测 | 手框 | ✅ 有 | ⭐⭐ 只做手检测 |

**结论：唯一完全对口的是 `hand_object_contact_prediction`。**

---

## 二、⭐ `hand_object_contact_prediction` 详解（唯一对口）

**论文**：*Hand-Object Contact Prediction via Motion-Based Pseudo-Labeling and Guided Progressive Label Correction*（BMVC 2021，Yagi / Hasan / Sato，东京大学）

### 2.1 训练目标
| 项 | 内容 |
|----|------|
| **任务** | **手-物接触预测（二分类）**：接触 / 未接触 |
| **损失** | **`nn.BCELoss()`**（二元交叉熵）|
| **输出** | **接触概率**（0–1）|
| **评测** | accuracy（`total_acc`）|

### 2.2 输入
| 项 | 内容 |
|----|------|
| 模态 | **RGB 帧 + 光流（optical flow）** |
| 额外输入 | **手/物边界框**（bounding boxes）|
| 结构 | **轨迹（track）序列**——按时序喂给 LSTM |
| 数据 | **EPIC-KITCHENS-100** 视频（480p）|

### 2.3 模型
| 模型名 | 说明 |
|--------|------|
| **`UnionLSTMHO`** | 基线（手-物特征 + LSTM）|
| **`CrUnionLSTMHO`** | **提出的模型**（带修正机制）|
| `CrUnionLSTMHORGB` | RGB-only 基线 |
| `CrUnionLSTMHOFlow` | 光流-only 基线 |

→ **核心是 LSTM 时序模型**（不是 Transformer）

### 2.4 方法亮点（对你的启发最大）
| 技术 | 作用 |
|------|------|
| **Motion-Based Pseudo-Labeling** | 用**运动信息**自动生成接触伪标签 → **解决标注难题** |
| **Guided Progressive Label Correction (gPLC)** | 半监督：逐步修正伪标签中的错误 |
| 半监督训练 | 用少量精确标签 + 大量伪标签 |

### 2.5 开销（要注意）
| 项 | 需求 |
|----|------|
| 数据 | EPIC-KITCHENS-100（需自行申请下载）|
| **空间** | 测试 ~100GB / trusted 训练 ~200GB / **完整训练 ~3TB** |
| 依赖 | torch 1.8.1、**FlowNet2**（光流，需 torch 1.4）|
| 其他 | 预训练模型（Google Drive）、伪标签（11GB）|

### 2.6 训练命令
```bash
# 监督版（用 trusted 标签）
python train_v1.py --model UnionLSTMHO --supervised --nb_iters 25000 ...

# 完整版（半监督 + 伪标签修正）
python train.py --model CrUnionLSTMHO --semisupervised --plc --update_clean ...
```

---

## 三、为什么另外两个不适用

### `ContactOpt`（CVPR 2021）
- **任务**：**优化手部姿态**以改善抓握（不是接触检测）
- **输入**：**手网格 + 物体网格**（已经是 3D mesh）
- **输出**：优化后的手姿态
- → **需要 3D 网格 + MANO**，你不做 3D，**不适用**

### `GraspTTA`（ICCV 2021）
- **任务**：人类**抓握生成**（从物体点云生成 MANO 手）
- **训练代码**：README 明说 "*Please email me if you have interest in the training code...*" → **未开源**
- → **不适用**

### `hand_detector.d2`
- **任务**：**只检测手**（`class_names=["hand"]`，1 类）
- **输出**：手框 + 类别 + 置信度
- → 可作为"手检测"组件，**但不做接触判断**

---

## 四、⭐ 对你的三条启示

### ① 训练目标是**二分类**（BCE）—— 和我给你的脚本一致
```
输入: RGB(±光流) + 手/物框 + 时序
输出: 接触概率 (0-1)
损失: BCE
```

### ② **时序**比单帧好 —— 建议加 LSTM/GRU
- 它用 **LSTM 处理轨迹序列**（而非单帧分类）
- 你可以：单帧 ResNet 特征 → LSTM 聚合 → 接触概率
- 这样"接近→接触→松开"的**动态过程**能被建模

### ③ ⭐ **伪标签**解决"负样本不足"的问题
- 该论文用 **motion-based pseudo-labeling**：从**运动模式**自动推断接触状态
- **对你的价值**：EPIC-Contact 只有正样本，但你可以用"运动/时序规律"自动生成伪标签（比如：手和物体的相对运动停止 = 可能接触）
- 这是一个**可写进论文的方法点**

---

## 五、推荐给你的技术路线

```
阶段1（最快）: 单帧二分类
  ResNet18(ImageNet) + 自己的正负样本 → 接触概率
  （脚本已给: train_contact_detector.py）

阶段2（升级）: 时序模型
  ResNet18 逐帧特征 → LSTM/GRU → 接触概率
  （借鉴 hand_object_contact_prediction 的结构）

阶段3（进阶）: 半监督 / 伪标签
  少量精标 + 大量运动伪标签 → gPLC 式修正
  （借鉴它的 motion-based pseudo-labeling）
```

---

## 六、文件

| 文件 | 说明 |
|------|------|
| 本文件 | 公开方案对比 + 训练目标/输出分析 |
| `proj/hand_object_contact_prediction/` | ⭐ clone 的接触预测仓库 |
| `proj/ContactOpt/` | clone 的接触优化仓库 |
| `proj/GraspTTA/` | clone 的抓握生成仓库 |
| `proj/hand_detector.d2/` | clone 的手检测仓库 |
