# TSM 视觉分支 — 环境准备与验证报告

> 目标：为多模态架构（视觉 + EMG）准备可用的 **TSM 视觉方案 + 公开数据集**，并验证 TSM 模块可跑。
> 目录：`/mnt/storage/siat_cjh/proj/tsm_visual/`
> 日期：2026-09-14

---

## 1. 目录结构

```
proj/tsm_visual/
├── temporal-shift-module/        # 官方 TSM 实现（git clone, MIT）
│   ├── ops/temporal_shift.py     # ★ 核心：TemporalShift / make_temporal_shift
│   ├── archs/  main.py  opts.py  # 完整训练框架（依赖 mmaction2，本项目未用）
│   └── README.md                 # 官方说明
├── data/
│   ├── ucf101_subset.tar.gz      # 数据集原始包（163 MB）
│   └── UCF101_subset/            # 解压后：train/ val/ test/ × 10 类 × 31 视频
├── test_tsm.py                   # 测试①：shift 行为 + 前向 + 显存/速度
├── test_tsm_training.py          # 测试②：数据加载 + 训练 + 验证（端到端）
├── tsm_backbone.py               # ★ 可复用封装：TSMBackbone
└── README_TSM.md                 # 本文件
```

## 2. 环境（已验证）

| 项 | 值 |
|----|----|
| Python | 3.9.18（conda env `AnyEMG`）|
| PyTorch | **2.1.0** + CUDA **12.1** |
| GPU | NVIDIA TITAN V ×4（12 GB，测试用 `cuda:1`）|
| 视频解码 | **OpenCV (`cv2`)**（decord / av 未装）|
| TSM 依赖 | **仅 torch + torchvision**（核心模块不依赖 mmaction2/mmcv）|

## 3. 数据集：UCF101 子集（公开）

| 项 | 值 |
|----|----|
| 来源 | HuggingFace `sayakpaul/ucf101-subset`（镜像 `hf-mirror.com` 下载更快）|
| 大小 | 163 MB（**注意**：文件名 `.tar.gz` 但实为**未压缩 tar**）|
| 内容 | **405 个视频 / 10 类**（ApplyEyeMakeup, ApplyLipstick, Archery, BabyCrawling, BalanceBeam, BandMarching, BaseballPitch, Basketball, BasketballDunk, BenchPress）|
| 结构 | `train/ val/ test/` × 类别目录 × `.avi`（ImageFolder 风格）|
| 用途 | TSM 训练流程验证；**正式研究建议换手势专向数据集**（见 §6）|

## 4. 验证结果

### 测试① `test_tsm.py`
```
✓ TemporalShift.shift 行为正确（前1/3 左移取未来帧、中1/3 右移取过去帧、后1/3 不动）
✓ TSM-ResNet18 前向: (16,3,224,224) -> (16,6)，插入 8 个 TemporalShift
✓ TSM-ResNet50 前向: 插入 16 个 TemporalShift
✓ shift 生效（与普通 ResNet 输出差异 1.2982）
⚠ n_segment=8 的模型喂 1 帧会报错（输入帧数必须是 n_segment 的整数倍）
```

### 测试② `test_tsm_training.py`（端到端）
```
[1] 数据加载: 单样本 (8,3,224,224)，0.89 s/video
[2] 训练 3 步: loss 2.5722 -> 1.8237（反向传播正常）
[3] 验证集 top-1: 10.0%（仅 40 训练样本 / 3 步，随机水平=10%）
[4] 训练吞吐: 46.5 iter/s（batch 16 帧 ≈ 744 帧/s）
```

### 测试③ `tsm_backbone.py`
```
resnet18: TSM模块=8  | feat=(16,512)  | logits=(16,6) | clip=(2,512)  | 11.2M 参数
resnet50: TSM模块=16 | feat=(16,2048) | logits=(16,6) | clip=(2,2048) | 23.5M 参数
```

## 5. 用法

```python
from tsm_backbone import TSMBackbone

# n_segment 决定片段长度（T=8 → 8 帧）
backbone = TSMBackbone(arch="resnet18", n_segment=8, num_classes=6).to("cuda:1")

x = torch.randn(2, 8, 3, 224, 224)          # (B, T, 3, H, W)
feat, logits = backbone(x)                   # feat (16,512) / logits (16,6)
clip_feat = backbone.encode_clip(x)          # (2,512) 片段级特征 → 送融合模块
```

融合时：把 `feat`（每帧特征）或 `encode_clip`（片段特征）与 EMG 分支特征做 cross-attention。

## 6. ⚠️ 流式推理的关键限制（与目标架构相关）

**TSM 是"片段级"模块**：`shift` 依赖 `n_segment` 维度重排，因此

| 问题 | 说明 | 对策 |
|------|------|------|
| 输入必须是 T 的整数倍 | 单帧会报错 | 维护长度 T 的**帧缓存** |
| 输出延迟 = T 帧 | T=8 @30fps ≈ **267 ms**，超过目标 33–100 ms | ① 减小 n_segment（T=4 → 133 ms）② 滑动窗口 + 步长<T ③ 换因果 3D CNN / 流式 Transformer |
| 每步需重算整片段 | 无状态复用 | 用**增量缓存 + 低步长**近似 |

**建议**：视觉分支用 TSM 做"特征提取器"（可接受 ~130–270 ms 片段延迟），若严格要 33–100 ms，则把 TSM 换成**因果 3D CNN**或**流式 Transformer**（配 KV cache）。

## 7. 正式研究建议的数据集（手势专向）

UCF101 只用于"跑通流程"。手势课题建议：

| 数据集 | 内容 | 获取 |
|--------|------|------|
| **Jester (20BN)** | 27 类手势，148k 视频 | 需在 20bn 官网注册 |
| **EgoGesture** | 50 类（第一人称，与头戴场景接近）| 需申请 |
| **NVGesture** | 25 类，含深度 | 需申请 |
| **IPN Hand** | 13 类连续手势 | 公开 |
| **Hagrid** | 图像为主（可做静态手势预训练）| HuggingFace 公开 |

## 8. 复现命令

```bash
cd /mnt/storage/siat_cjh/proj/tsm_visual
PY=/home/siat_cjh/miniconda3/envs/AnyEMG/bin/python

$PY test_tsm.py            # 模块行为 + 前向
$PY test_tsm_training.py   # 端到端训练（UCF101 子集）
$PY tsm_backbone.py        # 封装自检
```

数据重新下载（若丢失）：
```bash
curl -L -o data/ucf101_subset.tar.gz \
  "https://hf-mirror.com/datasets/sayakpaul/ucf101-subset/resolve/main/UCF101_subset.tar.gz"
tar -xf data/ucf101_subset.tar.gz -C data/      # 注意：无 z
```

## 9. 参考

- TSM 论文：*TSM: Temporal Shift Module for Efficient Video Understanding*, arXiv:1811.08383
- 官方实现：https://github.com/mit-han-lab/temporal-shift-module
