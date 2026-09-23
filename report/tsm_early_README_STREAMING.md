# 流式视频 Transformer（KV cache）— 调研 + 实现 + 验证

> 目标：为多模态架构准备**流式视觉分支**（配 KV cache），并验证**能与 EMG 做 cross-attention**。
> 目录：`/mnt/storage/siat_cjh/proj/tsm_visual/`　日期：2026-09-14

---

## 1. 调研结论：没有现成的轻量「流式视频 backbone」

检索了 GitHub / awesome-list（`Awesome-Streaming-Video-Understanding`、`Awesome-VLM-Streaming-Video`），结论：

| 类型 | 代表 | 是否适用 |
|------|------|---------|
| **流式视频 LLM** | VideoLLM-online(682★)、LiveCC、Streamo、ThinkStream | ❌ 太重（LLM 底座），不直接给逐帧特征 |
| **在线动作检测（预提取特征）** | **OadTR**(ICCV21)、**LSTR**(NeurIPS21)、realtime-action-detection | ❌ 吃**预提取特征**（非端到端），且依赖 PyTorch 1.6/1.7 |
| **流式视频分割** | SAM-2 real-time | ❌ 任务不符 |
| ~~StreamingCLIP~~ | — | ❌ 仓库已不可达 |

**故采用「成熟组件组合」路线**（这也是业界标准做法）：
- 空间编码：**timm** ViT（预训练可用）
- 时间建模：**自实现因果注意力 + KV cache**
- 参考：VideoLLM-online 的流式/KV cache 设计（已 clone 到 `proj/videollm-online`）

## 2. 实现的架构

```
帧序列 (B,T,3,H,W)
  → 逐帧空间编码: timm ViT (vit_small_patch16_224, d=384)   ← 共享权重
  → 帧级 token (B,T,384)
  → 因果时间注意力 × N 层 (CausalTemporalBlock, 带 KV cache)
  → z_v(t) (B,T,384)   ← 直接送 cross-attention 与 EMG 融合
  → 分类头 (B,T,C)
```

**流式推理**：`reset_cache()` → 逐帧 `stream_step(frame)`，O(1) 增量，无需重算历史。

## 3. 文件

| 文件 | 说明 |
|------|------|
| `streaming_video_transformer.py` | ★ 流式视频 Transformer（含 CausalTemporalBlock + KV cache）|
| `test_fusion_streaming.py` | ★ 视觉+EMG+因果 cross-attention 融合验证 |
| `tsm_backbone.py` / `test_tsm*.py` | 上一轮的 TSM 方案（保留，可切换对比）|
| `README_TSM.md` | TSM 报告 |
| `README_STREAMING.md` | 本文件 |

## 4. 验证结果（全部通过）

### 测试①：流式 KV cache 正确性
```
[整段] z_v=(2,6,384) logits=(2,6,6)
[流式] z_v=(2,6,384) logits=(2,6,6)
[一致性] 最大差异 z=1.222e-06  logits=5.960e-07   ✓
参数量 24.0M | d_model=384
```

### 测试②：视觉 + EMG + 融合（含流式与延迟）
```
[整段] z_v=(2,6,384) z_e=(2,6,128) logits=(2,6,6)
[流式] logits=(2,6,6) | 与整段差异 5.364e-07      ✓
[训练] loss=1.8195 | 视觉梯度范数和=94.434         ✓
[延迟] 每帧「视觉+EMG+融合」18.2 ms（@30fps 预算 33 ms）✓
[规模] 视觉 24.0M + EMG 0.51M + 融合 0.40M
```

## 5. ⚠️ 本次发现的关键工程问题（重要）

**融合模块也必须做成因果的！**

第一版用标准 `nn.MultiheadAttention` 做融合，结果：
```
整段 vs 流式 差异 = 0.24   ← 不一致！
```
原因：整段时视觉帧 t 能 attend **所有** EMG 帧；流式时只能 attend **历史** EMG 帧。

改成 **因果 cross-attention + KV cache**（`CausalCrossFusion`）后 → 差异降到 **5.4e-07**。

> 这条经验直接对应你架构里的第 1 条工程要点（因果性）——**不只是视觉/EMG 分支，融合层同样要因果**。

## 6. 用法

```python
from streaming_video_transformer import StreamingVideoTransformer
from test_fusion_streaming import EMGEncoder, CausalCrossFusion

vis = StreamingVideoTransformer(num_temp_layers=2, num_classes=0).cuda()
emg = EMGEncoder(in_ch=16, d_model=128).cuda()
fuse = CausalCrossFusion(d_v=vis.d_model, d_e=emg.out_dim, num_classes=6).cuda()

# --- 训练/整段 ---
z_v, _ = vis(video)              # (B,T,3,H,W) -> (B,T,384)
z_e = emg(emgx)                  # (B,T,16,640) -> (B,T,128)
logits = fuse(z_v, z_e)          # (B,T,6)

# --- 流式推理 ---
vis.reset_cache(); fuse.reset_cache()
for frame, emg_win in stream:
    z_v_t, _ = vis.stream_step(frame)        # (B,3,H,W) -> (B,1,384)
    z_e_t = emg(emg_win.unsqueeze(1))        # (B,1,128)
    logits_t = fuse.stream_step(z_v_t, z_e_t)
```

## 7. 复现命令

```bash
cd /mnt/storage/siat_cjh/proj/tsm_visual
PY=/home/siat_cjh/miniconda3/envs/AnyEMG/bin/python
$PY streaming_video_transformer.py    # 流式一致性
$PY test_fusion_streaming.py          # 视觉+EMG 融合 + 延迟
```
依赖：`timm==1.0.29`（已装，含 huggingface_hub/safetensors）

## 8. 与 TSM 方案对比

| 项 | TSM-ResNet18 | **流式视频 Transformer** |
|----|-------------|----------------------|
| 延迟 | ~267 ms（T=8 片段）| **18.2 ms/帧** ✓ |
| 流式 | ❌ 片段级，需缓存 | ✅ 真流式（KV cache）|
| 与 EMG cross-attention | 可用但段落级 | ✅ 逐帧对齐，天然契合 |
| 参数量 | 11.2M | 24.0M |
| 预训练 | ImageNet | timm（可加载 ImageNet/更多）|

**结论：你的架构应按流式 Transformer 走**（延迟满足 33 ms 预算、逐帧特征便于与 EMG 对齐融合）。

## 9. 下一步可选

1. **加载 timm 预训练权重**（`pretrained=True`，需网络；或用 `HF_ENDPOINT=https://hf-mirror.com`）
2. **在 UCF101 子集上跑真实训练**（验证收敛）
3. **接上 EMG 真实数据**（当前 EMG 分支用随机输入做了接口验证）
4. **加快捷推理**（torch.compile / FP16）进一步降延迟
