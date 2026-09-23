# 实验记录 003：ViT-Small（排除容量因素）

- 与 EXP_002 完全同设置，只把骨干 ViT-Tiny → ViT-Small（5.8M → 22M）
- 数据：EgoHOS 伪标签 863 帧（train 543 / val 320）

## ViT-Small 三组结果

| 组 | 参数量_M | 最佳hand_F1 | 末轮hand_F1 | 最佳contact_F1 | 最佳object_F1 | epochs |
|---|---|---|---|---|---|---|
| A_aux(small) | 0 | 0.7456 | 0.6901 | 0.2412 | 0.3414 | 30 |
| B_noaux(small) | 0 | 0.6928 | 0.5765 | — | — | 30 |
| C_aux+featdistill(small) | 0 | 0.725 | 0.6291 | 0.2106 | 0.1451 | 30 |

## 与 ViT-Tiny 对照（跨规模）

- **A**：Tiny 0.6863 → Small 0.7456（+0.0593）
- **B**：Tiny 0.7065 → Small 0.6928（-0.0137）
- **C**：Tiny 0.6945 → Small 0.725（+0.0305）

## 结论（ViT-Small）

- 辅助头：A−B = **+0.0528**
- 特征蒸馏：C−A = **-0.0206**
- 完整 vs 基线：C−B = **+0.0322**
