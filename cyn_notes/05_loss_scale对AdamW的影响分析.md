# Loss Scale 变更对 AdamW 优化器的影响分析

> 本文档分析 verl 版本升级（v0.3.x → v0.7+）导致的 loss scale 变化对训练的理论和实践影响。

---

## 背景

verl v0.7+ 引入了全局归一化机制（`agg_loss`），使得 loss 值"并行度无关"（parallelism-agnostic）。
假设新版本框架在 `token-mean` 模式下计算的 loss 是旧版本的 **1/K**
（K 取决于 gradient accumulation steps 数和并行设置，本文假设 K=32 作为示例）。

**两个分析场景**：

| 场景 | 总 loss 公式 | 说明 |
|------|-------------|------|
| 场景一 | `L = L_pg` | 只有 pg_loss |
| 场景二 | `L = L_pg - α·L_ent` | pg_loss + entropy_loss（α = entropy_coeff） |

---

## 第一部分：梯度传播链路分析

### 1.1 完整的 loss → gradient 链路

根据源码分析（`verl/workers/actor/dp_actor.py`, `verl/workers/utils/losses.py`），
完整的 loss 到梯度的链路如下：

```
┌───────────────────────────────────────────────────────────────────────────┐
│ (1) agg_loss: 全局归一化                                                   │
│     pg_loss_new = masked_sum(loss_mat, mask) / batch_num_tokens * dp_size │
│     entropy_loss_new = agg_loss(entropy, ..., **global_batch_info)        │
│     policy_loss = pg_loss_new - α * entropy_loss_new                     │
├───────────────────────────────────────────────────────────────────────────┤
│ (2) 梯度累积缩放                                                          │
│     loss = policy_loss * loss_scale_factor                               │
│     loss_scale_factor = 1 / gradient_accumulation_steps                  │
├───────────────────────────────────────────────────────────────────────────┤
│ (3) backward                                                             │
│     loss.backward()  或  scaler.scale(loss).backward()                   │
├───────────────────────────────────────────────────────────────────────────┤
│ (4) 梯度裁剪                                                             │
│     clip_grad_norm_(parameters, max_norm=grad_clip)  # 默认 1.0          │
├───────────────────────────────────────────────────────────────────────────┤
│ (5) AdamW optimizer step                                                 │
│     m = β₁m + (1-β₁)g                                                   │
│     v = β₂v + (1-β₂)g²                                                  │
│     θ = θ - lr * (m̂/(√v̂+ε) + λθ)   # λ = weight_decay                  │
└───────────────────────────────────────────────────────────────────────────┘
```

### 1.2 关键观察

**agg_loss 的缩放和梯度累积缩放是独立的两层缩放。**

来源于代码 `dp_actor.py` line 585 和 663：
```python
# 第一层：agg_loss 内部（全局归一化）
policy_loss = ppo_loss(...)  # 已经被 agg_loss 用 global tokens 归一化

# 第二层：梯度累积缩放
loss_scale_factor = 1 / self.gradient_accumulation
loss = policy_loss * loss_scale_factor  # 再乘以梯度累积因子

loss.backward()
```

所以**最终参与 backward 的 loss 经历了两次缩放**。在分析时需要区分这两层。

---

## 第二部分：场景一 — 仅有 pg_loss

### 2.1 设定

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 学习率 (lr) | η |
| 权重衰减 (λ) | 0.01 |
| β₁, β₂ | 0.9, 0.999 |
| ε | 1e-8 |
| grad_clip | 1.0 |
| Loss 缩放因子 | K = 32 (新版本 loss 是旧版本的 1/32) |

### 2.2 理论分析

#### AdamW 的更新规则

```
g_t = ∇_θ L_t                         # 梯度
m_t = β₁ · m_{t-1} + (1-β₁) · g_t     # 一阶矩估计
v_t = β₂ · v_{t-1} + (1-β₂) · g_t²    # 二阶矩估计
m̂_t = m_t / (1 - β₁^t)                # 偏差修正
v̂_t = v_t / (1 - β₂^t)                # 偏差修正
θ_{t+1} = (1 - η·λ) · θ_t - η · m̂_t / (√v̂_t + ε)
```

#### 当梯度被缩放 1/K 时

设旧版本梯度为 `g`，新版本梯度为 `g' = g/K`：

```
m'_t = β₁ · m'_{t-1} + (1-β₁) · g/K = m_t / K
v'_t = β₂ · v'_{t-1} + (1-β₂) · (g/K)² = v_t / K²
```

AdamW 的自适应步长：

```
m̂'_t / (√v̂'_t + ε)
= (m_t/K) / (√(v_t/K²) + ε)
= (m_t/K) / (√v_t/K + ε)
= m_t / (√v_t + K·ε)           ← 关键！
```

对比旧版本的：
```
m̂_t / (√v̂_t + ε)
= m_t / (√v_t + ε)
```

#### ⚡ 核心结论

```
新版本的有效更新 = m_t / (√v_t + K·ε)
旧版本的有效更新 = m_t / (√v_t + ε)

差异比 = (√v_t + K·ε) / (√v_t + ε)
```

**两种极端情况**：

| 条件 | 差异比 | 说明 |
|------|--------|------|
| √v̂_t >> K·ε（大梯度） | ≈ 1 | 几乎无差异 |
| √v̂_t << K·ε（小梯度） | ≈ K = 32 | **有效步长缩小 32 倍！** |
| √v̂_t ≈ K·ε（临界区域） | ≈ 2 | 有显著差异 |

### 2.3 数值分析：ε 的影响

AdamW 默认 ε = 1e-8。当 K = 32 时：

```
K·ε = 32 × 1e-8 = 3.2e-7
```

**什么时候 √v̂_t 会接近 3.2e-7？**

v̂_t 是梯度的二阶矩（移动平均），√v̂_t 大致反映梯度的量级。

- 如果参数的梯度量级在 **1e-3 ~ 1e-1**（典型 RL 训练），√v̂_t ≈ 1e-3 ~ 1e-1
  - 远大于 3.2e-7 → **几乎无差异** ✅
- 如果参数的梯度量级在 **1e-6 ~ 1e-7**（几乎不活跃的参数），√v̂_t ≈ 1e-6 ~ 1e-7
  - 接近或小于 3.2e-7 → **有显著差异** ⚠️
- 对于完全冻结/几乎不更新的参数：差异最大

**实际影响评估**：对于大多数活跃参数，**ε 效应可以忽略**。但对于某些不活跃的参数（如冻结的 embedding 层、很少更新的 bias 等），可能会有细微差异。

### 2.4 梯度裁剪的影响

verl 默认使用 `grad_clip = 1.0`（L2 范数裁剪）：

```python
grad_norm = clip_grad_norm_(parameters, max_norm=1.0)
```

当梯度缩小 K 倍时，grad_norm 也缩小 K 倍。

| 版本 | 典型 grad_norm | 是否被裁剪 |
|------|---------------|-----------|
| v0.3.x | 例如 5.0 | 是，裁剪到 1.0（缩放 5 倍） |
| v0.7+ | 5.0/32 ≈ 0.156 | **否**，不需裁剪 |

**影响**：
- 旧版本中梯度经常被裁剪，起到了正则化的作用
- 新版本中梯度很少被裁剪，**失去了这层隐式正则化**
- **这可能导致训练不稳定**，特别是在梯度突然增大的时候

**⚡ 这是一个重要的实践差异！**

### 2.5 权重衰减的相对影响

AdamW 的更新公式中，权重衰减项与自适应梯度项是独立的：

```
θ_{t+1} = (1 - η·λ) · θ_t - η · m̂_t / (√v̂_t + ε)
           ───────────────   ─────────────────────────
           权重衰减项          自适应梯度项
```

当梯度缩小但权重衰减不变时：
- 权重衰减的绝对大小不变：`η·λ·θ_t`
- 自适应梯度项的绝对大小几乎不变（因为 AdamW 的自适应性，见 2.2 节）

**在场景一中，因为只有 pg_loss，权重衰减的相对效果不受影响。**

### 2.6 场景一结论

| 影响因素 | 严重程度 | 说明 |
|---------|---------|------|
| AdamW ε 效应 | ⚪ 低 | 对活跃参数几乎无影响 |
| 梯度裁剪行为变化 | 🔴 高 | 裁剪频率大幅降低，失去隐式正则化 |
| 权重衰减相对效果 | ⚪ 低 | AdamW 解耦设计，不受影响 |
| loss 数值本身 | ⚪ 低 | 仅影响日志可读性 |

**场景一的主要风险来自梯度裁剪行为的变化。**

---

## 第三部分：场景二 — pg_loss + entropy_loss

### 3.1 设定

总 loss = `L_pg - α · L_ent`

其中 α = `entropy_coeff`（配置项，默认为 0，典型设置为 0.01 ~ 0.1）。

### 3.2 两个 loss 的缩放关系

**关键事实**：`pg_loss` 和 `entropy_loss` 都使用**相同的 `agg_loss` 函数**和**相同的 `global_batch_info`**。

来源：`verl/workers/utils/losses.py` line 96-121：
```python
# pg_loss
pg_loss, pg_metrics = policy_loss_fn(
    ..., loss_agg_mode=loss_agg_mode, config=config, ...
)

# entropy_loss
entropy_loss = agg_loss(
    loss_mat=entropy, loss_mask=response_mask,
    loss_agg_mode=loss_agg_mode,
    **config.global_batch_info        # ← 同样的 global_batch_info
)

policy_loss = pg_loss - entropy_coeff * entropy_loss
```

因此：
```
L_pg_new = L_pg_old / K
L_ent_new = L_ent_old / K

policy_loss_new = L_pg_new - α · L_ent_new
                = L_pg_old/K - α · L_ent_old/K
                = (L_pg_old - α · L_ent_old) / K
                = policy_loss_old / K
```

### 3.3 ⚡ 核心结论：pg_loss 与 entropy_loss 的相对比例不变

```
L_pg_new / L_ent_new = L_pg_old / L_ent_old
```

**这意味着两个 loss 之间的权重关系不受版本升级影响！** 
`entropy_coeff` 的有效作用保持不变。

### 3.4 理论分析：对 AdamW 的影响

由于 `policy_loss` 整体缩放 1/K，其对参数的梯度也整体缩放 1/K：

```
g_new = ∂(policy_loss_new)/∂θ = (∂policy_loss_old/∂θ) / K = g_old / K
```

这里的梯度包含了 pg_loss 和 entropy_loss 的混合贡献。
与场景一的分析完全一致 — AdamW 的自适应性可以抵消均匀缩放，
**关键差异仍然来自 ε 效应和梯度裁剪**。

### 3.5 如果只有 pg_loss 缩放而 entropy_loss 不缩放？（假设性分析）

**注意：当前代码中两者都缩放，这里仅作假设性分析以加深理解。**

如果只有 pg_loss 缩放了 1/K：

```
policy_loss_hypothetical = L_pg_old/K - α · L_ent_old
```

此时：
- 梯度 = `(1/K)·∂L_pg/∂θ - α·∂L_ent/∂θ`
- pg_loss 的梯度贡献变弱，entropy_loss 的梯度贡献不变
- **相当于 entropy_coeff 有效放大了 K 倍**

这会导致：
- 策略过度偏向高熵方向（过度探索）
- pg_loss 的优化信号被淹没
- 训练可能退化为随机策略

**庆幸的是，verl 的实现避免了这个问题 — 两个 loss 使用完全相同的归一化。**

### 3.6 场景二结论

| 影响因素 | 严重程度 | 说明 |
|---------|---------|------|
| pg_loss 与 entropy_loss 相对权重 | ✅ 无影响 | 两者同步缩放 |
| AdamW ε 效应 | ⚪ 低 | 同场景一 |
| 梯度裁剪行为变化 | 🔴 高 | 同场景一 |
| 权重衰减相对效果 | ⚪ 低 | 同场景一 |
| loss 数值可读性 | ⚪ 低 | 仅影响日志 |

---

## 第四部分：实践层面的影响

### 4.1 计算精度问题

#### BFloat16 精度

verl 默认使用 BFloat16 进行前向计算：
```python
# verl/workers/engine/fsdp/transformer_impl.py
param_dtype = torch.bfloat16    # 模型参数
reduce_dtype = torch.float32     # 梯度归约
```

BFloat16 的特性：
- 指数位：8 位（与 FP32 相同），范围：~1e-38 到 ~3e38
- 尾数位：7 位（FP32 为 23 位）
- **最小正规格化数**：~1.18e-38
- **精度**：约 3-4 位有效数字

**loss 缩小 32 倍的影响**：
```
旧版 loss ≈ 0.01 → 新版 loss ≈ 3.125e-4
旧版 loss ≈ 0.001 → 新版 loss ≈ 3.125e-5
```

这些值远在 BFloat16 的表示范围内，不存在下溢问题。
但**精度损失会略微增大**，因为较小的数值在 BFloat16 中的相对精度较低。

#### FP16 精度（如果使用）

如果使用 FP16（verl 支持但非默认）：
- 最小正规格化数：~6.1e-5
- `loss ≈ 3.125e-5` 已经接近下溢风险！
- verl 使用 `ShardedGradScaler` 来处理 FP16 下溢，但额外的 loss 缩小可能增加 scaler 的压力

**建议**：如果使用 FP16，需要特别注意 GradScaler 的行为。

#### 梯度精度

由于梯度归约使用 FP32（`reduce_dtype = torch.float32`），
梯度在 AllReduce 过程中不会有精度问题。
但在 backward 过程中，如果中间结果使用 BFloat16，较小的梯度可能丢失精度。

### 4.2 梯度裁剪阈值

这是**最重要的实践差异**。

**verl 默认 `grad_clip = 1.0`**（`verl/workers/config/actor.py` line 294）。

| 场景 | 旧版 grad_norm 估计 | 新版 grad_norm 估计 | 裁剪行为 |
|------|--------------------|--------------------|---------|
| 正常训练 | ~2.0 | ~0.0625 | 旧版裁剪，新版不裁剪 |
| 梯度尖峰 | ~100 | ~3.125 | 旧版重度裁剪，新版轻度裁剪 |
| 小梯度 | ~0.1 | ~0.003 | 两版都不裁剪 |

**实际后果**：

1. **正常训练中**：新版本梯度很少触发裁剪
   - 好处：更完整地保留梯度信息
   - 坏处：失去了裁剪提供的隐式正则化和稳定性

2. **异常梯度时**：新版本的裁剪保护减弱
   - 旧版：100 → 1.0（100 倍压缩）
   - 新版：3.125 → 1.0（仅 3 倍压缩）
   - 虽然都裁剪了，但新版压缩比小得多

3. **训练稳定性**：如果依赖梯度裁剪来稳定训练，升级后可能出问题

### 4.3 优化器状态初始化

AdamW 的一阶矩 `m` 和二阶矩 `v` 初始化为 0：
```
m_0 = 0, v_0 = 0
```

在训练初期（前几步），偏差修正会放大这些估计：
```
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
```

**梯度缩小 K 倍时的初始阶段行为**：

```
步骤 1：
  v̂₁ = (1-β₂)·g'² / (1-β₂) = g'² = g²/K²
  √v̂₁ = |g|/K
  如果 |g|/K >> ε：更新步长 ≈ lr · sign(g)（与旧版一致）
  如果 |g|/K ≈ ε：更新步长偏小
```

**对于典型的 RL 梯度（量级 1e-3 ~ 1e-1）**：
- `|g|/K = |g|/32 ≈ 3e-5 ~ 3e-3`
- ε = 1e-8
- `|g|/K >> ε` → 初始阶段行为一致 ✅

### 4.4 日志和监控

新版本报告的 `actor/pg_loss` 值会小 K 倍，但这是经过设计的：

```python
# verl/workers/utils/losses.py line 111
metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
```

当 `metric_aggregation = AggregationType.SUM` 时（dp_size > 1），
metric 值会在不同 workers 间**求和**而非求平均。
这意味着最终 W&B/TensorBoard 上显示的 loss 值 = pg_loss × dp_size / dp_size = pg_loss。

**注意**：`metric_aggregation` 的设置可能影响最终显示的值。
建议同时监控 `actor/ppo_kl` 和 `actor/pg_clipfrac`，这些 metric 不受 loss scale 影响。

### 4.5 学习率调度器交互

如果使用 cosine 或其他 learning rate scheduler，schedule 本身不受 loss scale 影响。
但如果用户根据旧版本的 loss 值设置了 early stopping 或 adaptive lr 等条件，需要更新阈值。

---

## 第五部分：数学补充 — AdamW 的近似尺度不变性

### 5.1 严格推导

设原始梯度序列为 g₁, g₂, ..., gₜ，缩放后为 g'ᵢ = gᵢ/K。

**一阶矩**：
```
m'_t = Σᵢ β₁^(t-i) · (1-β₁) · gᵢ/K = m_t / K
```

**二阶矩**：
```
v'_t = Σᵢ β₂^(t-i) · (1-β₂) · (gᵢ/K)² = v_t / K²
```

**更新量**（无偏差修正，简化表示）：
```
Δ'_t = -lr · m'_t / (√v'_t + ε)
     = -lr · (m_t/K) / (√(v_t/K²) + ε)
     = -lr · (m_t/K) / (√v_t/K + ε)
     = -lr · m_t / (√v_t + K·ε)

Δ_t  = -lr · m_t / (√v_t + ε)
```

**比值**：
```
Δ'_t / Δ_t = (√v_t + ε) / (√v_t + K·ε)
```

### 5.2 不同参数量级下的影响

设某参数的 RMS 梯度为 σ_g（即 √v̂_t ≈ σ_g）：

| σ_g 量级 | √v̂_t | K·ε (K=32) | Δ'/Δ 比值 | 有效 lr 变化 |
|----------|-------|-----------|-----------|-------------|
| 1e-1 | 0.1 | 3.2e-7 | ≈ 1.0000 | 无变化 |
| 1e-2 | 0.01 | 3.2e-7 | ≈ 1.0000 | 无变化 |
| 1e-3 | 0.001 | 3.2e-7 | ≈ 1.0000 | 无变化 |
| 1e-4 | 1e-4 | 3.2e-7 | ≈ 0.9997 | ~0.03% |
| 1e-5 | 1e-5 | 3.2e-7 | ≈ 0.969 | ~3% |
| 1e-6 | 1e-6 | 3.2e-7 | ≈ 0.76 | ~24% |
| 1e-7 | 1e-7 | 3.2e-7 | ≈ 0.31 | **~69%** |

**结论**：只有当参数的 RMS 梯度小于 ~1e-5 时，ε 效应才变得显著。
对于 LLM 的大部分参数（梯度量级 1e-4 ~ 1e-1），影响可忽略。

---

## 第六部分：总结与结论

### 6.1 场景一（仅 pg_loss）

| 维度 | 理论影响 | 实践影响 | 严重性 |
|------|---------|---------|--------|
| AdamW 自适应步长 | 近似不变（ε 效应可忽略） | 对活跃参数无影响 | ⚪ 低 |
| 梯度裁剪 | 裁剪阈值有效放大 K 倍 | 裁剪频率大幅降低，稳定性可能下降 | 🔴 高 |
| 权重衰减 | 绝对量不变（AdamW 解耦） | 无影响 | ⚪ 低 |
| 数值精度 | BF16 下无下溢风险 | FP16 下需注意 | 🟡 中 |
| 训练日志 | loss 数值变小 K 倍 | 需更新监控阈值 | ⚪ 低 |

### 6.2 场景二（pg_loss + entropy_loss）

| 维度 | 理论影响 | 实践影响 | 严重性 |
|------|---------|---------|--------|
| loss 间相对权重 | **完全不变** | entropy_coeff 有效作用不变 | ✅ 无影响 |
| 其他影响 | 同场景一 | 同场景一 | 同场景一 |

### 6.3 最终结论

**理论上**：
- AdamW 的自适应机制可以自动补偿均匀的 loss 缩放，对大多数参数无影响
- 多个 loss 项（pg_loss + entropy_loss）之间的相对权重不受影响
- 唯一的理论差异来自 AdamW 的 ε 参数，但对典型的 LLM 梯度量级可忽略

**实践上**：
- **🔴 最大风险：梯度裁剪行为变化**。如果旧版本依赖梯度裁剪来稳定训练，升级后需要调整 `grad_clip`
- 🟡 FP16 场景下需注意数值精度
- ⚪ 日志中的 loss 值需要重新解读

**结论：如果只看 AdamW + 同步缩放的 pg_loss 和 entropy_loss，理论上训练行为应该几乎不变。
但实践中，梯度裁剪阈值需要同步缩小 K 倍才能保持相同的训练动态。**

---

## 参考源码

| 文件 | 说明 | 关键行号 |
|------|------|---------|
| `verl/workers/utils/losses.py` | ppo_loss：pg_loss + entropy_loss 组合 | 57-136, 特别是 116-121 |
| `verl/trainer/ppo/core_algos.py` | agg_loss：全局归一化实现 | 1138-1199 |
| `verl/workers/actor/dp_actor.py` | 梯度累积 + backward + 裁剪 | 562-585, 659-675, 391-422 |
| `verl/workers/config/optimizer.py` | 优化器配置（默认值） | 33-61, 88-124 |
| `verl/workers/config/actor.py` | grad_clip 默认值 1.0 | 294 |
| `verl/workers/engine/fsdp/transformer_impl.py` | 混合精度配置 | 308-324 |
