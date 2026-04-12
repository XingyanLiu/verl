# VeRL核心算法代码解析 - 优先级1

> **面试重点**: Losses (SFT, DPO, Policy Loss, Value Loss, KL Loss, Entropy Loss, Policy Distillation) 和 Advantage Estimation (GAE, GRPO)

---

## 目录

1. [损失函数 (Losses)](#1-损失函数-losses)
   - [1.1 SFT Loss](#11-sft-loss)
   - [1.2 PPO Policy Loss](#12-ppo-policy-loss)
   - [1.3 Value Loss](#13-value-loss)
   - [1.4 KL Loss / KL Penalty](#14-kl-loss--kl-penalty)
   - [1.5 Entropy Loss](#15-entropy-loss)
   - [1.6 Policy Distillation Loss](#16-policy-distillation-loss)
2. [优势估计 (Advantage Estimation)](#2-优势估计-advantage-estimation)
   - [2.1 GAE (Generalized Advantage Estimation)](#21-gae-generalized-advantage-estimation)
   - [2.2 GRPO (Group Relative Policy Optimization)](#22-grpo-group-relative-policy-optimization)

---

## 1. 损失函数 (Losses)

### 1.1 SFT Loss

**文件位置**: `verl/workers/utils/losses.py: 28-54`

**数学公式**:
$$\mathcal{L}_{SFT} = -\frac{1}{N} \sum_{i=1}^{N} \log \pi_\theta(y_i | x, y_{<i})$$

**代码实现**:
```python
def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """
    计算监督微调损失 (Supervised Fine-Tuning Loss)
    
    输入:
    - model_output["log_probs"]: 模型输出的log概率
      - shape: (bsz, seq_len) 或 nested tensor [bsz, j1]
    - data["loss_mask"]: 损失掩码，标记哪些token参与损失计算
      - shape: (bsz, seq_len) 或 nested tensor [1, prompt_length + response_length]
    
    输出:
    - loss: 标量tensor，平均负对数似然
    """
    log_prob = model_output["log_probs"]
    
    if pad_mode == DatasetPadMode.NO_PADDING:
        # 处理nested tensor格式
        log_prob_flatten = log_prob.values()          # (total_tokens,)
        loss_mask_flatten = loss_mask.values()         # (total_tokens,)
        
        # 左移loss_mask一个token，与log_prob对齐（预测下一个token）
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)
        
        # 核心计算: -sum(log_prob * mask) / batch_num_tokens * dp_size
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        # 标准padding模式
        response_mask = data["response_mask"].to(bool)  # (bsz, seq_len)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size
    
    return loss, {}
```

**维度追踪**:
| 变量 | Shape | 说明 |
|------|-------|------|
| `log_prob` | `(bsz, seq_len)` | 每个token的log概率 |
| `loss_mask` | `(bsz, seq_len)` | 1表示参与计算，0表示忽略 |
| `loss` | `scalar` | 标量损失值 |

**操作解释**:
- `masked_sum`: 只对mask为1的位置求和
- `/batch_num_tokens * dp_size`: 全局归一化，确保分布式训练时损失一致

---

### 1.2 PPO Policy Loss

**文件位置**: `verl/trainer/ppo/core_algos.py: 1278-1369`

**数学公式** (PPO Clipped Objective):
$$\mathcal{L}^{CLIP}(\theta) = -\mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

其中重要性采样比率:
$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} = \exp(\log\pi_\theta - \log\pi_{\theta_{old}})$$

**代码实现**:
```python
@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,  # (batch_size, response_length)
    log_prob: torch.Tensor,      # (batch_size, response_length)
    advantages: torch.Tensor,    # (batch_size, response_length)
    response_mask: torch.Tensor, # (batch_size, response_length)
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    PPO Clipped Policy Loss with Dual-Clip
    
    输入维度:
    - old_log_prob: (bsz, response_len) - 旧策略的log概率
    - log_prob: (bsz, response_len) - 当前策略的log概率  
    - advantages: (bsz, response_len) - 优势估计
    - response_mask: (bsz, response_len) - 有效token掩码
    
    输出:
    - pg_loss: scalar - 策略梯度损失
    - pg_metrics: dict - 包含clipfrac, kl等指标
    """
    
    # Step 1: 计算重要性采样比率 r(θ) = π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob  # (bsz, response_len)
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)  # 数值稳定
    ratio = torch.exp(negative_approx_kl)  # (bsz, response_len)
    
    # Step 2: 计算KL散度 (近似)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)  # scalar
    
    # Step 3: 标准PPO clipping
    # 无裁剪损失: -A * r(θ)
    pg_losses1 = -advantages * ratio  # (bsz, response_len)
    
    # 裁剪损失: -A * clip(r(θ), 1-ε, 1+ε)
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # (bsz, response_len)
    
    # 取max确保保守更新
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # (bsz, response_len)
    
    # Step 4: Dual-clip (针对负优势，限制ratio下界)
    pg_losses3 = -advantages * clip_ratio_c  # clip_ratio_c默认为3.0
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    
    # Step 5: 组合最终损失
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    
    # Step 6: 聚合损失
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, 
                       loss_agg_mode=loss_agg_mode, **config.global_batch_info)
    
    return pg_loss, pg_metrics
```

**维度追踪**:
| 变量 | Shape | 说明 |
|------|-------|------|
| `old_log_prob` | `(bsz, response_len)` | rollout时的log概率 |
| `log_prob` | `(bsz, response_len)` | 当前策略的log概率 |
| `ratio` | `(bsz, response_len)` | 重要性采样比率 |
| `advantages` | `(bsz, response_len)` | 优势值 |
| `pg_losses` | `(bsz, response_len)` | 每个token的策略损失 |
| `pg_loss` | `scalar` | 聚合后的标量损失 |

**关键操作**:
- `clip`: 限制策略更新幅度，防止过大更新破坏性能
- `dual-clip`: 对于负优势(A<0)，额外限制ratio下界

---

### 1.3 Value Loss

**文件位置**: `verl/trainer/ppo/core_algos.py: 2084-2123`

**数学公式** (Clipped Value Loss):
$$\mathcal{L}^{V}(\phi) = \frac{1}{2} \mathbb{E}_t \left[ \max \left( (V_\phi(s_t) - R_t)^2, (V^{clip}_\phi(s_t) - R_t)^2 \right) \right]$$

其中:
$$V^{clip}_\phi(s_t) = V_{\phi_{old}}(s_t) + \text{clip}(V_\phi(s_t) - V_{\phi_{old}}(s_t), -\epsilon_v, \epsilon_v)$$

**代码实现**:
```python
def compute_value_loss(
    vpreds: torch.Tensor,       # (batch_size, response_length)
    returns: torch.Tensor,      # (batch_size, response_length) - 目标return
    values: torch.Tensor,       # (batch_size, response_length) - 旧的value预测
    response_mask: torch.Tensor,# (batch_size, response_length)
    cliprange_value: float,     # value clipping范围
    loss_agg_mode: str = "token-mean",
):
    """
    PPO Value Function Clipped Loss
    
    输入维度:
    - vpreds: (bsz, response_len) - 当前value head预测
    - values: (bsz, response_len) - rollout时的value预测
    - returns: (bsz, response_len) - 目标returns (GAE计算得到)
    
    输出:
    - vf_loss: scalar - value function损失
    - vf_clipfrac: scalar - 被clip的比例
    """
    
    # Step 1: Clip value预测，限制更新幅度
    vpredclipped = verl_F.clip_by_value(
        vpreds, 
        values - cliprange_value, 
        values + cliprange_value
    )  # (bsz, response_len)
    
    # Step 2: 计算两种MSE损失
    vf_losses1 = (vpreds - returns) ** 2       # 无clip的MSE
    vf_losses2 = (vpredclipped - returns) ** 2 # clip后的MSE
    
    # Step 3: 取max (保守更新)
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)  # (bsz, response_len)
    
    # Step 4: 聚合 (乘0.5是标准MSE系数)
    vf_loss = 0.5 * agg_loss(
        loss_mat=clipped_vf_losses, 
        loss_mask=response_mask, 
        loss_agg_mode=loss_agg_mode
    )  # scalar
    
    # 计算clip比例统计
    vf_clipfrac = verl_F.masked_mean(
        torch.gt(vf_losses2, vf_losses1).float(), response_mask
    )
    
    return vf_loss, vf_clipfrac
```

**维度追踪**:
| 变量 | Shape | 说明 |
|------|-------|------|
| `vpreds` | `(bsz, response_len)` | 当前Critic预测 |
| `values` | `(bsz, response_len)` | Rollout时的Critic预测 |
| `returns` | `(bsz, response_len)` | GAE计算的目标值 |
| `vf_losses` | `(bsz, response_len)` | 每token的value损失 |
| `vf_loss` | `scalar` | 聚合后损失 |

---

### 1.4 KL Loss / KL Penalty

**文件位置**: `verl/trainer/ppo/core_algos.py: 2126-2187`

**数学公式** (多种KL估计器):

| 类型 | 公式 | 特点 |
|------|------|------|
| k1 (forward KL) | $\log \pi_\theta - \log \pi_{ref}$ | 简单，有偏 |
| abs | $|\log \pi_\theta - \log \pi_{ref}|$ | 对称 |
| k2 (MSE) | $\frac{1}{2}(\log \pi_\theta - \log \pi_{ref})^2$ | 无偏梯度 |
| k3 (low-var) | $r - \log r - 1$, where $r = \frac{\pi_{ref}}{\pi_\theta}$ | 低方差 |

**代码实现**:
```python
def kl_penalty(
    logprob: torch.FloatTensor,      # (bsz, response_len) - 当前策略log概率
    ref_logprob: torch.FloatTensor,  # (bsz, response_len) - 参考策略log概率
    kl_penalty: str                   # 选择的KL类型: "kl"/"k1", "abs", "mse"/"k2", "k3"
) -> torch.FloatTensor:
    """
    计算KL惩罚项 (用于约束策略与参考策略的距离)
    
    输入维度:
    - logprob: (bsz, response_len) - π_θ(a|s) 的log
    - ref_logprob: (bsz, response_len) - π_ref(a|s) 的log
    
    输出:
    - kl_estimate: (bsz, response_len) - 每token的KL估计
    """
    
    # K1 估计器: log π_θ - log π_ref (forward KL的采样估计)
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob  # (bsz, response_len)
    
    # 绝对值估计器
    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()
    
    # K2 估计器: MSE (提供无偏梯度)
    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()
    
    # K3 估计器: 低方差KL (Schulman 2020)
    # KL = E_π_ref[log(π_ref/π_θ)] ≈ r - log(r) - 1
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob  # log(π_ref/π_θ)
        kl = torch.clamp(kl, min=-20, max=20)  # 数值稳定
        ratio = torch.exp(kl)  # r = π_ref/π_θ
        kld = (ratio - kl - 1).contiguous()  # r - log(r) - 1
        return torch.clamp(kld, min=-10, max=10)
```

**在PPO Loss中的使用** (`verl/workers/utils/losses.py: 123-134`):
```python
# 在ppo_loss函数中
if config.use_kl_loss:
    ref_log_prob = data["ref_log_prob"]
    
    # 计算KL散度
    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, 
                     kl_penalty=config.kl_loss_type)  # (bsz, response_len)
    
    # 聚合KL损失
    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, 
                       loss_agg_mode=config.loss_agg_mode, **config.global_batch_info)
    
    # 加入总损失 (带系数)
    policy_loss += kl_loss * config.kl_loss_coef
```

**维度追踪**:
| 变量 | Shape | 说明 |
|------|-------|------|
| `logprob` | `(bsz, response_len)` | 当前策略log概率 |
| `ref_logprob` | `(bsz, response_len)` | 参考策略log概率 |
| `kld` | `(bsz, response_len)` | 每token的KL估计 |
| `kl_loss` | `scalar` | 聚合后的KL损失 |

---

### 1.5 Entropy Loss

**文件位置**: `verl/trainer/ppo/core_algos.py: 2067-2081`

**数学公式**:
$$\mathcal{L}_{entropy} = -\mathbb{E}_t[H(\pi_\theta(\cdot|s_t))] = -\mathbb{E}_t\left[-\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)\right]$$

**代码实现**:
```python
def compute_entropy_loss(
    logits: torch.Tensor,       # (bs, response_length, vocab_size)
    response_mask: torch.Tensor,# (bs, response_length)
    loss_agg_mode: str = "token-mean"
):
    """
    计算分类熵损失 (鼓励探索)
    
    输入维度:
    - logits: (bsz, response_len, vocab_size) - 模型原始输出
    - response_mask: (bsz, response_len) - 有效token掩码
    
    输出:
    - entropy_loss: scalar - 负熵(用于最小化=最大化熵)
    """
    
    # 计算每个token位置的熵 H = -Σ p*log(p)
    token_entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_len)
    
    # 聚合
    entropy_loss = agg_loss(
        loss_mat=token_entropy, 
        loss_mask=response_mask, 
        loss_agg_mode=loss_agg_mode
    )  # scalar
    
    return entropy_loss
```

**在PPO Loss中的使用** (`verl/workers/utils/losses.py: 114-121`):
```python
# 在ppo_loss函数中
if entropy is not None:
    entropy_loss = agg_loss(
        loss_mat=entropy, loss_mask=response_mask, 
        loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    entropy_coeff = config.entropy_coeff
    
    # 减去熵损失 = 鼓励高熵(探索)
    policy_loss -= entropy_coeff * entropy_loss
```

**维度追踪**:
| 变量 | Shape | 说明 |
|------|-------|------|
| `logits` | `(bsz, response_len, vocab_size)` | 原始模型输出 |
| `token_entropy` | `(bsz, response_len)` | 每token的熵 |
| `entropy_loss` | `scalar` | 聚合后的熵 |

---

### 1.6 Policy Distillation Loss

**文件位置**: `verl/trainer/distillation/losses.py: 221-281` 和 `verl/trainer/distillation/fsdp/losses.py: 26-77`

**数学公式** (Forward KL Distillation):
$$\mathcal{L}_{distill} = \text{KL}(P_{teacher} \| Q_{student}) = \sum_{k \in \text{top-K}} P_k \log \frac{P_k}{Q_k}$$

**代码实现 - KL散度计算**:
```python
# verl/trainer/distillation/fsdp/losses.py
def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """
    计算KL散度 KL(P||Q)
    
    输入:
    - log_q: (bsz, seqlen, topk) - student的log概率
    - log_p: (bsz, seqlen, topk) - teacher的log概率
    
    输出:
    - kld: (bsz, seqlen) - 每个位置的KL散度
    """
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()  # teacher概率
    kld = p * (log_p - log_q)  # P * log(P/Q)
    return kld.sum(dim=-1)  # 在topk维度求和


def compute_forward_kl_topk(
    student_logits: torch.Tensor,      # (bsz, seqlen/sp_size, vocab_size)
    teacher_topk_log_probs: torch.Tensor, # (bsz, seqlen, topk)
    teacher_topk_ids: torch.Tensor,    # (bsz, seqlen, topk)
    config: DistillationConfig,
    data_format: str,
) -> dict:
    """
    Top-K Forward KL蒸馏损失
    
    输入维度:
    - student_logits: (bsz, seqlen/sp_size, vocab_size) - 学生模型logits
    - teacher_topk_log_probs: (bsz, seqlen, topk) - 教师top-k log概率
    - teacher_topk_ids: (bsz, seqlen, topk) - 教师top-k token IDs
    
    输出:
    - distillation_losses: (bsz, seqlen/sp_size) - 每位置的蒸馏损失
    - student_mass: (bsz, seqlen/sp_size) - 学生在top-k上的概率质量
    - teacher_mass: (bsz, seqlen/sp_size) - 教师在top-k上的概率质量
    """
    
    # Step 1: 学生logits -> log_softmax
    student_log_probs = F.log_softmax(student_logits, dim=-1)  # (bsz, seqlen, vocab)
    
    # Step 2: 提取学生在teacher top-k token上的log概率
    student_topk_log_probs = torch.gather(
        student_log_probs, dim=-1, index=teacher_topk_ids
    )  # (bsz, seqlen, topk)
    
    # Step 3: 计算概率质量 (用于监控)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)  # (bsz, seqlen)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)  # (bsz, seqlen)
    
    # Step 4: 计算Forward KL: KL(teacher || student)
    distillation_losses = kl_divergence(
        log_q=student_topk_log_probs, 
        log_p=teacher_topk_log_probs
    )  # (bsz, seqlen)
    
    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass
    }
```

**Reverse KL 估计器** (`verl/trainer/distillation/losses.py: 322-354`):
```python
@register_distillation_loss(
    DistillationLossSettings(
        names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], 
        use_estimator=True
    )
)
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    使用单样本KL估计器计算蒸馏损失
    
    输入:
    - model_output["log_probs"]: (bsz, response_len) - 学生采样token的log概率
    - data["teacher_logprobs"]: (bsz, response_len, 1) - 教师在同一token上的log概率
    
    输出:
    - distillation_losses: (bsz, response_len) - 蒸馏损失
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    
    # 使用与KL penalty相同的估计器 (k1, k2, k3等)
    distillation_losses = kl_penalty(
        logprob=student_log_probs, 
        ref_logprob=teacher_log_probs, 
        kl_penalty=loss_config.loss_mode
    )
    
    return distillation_losses, metrics
```

**Policy Gradient结合蒸馏** (`verl/trainer/distillation/losses.py: 253-279`):
```python
if loss_config.use_policy_gradient:
    # 将负蒸馏损失作为reward, 使用PPO更新
    # 参考: https://thinkingmachines.ai/blog/on-policy-distillation/
    policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
    
    distillation_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=-distillation_losses.detach(),  # 负损失作为优势
        response_mask=response_mask,
        ...
    )
else:
    # 直接反向传播蒸馏损失 (监督学习方式)
    distillation_loss = agg_loss(
        loss_mat=distillation_losses,
        loss_mask=response_mask,
        ...
    )
```

**维度追踪**:
| 变量 | Shape | 说明 |
|------|-------|------|
| `student_logits` | `(bsz, seqlen, vocab_size)` | 学生模型输出 |
| `teacher_topk_log_probs` | `(bsz, seqlen, topk)` | 教师top-k概率 |
| `distillation_losses` | `(bsz, seqlen)` | 每位置蒸馏损失 |
| `distillation_loss` | `scalar` | 聚合后损失 |

---

## 2. 优势估计 (Advantage Estimation)

### 2.1 GAE (Generalized Advantage Estimation)

**文件位置**: `verl/trainer/ppo/core_algos.py: 215-263`

**数学公式**:
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

递归形式:
$$\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}$$

**代码实现**:
```python
@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    values: torch.Tensor,               # (bs, response_length)
    response_mask: torch.Tensor,        # (bs, response_length)
    gamma: torch.Tensor,                # 折扣因子
    lam: torch.Tensor,                  # GAE lambda参数
):
    """
    计算Generalized Advantage Estimation
    
    输入维度:
    - token_level_rewards: (bsz, response_len) - 每token奖励
    - values: (bsz, response_len) - Value function预测
    - response_mask: (bsz, response_len) - 有效token掩码
    - gamma: scalar - 折扣因子 (如0.99)
    - lam: scalar - GAE lambda (如0.95)
    
    输出:
    - advantages: (bsz, response_len) - 优势估计
    - returns: (bsz, response_len) - 目标returns
    """
    with torch.no_grad():
        nextvalues = 0    # 终止状态的V(s')=0
        lastgaelam = 0    # 终止状态的A=0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        
        # 从后向前遍历 (时间反向)
        for t in reversed(range(gen_len)):
            # TD误差: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            
            # GAE递归: A_t = δ_t + γλA_{t+1}
            lastgaelam_ = delta + gamma * lam * lastgaelam
            
            # 处理mask: 跳过非响应token
            nextvalues = values[:, t] * response_mask[:, t] + \
                        (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + \
                        (1 - response_mask[:, t]) * lastgaelam
            
            advantages_reversed.append(lastgaelam)
        
        # 反转得到正确顺序
        advantages = torch.stack(advantages_reversed[::-1], dim=1)  # (bsz, response_len)
        
        # Returns = Advantages + Values (用于训练Value function)
        returns = advantages + values  # (bsz, response_len)
        
        # 标准化优势 (减均值除标准差)
        advantages = verl_F.masked_whiten(advantages, response_mask)
        
    return advantages, returns
```

**维度追踪**:
| 变量 | Shape | 说明 |
|------|-------|------|
| `token_level_rewards` | `(bsz, response_len)` | token级奖励 |
| `values` | `(bsz, response_len)` | Critic预测值 |
| `delta` | `(bsz,)` | 当前时间步TD误差 |
| `lastgaelam` | `(bsz,)` | 累积GAE优势 |
| `advantages` | `(bsz, response_len)` | 最终优势估计 |
| `returns` | `(bsz, response_len)` | Value训练目标 |

**关键操作**:
- 时间反向遍历: 利用递归公式从终止状态向起始状态计算
- `masked_whiten`: 仅对有效位置进行标准化

---

### 2.2 GRPO (Group Relative Policy Optimization)

**文件位置**: `verl/trainer/ppo/core_algos.py: 267-331`

**数学公式**:
对于同一prompt生成的多个response, GRPO计算组内相对优势:

$$A_i = \frac{r_i - \mu_{group}}{\sigma_{group} + \epsilon}$$

或Dr.GRPO变体 (不除以标准差):
$$A_i = r_i - \mu_{group}$$

**代码实现**:
```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    response_mask: torch.Tensor,        # (bs, response_length)
    index: np.ndarray,                  # (bs,) - prompt分组索引
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,  # 是否除以标准差
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO优势估计: 组内相对奖励
    
    输入维度:
    - token_level_rewards: (bsz, response_len) - token级奖励
    - response_mask: (bsz, response_len) - 有效token掩码
    - index: (bsz,) - 每个样本属于哪个prompt组
    
    输出:
    - advantages: (bsz, response_len) - 优势估计
    - returns: (bsz, response_len) - 与advantages相同
    
    说明:
    - GRPO是Outcome-only方法，只用response级别的总奖励
    - 同一prompt的多个response组成一组，计算组内均值和标准差
    """
    
    # Step 1: 计算每个response的总奖励
    scores = token_level_rewards.sum(dim=-1)  # (bsz,)
    
    # Step 2: 按prompt分组计算统计量
    id2score = defaultdict(list)  # prompt_id -> list of scores
    id2mean = {}
    id2std = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        
        # 收集每组的分数
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # 计算每组的均值和标准差
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # 单样本组: 不减均值，不除标准差
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        # Step 3: 计算归一化优势
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                # 标准GRPO: (r - μ) / (σ + ε)
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                # Dr.GRPO变体: r - μ
                scores[i] = scores[i] - id2mean[index[i]]
        
        # Step 4: 广播到token级别
        scores = scores.unsqueeze(-1) * response_mask  # (bsz, response_len)
    
    return scores, scores  # advantages和returns相同
```

**向量化版本** (`verl/trainer/ppo/core_algos.py: 334-358`):
```python
@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(...):
    """
    向量化GRPO实现 (更高效)
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)  # (bsz,)
        
        # 使用向量化group操作
        g = as_torch_index(index, device=scores.device)  # 转换为tensor索引
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon, device=scores.device)
        
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages
```

**维度追踪**:
| 变量 | Shape | 说明 |
|------|-------|------|
| `token_level_rewards` | `(bsz, response_len)` | token级奖励 |
| `scores` | `(bsz,)` | 每response总奖励 |
| `index` | `(bsz,)` | prompt分组索引 |
| `mean_g` | `(num_groups,)` | 每组均值 |
| `std_g` | `(num_groups,)` | 每组标准差 |
| `advantages` | `(bsz, response_len)` | 归一化后的优势 |

**与GAE的关键区别**:
| 特性 | GAE | GRPO |
|------|-----|------|
| 是否需要Critic | ✓ 需要 | ✗ 不需要 |
| 奖励类型 | Token-level | Outcome-level |
| 基线 | V(s) | 组内均值 |
| 适用场景 | 过程奖励 | 结果奖励 |

---

## 附录: 聚合损失函数 agg_loss

**文件位置**: `verl/trainer/ppo/core_algos.py: 1138-1199`

```python
def agg_loss(
    loss_mat: torch.Tensor,    # (bs, response_length) - 损失矩阵
    loss_mask: torch.Tensor,   # (bs, response_length) - 掩码
    loss_agg_mode: str,        # 聚合模式
    dp_size: int = 1,          # 数据并行大小
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    聚合损失到标量，支持多种模式:
    
    - "token-mean": sum(loss) / total_tokens
    - "seq-mean-token-sum": mean_over_seqs(sum_over_tokens(loss))
    - "seq-mean-token-sum-norm": 同上，但除以loss_scale_factor
    - "seq-mean-token-mean": mean_over_seqs(mean_over_tokens(loss))
    """
    if loss_agg_mode == "token-mean":
        loss = masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode in ["seq-mean-token-sum", "seq-mean-token-sum-norm"]:
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # (bsz,)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
        loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
        if loss_agg_mode == "seq-mean-token-sum-norm":
            loss /= loss_scale_factor
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)
        seq_mask = (seq_mask > 0).float()
        loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
    return loss
```
