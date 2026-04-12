# VeRL核心算法代码解析 - 优先级2

> **扩展算法**: ./recipe目录算法 + 其他核心算法变体

---

## 目录

1. [On-Policy Distillation](#1-on-policy-distillation-在线策略蒸馏)
2. [其他Advantage Estimator变体](#2-其他advantage-estimator变体)
   - [RLOO](#21-rloo-reinforcement-learning-from-leave-one-out)
   - [REINFORCE++](#22-reinforce)
   - [ReMax](#23-remax)
   - [GDPO](#24-gdpo-group-decoupled-policy-optimization)
   - [OPO](#25-opo)
   - [Optimal Token Baseline](#26-optimal-token-baseline)
3. [Policy Loss变体](#3-policy-loss变体)
   - [DPPO (Divergence-Constrained PPO)](#31-dppo-divergence-constrained-ppo)
   - [GSPO (Geometric Sequence PPO)](#32-gspo-geometric-sequence-ppo)
   - [SAPO (Smoothed Advantage Policy Optimization)](#33-sapo-smoothed-advantage-policy-optimization)
   - [GPG (Group Policy Gradient)](#34-gpg-group-policy-gradient)
   - [GMPO (Geometric Mean Policy Optimization)](#35-gmpo-geometric-mean-policy-optimization)
   - [CISPO (Clipped Importance Sampling PO)](#36-cispo-clipped-importance-sampling-po)

---

## 注意事项

**关于 `./recipe` 目录**: 经检查，该目录当前为空。本文档涵盖的扩展算法主要来自 `./verl/trainer/ppo/core_algos.py` 和 `./verl/trainer/distillation/` 中的实现。

---

## 1. On-Policy Distillation (在线策略蒸馏)

**文件位置**: 
- 配置: `verl/workers/config/distillation.py`
- 损失函数: `verl/trainer/distillation/losses.py`
- 文档: `docs/advance/async-on-policy-distill.md`

### 1.1 概述

On-Policy Knowledge Distillation训练学生策略模仿更强的教师，使用学生当前策略生成的样本。对于每个on-policy rollout，教师返回soft top-k token分布，学生使用token级sparse KL目标进行优化。

### 1.2 核心配置

```python
# verl/workers/config/distillation.py: 31-112
@dataclass
class DistillationLossConfig(BaseConfig):
    """
    蒸馏损失配置
    """
    loss_mode: str = "k3"  # 损失类型: "k1", "k2", "k3", "forward_kl_topk"等
    topk: Optional[int] = 128  # top-k token数量
    use_task_rewards: bool = True  # 是否结合任务奖励
    distillation_loss_coef: float = 1.0  # 蒸馏损失系数
    loss_max_clamp: Optional[float] = 10.0  # 最大裁剪值
    log_prob_min_clamp: Optional[float] = -10.0  # log概率最小裁剪
    
    # Policy Gradient结合蒸馏
    use_policy_gradient: bool = True  # 是否使用PG方式
    policy_loss_mode: str = "vanilla"  # PPO类型
    clip_ratio: float = 0.2  # PPO clip参数
```

### 1.3 蒸馏损失计算流程

**文件位置**: `verl/trainer/distillation/losses.py: 161-218`

```python
def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    student_logits: torch.Tensor = None,
):
    """
    蒸馏+PPO联合损失
    
    工作流程:
    1. [序列并行分片] split sequence across sp/cp groups
    2. [模型前向] model forward → logits: (bsz, seqlen/cp_size, vocab_size/tp_size)
    3. [logits处理器计算topk损失] → (bsz, seqlen/cp_size)
    4. [all_gather跨sp/cp组] → (bsz, seqlen)
    5. [组合topk损失与policy损失]
    
    输入:
    - model_output: 模型输出，包含log_probs, entropy
    - data: 微批次输入
      - teacher_logprobs: (bsz, seqlen, topk) - 教师top-k log概率
      - teacher_ids: (bsz, seqlen, topk) - 教师top-k token IDs
    - student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size)
    
    输出:
    - 当student_logits非空: topk损失tensor (bsz, seqlen/cp_size)
    - 当student_logits为空: 最终策略损失标量和metrics
    """
    
    # 在logits处理器中调用 (用于top-k损失计算)
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits)
    
    # 作为最终策略损失调用
    distill_loss, distill_metrics = distillation_loss(
        config, distillation_config, model_output, data
    )
    policy_loss, policy_metrics = ppo_loss(config, model_output, data)
    
    # 如果不使用任务奖励，只用蒸馏损失
    if not distillation_loss_config.use_task_rewards:
        policy_loss = 0.0
    
    # 组合损失
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = distillation_loss_config.distillation_loss_coef
    policy_loss += distill_loss * distillation_loss_coef
    
    return policy_loss, policy_metrics
```

### 1.4 Forward KL Top-K损失

**文件位置**: `verl/trainer/distillation/fsdp/losses.py: 35-77`

```python
def compute_forward_kl_topk(
    student_logits: torch.Tensor,      # (bsz, seqlen/sp_size, vocab_size)
    teacher_topk_log_probs: torch.Tensor, # (bsz, seqlen, topk)
    teacher_topk_ids: torch.Tensor,    # (bsz, seqlen, topk)
    config: DistillationConfig,
):
    """
    使用教师top-k信息计算Forward KL蒸馏损失
    
    数学公式:
    KL(P_teacher || Q_student) = Σ_k P_k * log(P_k / Q_k)
    
    维度追踪:
    - student_logits: (bsz, seqlen/sp, vocab) → log_softmax → (bsz, seqlen/sp, vocab)
    - gather with teacher_topk_ids → student_topk_log_probs: (bsz, seqlen/sp, topk)
    - kl_divergence → losses: (bsz, seqlen/sp)
    """
    
    # 1. 处理序列并行分片
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    
    # 2. 学生log概率
    student_log_probs = F.log_softmax(student_logits, dim=-1)  # (bsz, seqlen, vocab)
    
    # 3. 提取学生在教师top-k位置的log概率
    student_topk_log_probs = torch.gather(
        student_log_probs, dim=-1, index=teacher_topk_ids
    )  # (bsz, seqlen, topk)
    
    # 4. 计算概率质量 (监控用)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    
    # 5. 可选: clamp log概率以增加稳定性
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(log_prob_min_clamp)
    
    # 6. 计算KL散度
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

### 1.5 使用蒸馏损失作为Policy Gradient

**文件位置**: `verl/trainer/distillation/losses.py: 253-279`

```python
def distillation_loss(...):
    """
    蒸馏损失的两种使用方式:
    
    方式1: Policy Gradient (use_policy_gradient=True)
    - 将负蒸馏损失作为优势
    - 使用PPO更新
    - 参考: https://thinkingmachines.ai/blog/on-policy-distillation/
    - 推荐loss_mode: "k1"
    
    方式2: 监督学习 (use_policy_gradient=False)  
    - 直接反向传播蒸馏损失
    - 参考: https://arxiv.org/abs/2306.13649
    - 推荐loss_mode: "k3" 或 "forward_kl_topk"
    """
    
    if loss_config.use_policy_gradient:
        # 将负蒸馏损失作为reward/advantage
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),  # 关键: 负损失作为优势
            response_mask=response_mask,
            ...
        )
    else:
        # 直接监督学习方式
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            ...
        )
    
    return distillation_loss, metrics
```

---

## 2. 其他Advantage Estimator变体

### 2.1 RLOO (Reinforcement Learning from Leave-One-Out)

**文件位置**: `verl/trainer/ppo/core_algos.py: 587-636`

**数学公式**:
对于K个response，RLOO基线排除当前样本:
$$A_i = r_i \cdot \frac{K}{K-1} - \mu_{-i} \cdot \frac{K}{K-1}$$

其中 $\mu_{-i}$ 是排除第i个样本后的均值。

**代码实现**:
```python
@register_adv_est(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    response_mask: torch.Tensor,        # (bs, response_length)
    index: np.ndarray,                  # (bs,) - prompt分组索引
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RLOO优势估计 (https://arxiv.org/abs/2402.14740)
    
    核心思想: Leave-One-Out基线，避免样本自身影响基线
    """
    scores = token_level_rewards.sum(dim=-1)  # (bsz,)
    
    id2score = defaultdict(list)
    id2mean = {}
    
    with torch.no_grad():
        bsz = scores.shape[0]
        
        # 收集每组分数
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # 计算组均值
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            else:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
        
        # RLOO优势计算
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                # A_i = r_i * K/(K-1) - μ * K/(K-1)
                # 等价于: A_i = (r_i - μ_{-i}) where μ_{-i} = (Σr - r_i)/(K-1)
                scores[i] = (
                    scores[i] * response_num / (response_num - 1) - 
                    id2mean[index[i]] * response_num / (response_num - 1)
                )
        
        scores = scores.unsqueeze(-1) * response_mask
    
    return scores, scores
```

**向量化版本** (`core_algos.py: 831-866`):
```python
@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)
def compute_rloo_vectorized_outcome_advantage(...):
    scores = token_level_rewards.sum(dim=-1)
    
    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)
        
        # 向量化组计数和求和
        c = torch.bincount(inv)[inv].to(scores.dtype)  # 每个样本所在组的大小
        
        # RLOO公式: (c*r_i - sum_group) / (c-1)
        # = (c*r_i - sum_group) / (c-1) where sum_group = bincount(inv, weights=scores)[inv]
        adv = (
            (c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)
        ) * (c > 1)
        
        adv = adv.unsqueeze(-1) * response_mask
    
    return adv, adv
```

### 2.2 REINFORCE++

**文件位置**: `verl/trainer/ppo/core_algos.py: 693-729`

**数学公式**:
使用折扣return和whitening:
$$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$
$$A_t = \text{whiten}(G_t)$$

**代码实现**:
```python
@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    response_mask: torch.Tensor,        # (bs, response_length)
    config: Optional[AlgoConfig] = None,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    REINFORCE++ (https://arxiv.org/abs/2501.03262)
    
    特点: 不需要组分组，使用时序折扣return
    """
    assert config is not None
    gamma = config.gamma
    
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0
        
        # 从后向前计算折扣return
        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # EOS后重置
            running_return = running_return * response_mask[:, t]
        
        # Whitening标准化
        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask
    
    return advantages, returns
```

### 2.3 ReMax

**文件位置**: `verl/trainer/ppo/core_algos.py: 732-765`

**数学公式**:
使用greedy baseline:
$$A_i = G_t - b_{greedy}$$

**代码实现**:
```python
@register_adv_est(AdvantageEstimator.REMAX)
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    reward_baselines: torch.Tensor,     # (bs,) - greedy解码的奖励
    response_mask: torch.Tensor,        # (bs, response_length)
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ReMax (https://arxiv.org/abs/2310.10505)
    
    使用greedy解码的奖励作为baseline
    """
    with torch.no_grad():
        # 累积return (从后向前)
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        
        # 减去baseline
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask
    
    return advantages, returns
```

### 2.4 GDPO (Group Decoupled Policy Optimization)

**文件位置**: `verl/trainer/ppo/core_algos.py: 361-468`

**数学公式**:
对每个奖励维度独立归一化后加权聚合:
$$A_{final} = \text{whiten}\left(\sum_k w_k \cdot \text{GRPO}(r_k)\right)$$

**代码实现**:
```python
@register_adv_est(AdvantageEstimator.GDPO)
def compute_gdpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    non_tensor_batch: Optional[dict] = None,
    batch: Optional[dict] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GDPO (https://arxiv.org/abs/2601.05242)
    
    核心思想: 防止主导奖励信号压倒其他奖励
    
    步骤:
    1. 组内解耦归一化: 对每个奖励维度k独立GRPO
    2. 加权聚合: A_sum = Σ w_k * A_k
    3. 批次级归一化: A_final = whiten(A_sum)
    """
    
    # 获取各维度奖励
    gdpo_reward_keys = config.get("gdpo_reward_keys", None)  # e.g., ['format_reward', 'accuracy_reward']
    score_list = []
    
    for key in gdpo_reward_keys:
        comp = non_tensor_batch[key]
        rm_score = torch.tensor(np.asarray(comp, dtype=np.float32), device=device)
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        rm_scores[torch.arange(rm_scores.size(0)), valid_response_length] = rm_score
        score_list.append(rm_scores)
    
    # 权重
    gdpo_weights = config.get("gdpo_reward_weights", None)
    weights = torch.tensor(reward_weights, ...)
    
    # 对每个维度独立GRPO
    new_advantage = None
    for i in range(num_scores):
        normalized_score, _ = compute_grpo_outcome_advantage(
            token_level_rewards=score_list[i],
            response_mask=response_mask,
            index=index,
            ...
        )
        
        if new_advantage is None:
            new_advantage = weights[i] * normalized_score
        else:
            new_advantage += weights[i] * normalized_score
    
    # 最终whitening
    advantages = verl_F.masked_whiten(new_advantage, response_mask) * response_mask
    
    return advantages, advantages
```

### 2.5 OPO

**文件位置**: `verl/trainer/ppo/core_algos.py: 639-690`

**数学公式**:
使用长度加权基线:
$$b = \frac{\sum_i |y_i| \cdot r_i}{\sum_i |y_i|}$$

**代码实现**:
```python
@register_adv_est(AdvantageEstimator.OPO)
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    ...
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    OPO (https://arxiv.org/pdf/2505.23585)
    
    使用长度加权baseline，考虑response长度的影响
    """
    response_length = response_mask.sum(dim=-1)  # 每个response的长度
    scores = token_level_rewards.sum(dim=-1)
    
    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}  # 长度加权baseline
    
    with torch.no_grad():
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])
        
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            else:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                # 长度加权均值: Σ(len * score) / Σ(len)
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
        
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        
        scores = scores.unsqueeze(-1) * response_mask
    
    return scores, scores
```

### 2.6 Optimal Token Baseline

**文件位置**: `verl/trainer/ppo/core_algos.py: 869-985`

**数学公式**:
为每个时间步计算最优baseline:
$$B_t^* = \frac{\mathbb{E}[G_t \cdot W_t]}{\mathbb{E}[W_t]}$$

其中 $W_t = \sum_{j=1}^t \|s_j\|^2$ 是累积路径方差代理。

**代码实现**:
```python
@register_adv_est(AdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,    # (bs, response_length) - 训练策略log概率
    sum_pi_squared: torch.Tensor,   # (bs, response_length) - Σπ²
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimal Token Baseline (OTB)
    
    为每个时间步计算独立的baseline，而不是整个轨迹使用一个baseline
    
    理论:
    - W_t = Σ_{j=1}^t ||s_j||² (累积路径方差代理)
    - ||s_j||² = 1 - 2π_j + Σπ²
    - B_t* = E[G_t × W_t] / E[W_t]
    """
    with torch.no_grad():
        # 计算每个时间步的return (reward-to-go)
        returns = (token_level_rewards * response_mask).flip([-1]).cumsum(-1).flip([-1])
        
        # Step 1: 计算w_per_timestep = 1 - 2π_t + Σπ²
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared
        
        # Step 2: 可选的重要性采样修正
        if rollout_is_weights is not None:
            w_per_timestep = w_per_timestep * (rollout_is_weights ** 2)
        
        # Step 3: 累积路径方差: W_t = Σ_{j=1}^t w_j
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)
        
        # 按prompt分组
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)
        
        baselines = torch.zeros_like(returns)
        
        # 对每个prompt组计算per-step baseline
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                continue  # 单轨迹无baseline
            
            traj_idx = torch.tensor(trajectory_indices, device=device)
            
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]
            
            # B_t = Σ[G_t × W_t] / Σ[W_t]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon
            
            baseline_per_step = numerator / denominator
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)
        
        # A_t = G_t - B_t
        advantages = (returns - baselines) * response_mask
    
    return advantages, returns
```

---

## 3. Policy Loss变体

### 3.1 DPPO (Divergence-Constrained PPO)

**文件位置**: `verl/trainer/ppo/core_algos.py: 1372-1535`

提供两种变体: Binary-TV 和 Binary-KL

**DPPO-Binary-TV** (`core_algos.py: 1372-1450`):
```python
@register_policy_loss("dppo_tv")
def compute_policy_loss_dppo_tv(...):
    """
    DPPO-Binary-TV (https://arxiv.org/pdf/2602.04879)
    
    使用Total Variation距离约束而非ratio clipping
    """
    # 计算概率
    prob = torch.exp(log_prob)
    old_prob = torch.exp(old_log_prob)
    
    # TV距离约束
    valid_positive_mask = (prob - old_prob) <= clip_divergence_high
    valid_negative_mask = (prob - old_prob) >= -clip_divergence_low
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    
    # 截断重要性采样 (避免性能下降)
    truncated_ratio = torch.clamp(ratio, max=clip_ratio_c)  # 默认20.0
    truncated_ratio = truncated_ratio.detach()
    
    # 损失: 只在valid区域更新
    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask
    
    return pg_loss, pg_metrics
```

**DPPO-Binary-KL** (`core_algos.py: 1453-1535`):
```python
@register_policy_loss("dppo_kl")
def compute_policy_loss_dppo_kl(...):
    """
    DPPO-Binary-KL
    
    使用Binary KL距离约束
    """
    # Binary KL: p*log(p/q) + (1-p)*log((1-p)/(1-q))
    binary_kl = old_prob * (old_log_prob - log_prob) + (1 - old_prob) * torch.log(
        (1.0 - old_prob + 1e-8) / (1.0 - prob + 1e-8)
    )
    
    valid_positive_mask = (binary_kl <= clip_divergence_high) | (prob <= old_prob)
    valid_negative_mask = (binary_kl <= clip_divergence_low) | (prob >= old_prob)
    valid_mask = torch.where(advantages > 0, valid_positive_mask, valid_negative_mask)
    
    pg_losses = -advantages * truncated_ratio * log_prob * valid_mask
    
    return pg_loss, pg_metrics
```

### 3.2 GSPO (Geometric Sequence PPO)

**文件位置**: `verl/trainer/ppo/core_algos.py: 1538-1611`

**数学公式**:
使用序列级几何平均重要性比率:
$$s_i(\theta) = \left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{1/|y_i|}$$

**代码实现**:
```python
@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",  # GSPO推荐使用
    config: Optional[ActorConfig] = None,
    ...
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    GSPO (https://arxiv.org/pdf/2507.18071)
    
    核心思想: 使用序列级几何平均重要性比率
    """
    negative_approx_kl = log_prob - old_log_prob
    
    # 序列级重要性比率: s(θ) = exp[(1/|y|) * Σ log(π_θ/π_old)]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
    
    # Token级组合比率: s_{i,t}(θ) = sg[s_i(θ)] · π_θ(y_t) / sg[π_θ(y_t)]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
    
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)
    
    # 标准PPO clipping
    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    
    # GSPO使用seq-mean-token-mean聚合
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, 
                       loss_agg_mode="seq-mean-token-mean", ...)
    
    return pg_loss, pg_metrics
```

### 3.3 SAPO (Smoothed Advantage Policy Optimization)

**文件位置**: `verl/trainer/ppo/core_algos.py: 1614-1696`

**数学公式**:
使用平滑门控函数:
$$f(r, \tau) = \sigma(\tau(r-1)) \cdot \frac{4}{\tau}$$

**代码实现**:
```python
@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ...
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    SAPO (https://arxiv.org/pdf/2511.20347)
    
    使用可微门控函数替代hard clipping
    """
    # 温度参数: 正/负优势使用不同温度
    tau_pos = config.tau_pos
    tau_neg = config.tau_neg
    
    def gate_function(x, tau):
        """SAPO门控函数"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)
    
    # 重要性比率
    ratio = torch.exp(log_prob - old_log_prob)
    
    # 根据优势符号选择温度
    taus = torch.where(advantages > 0, tau_pos, tau_neg)
    
    # 计算门控权重
    gates = gate_function(ratio, taus)
    
    # 策略损失
    pg_losses = -gates * advantages
    
    # SAPO使用seq-mean-token-mean聚合
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask,
                       loss_agg_mode="seq-mean-token-mean", ...)
    
    return pg_loss, pg_metrics
```

### 3.4 GPG (Group Policy Gradient)

**文件位置**: `verl/trainer/ppo/core_algos.py: 1699-1732`

**代码实现**:
```python
@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ...
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    GPG (Group Policy Gradient)
    
    最简单的REINFORCE风格更新，不使用重要性采样
    """
    # 直接使用log概率 × 优势
    pg_losses = -log_prob * advantages
    
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, ...)
    
    return pg_loss, {}
```

### 3.5 GMPO (Geometric Mean Policy Optimization)

**文件位置**: `verl/trainer/ppo/core_algos.py: 1920-2003`

**数学公式**:
使用几何平均重要性比率:
$$r_{geo} = \exp\left(\frac{1}{|y|}\sum_t \text{clip}(\log r_t, -\epsilon, +\epsilon)\right)$$

**代码实现**:
```python
@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ...
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    GMPO (https://arxiv.org/abs/2507.20673)
    
    使用几何平均重要性比率，提供更稳定的序列级更新
    """
    negative_approx_kl = log_prob - old_log_prob
    
    # Token级clipping (比标准PPO更宽)
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(
        sgn_advantage * negative_approx_kl, 
        sgn_advantage * negative_approx_kl_clamp
    )
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min
    
    # 几何平均比率: exp(mean(clipped_log_ratio))
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp(
        (negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    )
    
    # 序列级优势
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    
    pg_losses = -advantage * ratio
    pg_loss = torch.mean(pg_losses)
    
    return pg_loss, pg_metrics
```

### 3.6 CISPO (Clipped Importance Sampling PO)

**文件位置**: `verl/trainer/ppo/core_algos.py: 2006-2064`

**数学公式**:
使用stop gradient的clipped ratio:
$$J = \text{sg}[\text{clip}(r)] \cdot A \cdot \log\pi_\theta$$

**代码实现**:
```python
@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ...
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    CISPO (https://arxiv.org/pdf/2506.13585)
    
    关键: 对clipped ratio使用stop gradient
    梯度只通过log_prob流动，不通过ratio
    """
    ratio = torch.exp(log_prob - old_log_prob)
    
    # Clip重要性采样权重
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()  # 关键: stop gradient
    
    # CISPO目标: J = sg(clip(ratio)) * A * log π_θ
    # 损失: L = -J = -sg(clip(ratio)) * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob
    
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, ...)
    
    return pg_loss, pg_metrics
```

---

## 附录: 算法比较总结

### Advantage Estimator比较

| 算法 | 需要Critic | 奖励类型 | 基线 | 特点 |
|------|-----------|---------|------|------|
| GAE | ✓ | Token | V(s) | 经典方法,可控方差 |
| GRPO | ✗ | Outcome | 组均值 | 简单,适合LLM |
| RLOO | ✗ | Outcome | LOO均值 | 无偏基线 |
| REINFORCE++ | ✗ | Token | 无/whiten | 折扣return |
| ReMax | ✗ | Outcome | Greedy | 需额外greedy rollout |
| GDPO | ✗ | Outcome | 多维GRPO | 多维奖励解耦 |
| OPO | ✗ | Outcome | 长度加权 | 考虑长度影响 |
| OTB | ✗ | Outcome | 时步最优 | Token级baseline |

### Policy Loss比较

| 算法 | 约束方式 | IS处理 | 特点 |
|------|---------|--------|------|
| PPO (vanilla) | Ratio clipping | Ratio | 标准方法 |
| DPPO-TV | TV distance | Truncated | 理论更优 |
| DPPO-KL | Binary KL | Truncated | 信息论动机 |
| GSPO | Seq-level geo ratio | Clipped | 序列级更新 |
| SAPO | Smooth gate | Gated | 可微clipping |
| GPG | 无 | 无IS | 最简单 |
| GMPO | Geo mean ratio | Clipped | 稳定序列更新 |
| CISPO | Clip + SG | SG ratio | 梯度简化 |
