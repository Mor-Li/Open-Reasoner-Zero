import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# 简化的KL散度计算
def compute_approx_kl(log_probs, log_probs_base, action_mask=None, use_k3=False, use_abs=False):
    log_ratio = log_probs - log_probs_base
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    
    print(f"Log ratio (对数比率): {log_ratio}")
    
    if use_k3:
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio
        print(f"使用K3估计器后的KL: {log_ratio}")
    
    if use_abs:
        log_ratio = log_ratio.abs()
        print(f"绝对值KL: {log_ratio}")
    
    return log_ratio

# 简化的奖励计算
def compute_reward(r, kl_coef, kl, custom_rewards=None, action_mask=None, use_kl_loss=False):
    print(f"原始奖励: {r}")
    print(f"KL系数: {kl_coef}")
    print(f"KL散度: {kl}")
    
    if action_mask is not None:
        if not use_kl_loss:
            kl_reward = -kl_coef * kl
            print(f"KL惩罚: {kl_reward}")
        else:
            kl_reward = torch.zeros_like(kl)
            
        if r is not None:
            # 找到每个序列的最后一个有效位置
            eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
            print(f"序列结束索引: {eos_indices}")
            
            # 在最后一个有效位置放置奖励
            last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))
            print(f"最后位置奖励: {last_reward}")
            
            reward = last_reward + kl_reward
        else:
            reward = kl_reward
            
        if custom_rewards is not None:
            custom_rewards_batch = pad_sequence(custom_rewards, batch_first=True, padding_value=0.0)
            print(f"自定义奖励: {custom_rewards_batch}")
            reward = reward + custom_rewards_batch
    else:
        # 无掩码的简化处理
        if not use_kl_loss:
            reward = -kl_coef * kl
        else:
            reward = torch.zeros_like(kl)
            
        if r is not None:
            # 假设r是标量奖励
            reward += r
    
    print(f"最终奖励: {reward}")
    return reward

# 简化的优势计算
def get_advantages_and_returns(values, rewards, action_mask=None, gamma=0.99, lambd=0.95):
    if action_mask is not None:
        if values is not None:
            values = action_mask * values
        rewards = action_mask * rewards
    
    print(f"价值估计: {values}")
    print(f"奖励: {rewards}")
    
    lastgaelam = 0
    advantages_reversed = []
    response_length = rewards.size(1)
    
    print("\n计算优势过程:")
    for t in reversed(range(response_length)):
        if values is not None:
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        else:
            delta = rewards[:, t]
        
        lastgaelam = delta + gamma * lambd * lastgaelam
        print(f"  时间步 {t}, delta: {delta}, 优势: {lastgaelam}")
        advantages_reversed.append(lastgaelam)
    
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    
    if values is not None:
        returns = advantages + values
    else:
        returns = advantages
        
    print(f"\n最终优势: {advantages}")
    print(f"最终回报: {returns}")
    
    return advantages.detach(), returns

# 示例使用
def main():
    # 创建一些假数据来模拟PPO计算
    batch_size = 2
    seq_length = 5
    
    # 创建日志概率，模拟策略网络输出
    log_probs = torch.randn(batch_size, seq_length)
    log_probs_base = torch.randn(batch_size, seq_length)  # 参考策略的日志概率
    
    # 创建动作掩码，标记有效的动作位置
    action_mask = torch.ones(batch_size, seq_length)
    action_mask[0, 3:] = 0  # 第一个序列在位置3之后结束
    action_mask[1, 4:] = 0  # 第二个序列在位置4之后结束
    
    # 创建值函数估计
    values = torch.randn(batch_size, seq_length)
    
    # 模拟基础奖励
    r = torch.tensor([1.0, 0.0])  # 第一个序列有正奖励，第二个有零奖励
    
    print("===== 计算KL散度 =====")
    kl = compute_approx_kl(log_probs, log_probs_base, action_mask)
    
    print("\n===== 计算奖励 =====")
    kl_coef = 0.1
    rewards = compute_reward(r, kl_coef, kl, action_mask=action_mask)
    
    print("\n===== 计算优势和回报 =====")
    advantages, returns = get_advantages_and_returns(values, rewards, action_mask)
    
    print("\n===== 关键结果汇总 =====")
    print(f"KL散度: {kl}")
    print(f"奖励: {rewards}")
    print(f"优势: {advantages}")
    print(f"回报: {returns}")

if __name__ == "__main__":
    main()