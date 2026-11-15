"""Device-independent SFT, normalized DPO, reward modeling and PPO math."""

import torch
from torch.nn import functional as F


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def token_logps(logits, tokens):
    return F.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    margin = beta * ((chosen - ref_chosen) - (rejected - ref_rejected))
    return -F.logsigmoid(margin).mean()


def gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """EOS or generation cap is terminal; zero bootstrap after each response."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        next_value = values[:, t + 1] if t + 1 < values.shape[1] else 0.0
        next_mask = mask[:, t + 1] if t + 1 < mask.shape[1] else 0.0
        delta = rewards[:, t] + gamma * next_value * next_mask - values[:, t]
        carry = (delta + gamma * lam * next_mask * carry) * mask[:, t]
        advantages[:, t] = carry
    return advantages, (advantages + values) * mask


def ppo_loss(new_logps, old_logps, values, old_values, advantages, returns, mask, clip=0.2):
    ratio = (new_logps - old_logps).exp()
    policy = masked_mean(torch.maximum(-advantages * ratio, -advantages * ratio.clamp(1 - clip, 1 + clip)), mask)
    clipped_values = old_values + (values - old_values).clamp(-clip, clip)
    value = 0.5 * masked_mean(torch.maximum((values - returns).square(), (clipped_values - returns).square()), mask)
    return policy, value
