# rlhf_ppo/reward.py
import torch
import torch.nn.functional as F

def compute_ppo_rewards(actor_logits, ref_logits, targets, beta_kl=0.02):
    """
    Computes step-wise explicit rewards:
    R_t = Log_Likelihood(target_t) - beta * KL(pi_theta || pi_ref)
    """
    log_probs_actor = F.log_softmax(actor_logits, dim=-1)
    log_probs_ref = F.log_softmax(ref_logits, dim=-1)
    
    # 1. Base Intrinsic Reward: Log probability assigned to the ground-truth token
    target_log_probs = log_probs_actor.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    
    # 2. Token-level KL Divergence Penalty: KL(pi_theta || pi_ref)
    probs_actor = torch.exp(log_probs_actor)
    kl_penalty = (probs_actor * (log_probs_actor - log_probs_ref)).sum(dim=-1)
    
    # 3. Explicit Combined Step Reward
    rewards = target_log_probs - beta_kl * kl_penalty
    return rewards, kl_penalty.mean().item()