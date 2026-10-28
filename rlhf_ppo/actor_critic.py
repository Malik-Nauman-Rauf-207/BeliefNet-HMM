import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Point sys.path to the NanoGPT folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'NanoGPT')))

# Import your existing GPT architecture
from model import GPT, GPTConfig

class ActorCriticNanoGPT(nn.Module):
    def __init__(self, config: GPTConfig, detach_value: bool = True):
        super().__init__()
        # Backbone Transformer
        self.gpt = GPT(config)

        # Critic Head: Maps transformer hidden state sigma_t -> scalar value V(s_t)
        self.value_head = nn.Linear(config.n_embd, 1, bias=False)

        # Initialize value head weights with small standard deviation
        nn.init.normal_(self.value_head.weight, std=0.02)

        # If True, the critic reads a detached copy of the hidden state, so the value loss
        # (whose targets are large, e.g. ~ -100 with gamma=0.99) cannot distort the pretrained
        # backbone. The backbone is then trained by the policy gradient only.
        self.detach_value = detach_value

    def forward(self, idx, targets=None):
        # idx shape: (batch_size, sequence_length)
        b, t = idx.size()

        # Pass through NanoGPT backbone up to LayerNorm before lm_head
        tok_emb = self.gpt.transformer.wte(idx)
        pos_emb = self.gpt.transformer.wpe(torch.arange(0, t, dtype=torch.long, device=idx.device))
        x = self.gpt.transformer.drop(tok_emb + pos_emb)

        for block in self.gpt.transformer.h:
            x = block(x)
        x = self.gpt.transformer.ln_f(x)  # Shape: (b, t, n_embd) -> sigma_t

        # 1. Actor Logits: (b, t, vocab_size)
        logits = self.gpt.lm_head(x)

        # 2. Critic State Values: (b, t)
        values = self.value_head(x.detach() if self.detach_value else x).squeeze(-1)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

        return logits, values, loss

    @torch.no_grad()
    def generate_rollout(self, idx, max_new_tokens, temperature=1.0):
        """Samples rollouts from the policy pi_theta for PPO buffer collection."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.gpt.config.block_size else idx[:, -self.gpt.config.block_size:]
            logits, values, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx