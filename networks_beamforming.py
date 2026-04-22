"""
networks_beamforming.py

Networks for the M2LLM-Driven Centralized DDPG paper adaptation.

Architecture summary
────────────────────
  LLMEncoder        — Deep MLP with LayerNorm + GELU, simulating the
                      LLaVA (M2LLM) hidden-layer projection used in the
                      original paper.  Input: joint observation (880-dim).
                      Output: fixed-dim embedding (default 256).

  CentralDDPGActor  — Single centralized agent.  Passes joint obs through
                      LLMEncoder, then outputs a 12-dim joint action
                      (4 UAVs × [vx, vy, intensity]) via Tanh.

  CentralDDPGCritic — Q(joint_obs, joint_action).  Encodes obs with a
                      separate LLMEncoder instance, concatenates the
                      action embedding, and predicts a scalar Q-value.

Imports from uav_env_beamforming so that JOINT_SIZE / JOINT_ACTION_DIM
stay in sync if those constants ever change.
"""

import torch
import torch.nn as nn

from uav_env_beamforming import JOINT_SIZE, JOINT_ACTION_DIM

_EMBED_DIM = 256


class LLMEncoder(nn.Module):
    """
    Simulates the M2LLM (LLaVA) hidden-layer projection.
    LayerNorm + GELU mirrors the normalization style of transformer blocks
    in the original paper's vision-language model.
    Trained end-to-end with the DDPG agent — no pre-training required for
    the comparison baseline.
    """

    def __init__(self, input_dim: int = JOINT_SIZE, embed_dim: int = _EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CentralDDPGActor(nn.Module):
    """
    Centralized DDPG actor.
    joint_obs (880) → LLMEncoder → 256 → 128 → JOINT_ACTION_DIM (12) → Tanh
    Output range [-1, 1] for each of 4×[vx, vy, intensity].
    """

    def __init__(self, embed_dim: int = _EMBED_DIM):
        super().__init__()
        self.encoder = LLMEncoder(embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, JOINT_ACTION_DIM),
            nn.Tanh(),
        )

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(joint_obs))


class CentralDDPGCritic(nn.Module):
    """
    Centralized Q-critic.
    Encodes joint obs with its own LLMEncoder instance (separate weights
    from the actor's encoder, matching the original paper's independent
    model paths), concatenates the joint action, then predicts Q.

    joint_obs (880) → LLMEncoder → 256
    [concat with joint_action (12)] → 268 → 256 → 128 → 1
    """

    def __init__(self, embed_dim: int = _EMBED_DIM):
        super().__init__()
        self.obs_encoder = LLMEncoder(embed_dim=embed_dim)
        self.q_net = nn.Sequential(
            nn.Linear(embed_dim + JOINT_ACTION_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, joint_obs: torch.Tensor,
                joint_action: torch.Tensor) -> torch.Tensor:
        obs_emb = self.obs_encoder(joint_obs)
        return self.q_net(torch.cat([obs_emb, joint_action], dim=-1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    actor  = CentralDDPGActor()
    critic = CentralDDPGCritic()

    print(f"LLMEncoder params (actor) : {count_params(actor.encoder):,}")
    print(f"Actor total params        : {count_params(actor):,}")
    print(f"Critic total params       : {count_params(critic):,}")

    obs_batch    = torch.randn(8, JOINT_SIZE)
    action_batch = torch.randn(8, JOINT_ACTION_DIM)
    print(f"Actor output shape  : {actor(obs_batch).shape}")
    print(f"Critic output shape : {critic(obs_batch, action_batch).shape}")
