from typing import Dict

import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


class StateObsExtractor(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.Space, embed_dim: int = 8, feature_dim: int = 256,
    ):
        super().__init__(
            observation_space=observation_space, features_dim=feature_dim,
        )

        self._extractor = nn.Sequential(
            nn.Embedding(42, embed_dim),
            nn.Flatten(),
            nn.Linear(5 * 5 * 4 * embed_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        state_obs = observations["1.LAYER"].long()
        return self._extractor(state_obs)
