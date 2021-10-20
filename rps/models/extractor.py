from typing import Dict

import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StateObsExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        embed_dim: int = 8,
        feature_dim: int = 256,
        state_shape: tuple = (5, 5, 11),
        inventory_shape: int = 3,
        shoot_ready_shape: int = 1,
    ):
        super().__init__(
            observation_space=observation_space, features_dim=feature_dim,
        )

        self._emb_layer = nn.Embedding(42, embed_dim)
        self._hidden_layer = nn.Linear(
            np.prod(state_shape) * embed_dim + inventory_shape + shoot_ready_shape,
            feature_dim,
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        state_emb = self._emb_layer(observations["LAYER"].long())
        state_emb = state_emb.view(state_emb.shape[0], -1)
        inventory = observations["INVENTORY"]
        shoot_ready = observations["READY_TO_SHOOT"]

        concat_features = torch.cat([state_emb, inventory, shoot_ready,], dim=-1,)
        return F.relu(self._hidden_layer(concat_features))
