from typing import Dict, List

import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from .utils import make_mlp


class BaselineModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        embed_dim = model_config.get("embed_dim", 8)
        num_embeddings = model_config.get("num_embeddings", 42)
        orientation_emb_dim = model_config.get("orientation_emb_dim", 4)
        orientation_n_emb = model_config.get("orientation_n_emb", 4)
        feature_dim = model_config.get("feature_dim", 256)
        layer_obs_shape = model_config.get("layer_obs_shape", (5, 5, 11))
        inventory_shape = model_config.get("inventory_shape", 3)
        shoot_ready_shape = model_config.get("shoot_ready_shape", 1)
        position_shape = model_config.get("position_shape", 2)
        interaction_inventories_shape = model_config.get(
            "interaction_inventories_shape", 6
        )
        pi_config = model_config.get(
            "pi_config",
            {
                "in_size": feature_dim,
                "hidden_size": 128,
                "out_size": action_space.n,
                "n_layers": 2,
                "activation": "ReLU",
            },
        )
        vf_config = model_config.get(
            "vf_config",
            {
                "in_size": feature_dim,
                "hidden_size": 128,
                "out_size": 1,
                "n_layers": 2,
                "activation": "ReLU",
            },
        )

        # create layers
        # feature extractor
        self._emb = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embed_dim,
        )
        self._ori_emb = nn.Embedding(
            num_embeddings=orientation_n_emb, embedding_dim=orientation_emb_dim,
        )
        self._feature_hidden = nn.Linear(
            np.prod(layer_obs_shape) * embed_dim
            + inventory_shape
            + shoot_ready_shape
            + orientation_emb_dim
            + position_shape
            + interaction_inventories_shape,
            feature_dim,
        )
        # policy MLP
        self._pi = make_mlp(**pi_config)
        # value function
        self._vf = make_mlp(**vf_config)

        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs"]
        layer_emb = self._emb(obs["LAYER"].long())
        layer_emb = layer_emb.view(layer_emb.shape[0], -1)
        ori_emb = self._ori_emb(obs["ORIENTATION"].long())
        ori_emb = ori_emb.view(ori_emb.shape[0], -1)
        feature_cat = torch.cat(
            [
                layer_emb,
                obs["INVENTORY"],
                obs["READY_TO_SHOOT"],
                ori_emb,
                obs["POSITION"],
                obs["INTERACTION_INVENTORIES"],
            ],
            dim=-1,
        )
        self._features = F.relu(self._feature_hidden(feature_cat))

        logits = self._pi(self._features)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._vf(self._features).squeeze(1)
