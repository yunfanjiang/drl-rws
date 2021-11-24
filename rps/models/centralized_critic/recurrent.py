from typing import Dict, List

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.typing import TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension

from ..utils import make_mlp


class CCRNNModel(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        lstm_state_size=256,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.lstm_state_size = lstm_state_size

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
        pre_action_emb_shape = model_config.get("pre_action_emb_dim", 4)
        pi_config = model_config.get(
            "pi_config",
            {
                "in_size": self.lstm_state_size,
                "hidden_size": 128,
                "out_size": action_space.n,
                "n_layers": 2,
                "activation": "ReLU",
            },
        )
        vf_config = model_config.get(
            "vf_config",
            {
                # the centralized value function receives extracted features for both the focal and bot agents
                "in_size": 2 * feature_dim,
                "hidden_size": 128,
                "out_size": 1,
                "n_layers": 2,
                "activation": "ReLU",
            },
        )

        self.lstm = nn.LSTM(feature_dim, self.lstm_state_size, batch_first=True)

        # create layers
        # feature extractor
        self._layer_emb = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embed_dim,
        )
        self._ori_emb = nn.Embedding(
            num_embeddings=orientation_n_emb, embedding_dim=orientation_emb_dim,
        )
        self._pre_action_emb = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=pre_action_emb_shape,
        )
        self._feature_hidden = nn.Linear(
            np.prod(layer_obs_shape) * embed_dim
            + inventory_shape
            + shoot_ready_shape
            + orientation_emb_dim
            + position_shape
            + interaction_inventories_shape
            + pre_action_emb_shape,
            feature_dim,
        )
        # policy MLP
        self._pi = make_mlp(**pi_config)
        # value function
        self._vf = make_mlp(**vf_config)

        self._features_focal = None
        self._features_bot = None

        self._f_names_to_add_time_d = [
            "LAYER",
            "INVENTORY",
            "READY_TO_SHOOT",
            "ORIENTATION",
            "POSITION",
            "INTERACTION_INVENTORIES",
            "GLOBAL_ACTIONS",
            "BOT_LAYER",
            "BOT_INVENTORY",
            "BOT_READY_TO_SHOOT",
            "BOT_ORIENTATION",
            "BOT_POSITION",
            "BOT_INTERACTION_INVENTORIES",
        ]

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            torch.zeros(
                (self.lstm_state_size,), device=self._feature_hidden.weight.device
            ),
            torch.zeros(
                (self.lstm_state_size,), device=self._feature_hidden.weight.device
            ),
        ]
        return h

    @override(ModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs"]
        obs_with_time_dimension = {}
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        for obs_name in self._f_names_to_add_time_d:
            each_obs = obs[obs_name]
            max_seq_len = each_obs.shape[0] // seq_lens.shape[0]
            obs_with_time_dimension[obs_name] = self._add_time_dimension(
                each_obs, max_seq_len
            )
        obs = obs_with_time_dimension

        # decentralized actions for the focal agent
        layer_emb = self._layer_emb(obs["LAYER"].long())
        layer_emb = layer_emb.view(layer_emb.shape[0], layer_emb.shape[1], -1)
        ori_emb = self._ori_emb(obs["ORIENTATION"].long())
        ori_emb = ori_emb.view(ori_emb.shape[0], ori_emb.shape[1], -1)
        pre_action_emb = self._pre_action_emb(obs["GLOBAL_ACTIONS"][..., 0].long())
        feature_cat = torch.cat(
            [
                layer_emb,
                obs["INVENTORY"],
                obs["READY_TO_SHOOT"],
                ori_emb,
                obs["POSITION"],
                obs["INTERACTION_INVENTORIES"],
                pre_action_emb,
            ],
            dim=-1,
        )
        self._features_focal = F.relu(self._feature_hidden(feature_cat))
        rnn_outputs, new_state = self.forward_rnn(self._features_focal, state, seq_lens)
        action_out = self._pi(rnn_outputs)
        output = torch.reshape(action_out, [-1, self.num_outputs])

        # extract features for the bot agent
        # decentralized actions for the focal agent
        layer_emb = self._layer_emb(obs["BOT_LAYER"].long())
        layer_emb = layer_emb.view(layer_emb.shape[0], layer_emb.shape[1], -1)
        ori_emb = self._ori_emb(obs["BOT_ORIENTATION"].long())
        ori_emb = ori_emb.view(ori_emb.shape[0], ori_emb.shape[1], -1)
        pre_action_emb = self._pre_action_emb(obs["GLOBAL_ACTIONS"][..., 1].long())
        feature_cat = torch.cat(
            [
                layer_emb,
                obs["BOT_INVENTORY"],
                obs["BOT_READY_TO_SHOOT"],
                ori_emb,
                obs["BOT_POSITION"],
                obs["BOT_INTERACTION_INVENTORIES"],
                pre_action_emb,
            ],
            dim=-1,
        )
        self._features_bot = F.relu(self._feature_hidden(feature_cat))
        return output, new_state

    def _add_time_dimension(self, x, max_seq_len):
        self.time_major = self.model_config.get("_time_major", False)
        output = add_time_dimension(
            x, max_seq_len=max_seq_len, framework="torch", time_major=self.time_major,
        )
        return output

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        x = inputs
        rnn_output, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        return rnn_output, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    @override(ModelV2)
    def value_function(self):
        assert (
            self._features_focal is not None and self._features_bot is not None
        ), "must call forward() first"
        centralized_features = torch.cat(
            [self._features_focal, self._features_bot], dim=-1
        )
        return torch.reshape(self._vf(centralized_features), [-1])
