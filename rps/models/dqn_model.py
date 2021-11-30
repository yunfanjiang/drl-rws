"""PyTorch model for DQN"""

from typing import Sequence
from typing import Dict, List
import numpy as np
import torch.nn as nn
import gym
import torch.nn.functional as F
import torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.utils.annotations import override

torch, nn = try_import_torch()


class DQNTorchModel(TorchModelV2, nn.Module):
    """Extension of standard TorchModelV2 to provide dueling-Q functionality.
    """

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            *,
            q_hiddens: Sequence[int] = (256, ),
            dueling: bool = False,
            dueling_activation: str = "relu",
            num_atoms: int = 1,
            use_noisy: bool = False,
            v_min: float = -10.0,
            v_max: float = 10.0,
            sigma0: float = 0.5,
            # TODO(sven): Move `add_layer_norm` into ModelCatalog as
            #  generic option, then error if we use ParameterNoise as
            #  Exploration type and do not have any LayerNorm layers in
            #  the net.
            add_layer_norm: bool = False):
        """Initialize variables of this model.
        Extra model kwargs:
            q_hiddens (Sequence[int]): List of layer-sizes after(!) the
                Advantages(A)/Value(V)-split. Hence, each of the A- and V-
                branches will have this structure of Dense layers. To define
                the NN before this A/V-split, use - as always -
                config["model"]["fcnet_hiddens"].
            dueling (bool): Whether to build the advantage(A)/value(V) heads
                for DDQN. If True, Q-values are calculated as:
                Q = (A - mean[A]) + V. If False, raw NN output is interpreted
                as Q-values.
            dueling_activation (str): The activation to use for all dueling
                layers (A- and V-branch). One of "relu", "tanh", "linear".
            num_atoms (int): If >1, enables distributional DQN.
            use_noisy (bool): Use noisy layers.
            v_min (float): Min value support for distributional DQN.
            v_max (float): Max value support for distributional DQN.
            sigma0 (float): Initial value of noisy layers.
            add_layer_norm (bool): Enable layer norm (for param noise).
        """
        nn.Module.__init__(self)
        super(DQNTorchModel, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)

        self.dueling = dueling
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.sigma0 = sigma0
        ins = num_outputs

        advantage_module = nn.Sequential()
        value_module = nn.Sequential()

        ## add configuration for obs
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
        ## create layers
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
        self._features = None

        # Dueling case: Build the shared (advantages and value) fc-network.
        for i, n in enumerate(q_hiddens):
            if use_noisy:
                advantage_module.add_module(
                    "dueling_A_{}".format(i),
                    NoisyLayer(
                        ins,
                        n,
                        sigma0=self.sigma0,
                        activation=dueling_activation))
                value_module.add_module(
                    "dueling_V_{}".format(i),
                    NoisyLayer(
                        ins,
                        n,
                        sigma0=self.sigma0,
                        activation=dueling_activation))
            else:
                advantage_module.add_module(
                    "dueling_A_{}".format(i),
                    SlimFC(ins, n, activation_fn=dueling_activation))
                value_module.add_module(
                    "dueling_V_{}".format(i),
                    SlimFC(ins, n, activation_fn=dueling_activation))
                # Add LayerNorm after each Dense.
                if add_layer_norm:
                    advantage_module.add_module("LayerNorm_A_{}".format(i),
                                                nn.LayerNorm(n))
                    value_module.add_module("LayerNorm_V_{}".format(i),
                                            nn.LayerNorm(n))
            ins = n

        # Actual Advantages layer (nodes=num-actions).
        if use_noisy:
            advantage_module.add_module(
                "A",
                NoisyLayer(
                    ins,
                    self.action_space.n * self.num_atoms,
                    sigma0,
                    activation=None))
        elif q_hiddens:
            advantage_module.add_module(
                "A",
                SlimFC(
                    ins, action_space.n * self.num_atoms, activation_fn=None))

        self.advantage_module = advantage_module

        # Value layer (nodes=1).
        if self.dueling:
            if use_noisy:
                value_module.add_module(
                    "V",
                    NoisyLayer(ins, self.num_atoms, sigma0, activation=None))
            elif q_hiddens:
                value_module.add_module(
                    "V", SlimFC(ins, self.num_atoms, activation_fn=None))
            self.value_module = value_module

    def get_q_value_distributions(self, model_out):
        """Returns distributional values for Q(s, a) given a state embedding.
        Override this in your custom model to customize the Q output head.
        Args:
            model_out (Tensor): Embedding from the model layers.
        Returns:
            (action_scores, logits, dist) if num_atoms == 1, otherwise
            (action_scores, z, support_logits_per_action, logits, dist)
        """
        action_scores = self.advantage_module(model_out)

        if self.num_atoms > 1:
            # Distributional Q-learning uses a discrete support z
            # to represent the action value distribution
            z = torch.range(
                0.0, self.num_atoms - 1,
                dtype=torch.float32).to(action_scores.device)
            z = self.v_min + \
                z * (self.v_max - self.v_min) / float(self.num_atoms - 1)

            support_logits_per_action = torch.reshape(
                action_scores, shape=(-1, self.action_space.n, self.num_atoms))
            support_prob_per_action = nn.functional.softmax(
                support_logits_per_action, dim=-1)
            action_scores = torch.sum(z * support_prob_per_action, dim=-1)
            logits = support_logits_per_action
            probs = support_prob_per_action
            return action_scores, z, support_logits_per_action, logits, probs
        else:
            logits = torch.unsqueeze(torch.ones_like(action_scores), -1)
            return action_scores, logits, logits

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

        # logits = self._pi(self._features)
        return self._features, state

    def get_state_value(self, model_out):
        """Returns the state value prediction for the given state embedding."""

        return self.value_module(model_out)