from copy import deepcopy
from typing import Optional

import gym
import numpy as np
from meltingpot.python import scenario
from ml_collections import config_dict


ALL_SPACES = {
    "RGB": gym.spaces.Box(low=0, high=255, shape=(40, 40, 3), dtype=np.uint8,),
    "INVENTORY": gym.spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32,),
    "READY_TO_SHOOT": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32,),
    "LAYER": gym.spaces.Box(low=0, high=41, shape=(5, 5, 11), dtype=np.int32,),
    "WORLD.RGB": gym.spaces.Box(low=0, high=255, shape=(120, 184, 3), dtype=np.uint8,),
    "POSITION": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32,),
    "ORIENTATION": gym.spaces.Box(low=0, high=3, shape=(1,), dtype=np.int32),
    "INTERACTION_INVENTORIES": gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
    ),
}


class RPSEnv(gym.Env):
    def __init__(
        self,
        scenario_name="running_with_scissors_in_the_matrix_1",
        state_obs: bool = True,
        centralized_critic: bool = False,
        world_rgb: bool = False,
        seed: Optional[int] = None,
    ):
        # create rwc env
        scenario_config = scenario.get_config(scenario_name)
        self._env = scenario.build(config_dict.ConfigDict(scenario_config))
        self._env._rng = np.random.RandomState(seed=seed)

        self._last_observation = None

        # observation space and action space
        obs_keys = [
            "INVENTORY",
            "READY_TO_SHOOT",
            "POSITION",
            "ORIENTATION",
            "INTERACTION_INVENTORIES",
        ]
        if state_obs:
            obs_keys.append("LAYER")
        else:
            obs_keys.append("RGB")
        if world_rgb:
            obs_keys.append("WORLD.RGB")
        self._world_rgb = world_rgb
        self._state_obs = state_obs
        self._centralized_critic = centralized_critic

        obs_space = {k: ALL_SPACES[k] for k in obs_keys}
        if centralized_critic:
            bot_obs_space = deepcopy(obs_space)
            if world_rgb:
                bot_obs_space.pop("WORLD.RGB")
            for k, v in bot_obs_space.items():
                obs_space[f"BOT_{k}"] = v
            obs_space["GLOBAL_ACTIONS"] = gym.spaces.MultiDiscrete(
                [
                    self._env.action_spec()[0].num_values,
                    self._env.action_spec()[0].num_values,
                ]
            )
        self._obs_space = gym.spaces.Dict(obs_space)
        self._action_space = gym.spaces.Discrete(self._env.action_spec()[0].num_values)

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def seed(self, seed=None):
        self._env._rng = np.random.RandomState(seed=seed)

    def reset(self):
        focal_timestep, bot_timestep = self._env.reset()
        bot_timestep = bot_timestep[0]
        focal_obs, bot_obs = focal_timestep.observation[0], bot_timestep.observation
        obs = self._obs_postprocess(focal_obs, bot_obs)
        self._last_observation = obs
        return obs

    def step(self, action):
        if isinstance(action, int):
            action = [action]
        elif isinstance(action, np.int64) and action.ndim == 0:
            action = [action.item()]
        focal_timestep, bot_timestep = self._env.step(action)
        bot_timestep = bot_timestep[0]
        focal_obs, bot_obs = focal_timestep.observation[0], bot_timestep.observation
        obs = self._obs_postprocess(focal_obs, bot_obs)
        self._last_observation = obs
        reward = focal_timestep.reward[0].item()
        return obs, reward, focal_timestep.last(), {}

    def _obs_postprocess(self, focal_obs, bot_obs):
        focal_obs["INVENTORY"] = np.float32(focal_obs["INVENTORY"])
        focal_obs["READY_TO_SHOOT"] = np.float32(focal_obs["READY_TO_SHOOT"])
        focal_obs["READY_TO_SHOOT"] = np.expand_dims(
            focal_obs["READY_TO_SHOOT"], axis=0
        )
        focal_obs["POSITION"] = np.float32(focal_obs["POSITION"])
        focal_obs["ORIENTATION"] = np.int32(focal_obs["ORIENTATION"])
        focal_obs["ORIENTATION"] = np.expand_dims(focal_obs["ORIENTATION"], axis=0)
        focal_obs["INTERACTION_INVENTORIES"] = np.reshape(
            focal_obs["INTERACTION_INVENTORIES"], (6,)
        )
        focal_obs["INTERACTION_INVENTORIES"] = np.float32(
            focal_obs["INTERACTION_INVENTORIES"]
        )
        if self._state_obs:
            del focal_obs["RGB"]
        if not self._world_rgb:
            del focal_obs["WORLD.RGB"]

        obs = focal_obs
        if self._centralized_critic:
            obs["BOT_INVENTORY"] = np.float32(bot_obs["INVENTORY"])
            obs["BOT_READY_TO_SHOOT"] = np.float32(bot_obs["READY_TO_SHOOT"])
            obs["BOT_READY_TO_SHOOT"] = np.expand_dims(
                obs["BOT_READY_TO_SHOOT"], axis=0
            )
            obs["BOT_POSITION"] = np.float32(bot_obs["POSITION"])
            obs["BOT_ORIENTATION"] = np.int32(bot_obs["ORIENTATION"])
            obs["BOT_ORIENTATION"] = np.expand_dims(obs["BOT_ORIENTATION"], axis=0)
            obs["BOT_INTERACTION_INVENTORIES"] = np.reshape(
                bot_obs["INTERACTION_INVENTORIES"], (6,)
            )
            obs["BOT_INTERACTION_INVENTORIES"] = np.float32(
                obs["BOT_INTERACTION_INVENTORIES"]
            )
            if self._state_obs:
                obs["BOT_LAYER"] = bot_obs["LAYER"]
            else:
                obs["BOT_RGB"] = bot_obs["RGB"]
            obs["GLOBAL_ACTIONS"] = np.int64(bot_obs["global"]["actions"])
        return obs
