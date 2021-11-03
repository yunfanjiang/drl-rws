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
}


class RPSEnv(gym.Env):
    def __init__(
        self,
        scenario_name="running_with_scissors_in_the_matrix_1",
        state_obs: bool = True,
        centralized_critic: bool = False,
        seed: Optional[int] = None,
    ):
        # create rwc env
        scenario_config = scenario.get_config(scenario_name)
        self._env = scenario.build(config_dict.ConfigDict(scenario_config))
        self._env._rng = np.random.RandomState(seed=seed)

        self._last_observation = None

        # observation space and action space
        obs_keys = ["INVENTORY", "READY_TO_SHOOT"]
        if state_obs:
            obs_keys.append("LAYER")
        else:
            obs_keys.append("RGB")
        if centralized_critic:
            obs_keys.append("WORLD.RGB")
        self._state_obs = state_obs
        self._centralized_critic = centralized_critic

        obs_space = {k: ALL_SPACES[k] for k in obs_keys}
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
        timestep = self._env.reset()
        obs = timestep.observation[0]
        obs = self._obs_postprocess(obs)
        self._last_observation = obs
        return obs

    def step(self, action):
        if isinstance(action, int):
            action = [action]
        elif isinstance(action, np.int64) and action.ndim == 0:
            action = [action.item()]
        timestep = self._env.step(action)
        obs = timestep.observation[0]
        obs = self._obs_postprocess(obs)
        self._last_observation = obs
        reward = timestep.reward[0].item()
        return obs, reward, timestep.last(), {}

    def _obs_postprocess(self, obs):
        obs["INVENTORY"] = np.float32(obs["INVENTORY"])
        obs["READY_TO_SHOOT"] = np.float32(obs["READY_TO_SHOOT"])
        obs["READY_TO_SHOOT"] = np.expand_dims(obs["READY_TO_SHOOT"], axis=0)
        if self._state_obs:
            del obs["RGB"]
        if not self._centralized_critic:
            del obs["WORLD.RGB"]
        return obs
