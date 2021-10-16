from typing import Optional

import gym
import dmlab2d
import numpy as np
from dmlab2d import runfiles_helper

from rps.rps_env.utils import spec2space


class RPSEnv(gym.Env):
    def __init__(
        self,
        state_obs: bool = True,
        centralized_critic: bool = False,
        seed: Optional[int] = None,
    ):
        # create dmlab2d rps_env
        settings = {
            "levelName": "running_with_scissors",
        }
        obs = ["1.LAYER"] if state_obs else ["1.RGB"]
        if centralized_critic:
            obs.append("WORLD.RGB")
        lab2d = dmlab2d.Lab2d(runfiles_helper.find(), settings,)
        self._env = dmlab2d.Environment(lab2d, obs, seed)

        self._last_observation = None

        # observation space and action space
        obs_spec = self._env.observation_spec()
        self._obs_space = spec2space(obs_spec)
        self._action_space = gym.spaces.Discrete(30)

        # compute the action mapping from cartesian product
        self._action_heads = [head for head in self._env.action_spec().keys()]
        self._action_map = {}
        move_actions = [0, 1, 2, 3, 4]
        turn_actions = [-1, 0, 1]
        fire_actions = [0, 1]
        idx = 0
        for move in move_actions:
            for turn in turn_actions:
                for fire in fire_actions:
                    self._action_map[idx] = (move, turn, fire)
                    idx += 1

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
        self._last_observation = timestep.observation
        return timestep.observation

    def step(self, action: np.ndarray):
        assert action < len(self._action_map)
        primitive_action = self._action_map[int(action)]
        converted_action = {
            head: primitive_action[i] for i, head in enumerate(self._action_heads)
        }
        timestep = self._env.step(converted_action)
        self._last_observation = timestep.observation
        reward = timestep.reward or 0.0
        return timestep.observation, reward, timestep.last(), {}
