
# python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""bsuite adapter for OpenAI gym run-loops."""

from typing import Any, Dict, Optional, Tuple, Union

import dm_env
from dm_env import specs
import gym
from gym import spaces
import numpy as np
import dmlab2d
from collections import defaultdict
# OpenAI gym step format = obs, reward, is_finished, other_info
_GymTimestep = Tuple[np.ndarray, float, bool, Dict[str, Any]]


import gym
import numpy as np

## concatenated one hot action space for dmlab2d actions
class OneHotEncoding(gym.Space):
    """
    {0,...,1,...,0}

    Example usage:
    self.observation_space = OneHotEncoding(size=4)
    """
    def __init__(self, size=None):
        assert isinstance(size, tuple)
        for i in size:
            assert i > 0 and isinstance(i, int), "initialize one-hot space wrong:{}".format(size)
        self.size = size
        gym.Space.__init__(self, (sum(self.size),), dtype=np.int64)

    def sample(self):
        sample = []
        for item in self.size:
            one_hot_vector = np.zeros(item)
            one_hot_vector[np.random.randint(item)] = 1
            sample.append(one_hot_vector)
        return np.concatenate(sample)

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray)):
            sum = 0
            flag = True
            for item in self.size:
                number_of_zeros = list(x[sum:sum+item]).contains(0)
                number_of_ones = list(x[sum:sum+item]).contains(1)
                sum += item
                if not ((number_of_zeros == (item - 1)) and (number_of_ones == 1)):
                    flag = False
                    break
        return not flag


    def __repr__(self):
        return "OneHotEncoding(%d)" % self.size

    def __eq__(self, other):
        return self.size == other.size

## gym wrapper for dmlab2d where we return a concatenated one-hot action space and observation space is a dict.
class GymFromDMEnv(gym.Env):
    """A wrapper that converts a dm_env.Environment to an OpenAI gym.Env."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env: dmlab2d.Environment, level_name, observation_keys,convert_action=True):
        self.level = level_name
        self._env = env  # type: dm_lab.Environment
        self._last_observation = None  # type: Optional[np.ndarray]
        self.convert_action = convert_action
        self.action_values = {}
        self.observation_keys = observation_keys
        self.viewer = None
        self.game_over = False  # Needed for Dopamine agents.

    def step(self, action: int) -> _GymTimestep:
        if self.convert_action:
            assert np.sum(action) == len(self.action_values.keys()), "one hot vector is wrong:{}, {}".format(self.action_values.keys(),action)
            action_convert = {}
            index = 0
            for name in self.action_values.keys():
                values = self.action_values[name]
                min, max = values['min'], values['max']
                current_range= max - min + 1
                action_convert[name] = np.argmax(action[index:index +current_range]) + min
                index += current_range
            # print('original action',action)
            # print('action_convert',action_convert)
            timestep = self._env.step(action_convert)
        else:
            timestep = self._env.step(action)
        self._last_observation = timestep.observation
        reward = timestep.reward or 0.
        if timestep.last():
            self.game_over = True
        return timestep.observation, reward, timestep.last(), {}

    def reset(self) -> np.ndarray:
        self.game_over = False
        timestep = self._env.reset()
        self._last_observation = timestep.observation
        return timestep.observation

    def render(self, mode: str = 'rgb_array') -> Union[np.ndarray, bool]:
        if self._last_observation is None:
            raise ValueError('Environment not ready to render. Call reset() first.')

        if isinstance(self.observation_keys, list) \
                and 'WORLD.RGB' in self.observation_keys :
            image_observation = self._last_observation['WORLD.RGB']
        elif isinstance(self.observation_keys,list) and '1.RGB' in self.observation_keys:
            image_observation = self._last_observation['1.RGB']
        elif isinstance(self.observation_keys,str) and self._last_observation.shape[-1] == 3:
            image_observation = self._last_observation
        else:
            raise ValueError('Image observation is not included in dmlab2d obs:{}'.format(self.observation_keys))
        if mode == 'rgb_array':
            return image_observation
        if mode == 'human':
            if self.viewer is None:
                # pylint: disable=import-outside-toplevel
                # pylint: disable=g-import-not-at-top
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(image_observation)
            return self.viewer.isopen

    @property
    def action_space(self) :
        if self.convert_action:
            return self.convert_action_space()
        action_spec = self._env.action_spec()
        return spec2space(action_spec)

    def convert_action_space(self):
        for action,space in self._env.action_spec().items():
            if space.dtype in ('int32','int64'):
                self.action_values[action] = dict(min=space.minimum,max = space.maximum)
            else:
                raise ValueError('Continuous action space(should be discrete): {}, {}'.format(action,space))
        return OneHotEncoding(size=tuple([int(value['max']-value['min']+1) for value in self.action_values.values()]))

    @property
    def observation_space(self) -> spaces.Box:
        obs_spec = self._env.observation_spec()  # type: specs.Array
        if isinstance(obs_spec, dict):
            return spec2space(obs_spec)
        elif isinstance(obs_spec, specs.BoundedArray):
            return spaces.Box(
                low=float(obs_spec.minimum),
                high=float(obs_spec.maximum),
                shape=obs_spec.shape,
                dtype=obs_spec.dtype)

        return spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=obs_spec.shape,
            dtype=obs_spec.dtype)

    @property
    def reward_range(self) -> Tuple[float, float]:
        reward_spec = self._env.reward_spec()
        if isinstance(reward_spec, specs.BoundedArray):
            return reward_spec.minimum, reward_spec.maximum
        return -float('inf'), float('inf')

    def __getattr__(self, attr):
        """Delegate attribute access to underlying environment."""
        return getattr(self._env, attr)

def spec2space(spec:dm_env.specs, name: Optional[str]=None):
    if isinstance(spec,specs.DiscreteArray):
        return spaces.Discrete(n=spec.num_values)
    elif isinstance(spec, specs.BoundedArray):
        if spec.minimum == 0.0 and spec.minimum == 1.0 and spec.dtype in ('int32', 'int64'):
            return spaces.MultiBinary(n=spec.shape)
        elif spec.minimum==0.0 and spec.dtype in ('int32', 'int64'):
            return spaces.MultiDiscrete(nvec = spec.maximum,dtype = spec.dtype)
        else:
            return spaces.Box(low=float(spec.minimum),
                    high=float(spec.maximum),
                    shape=spec.shape,
                    dtype=spec.dtype)
    elif isinstance(spec, tuple):
        return spaces.Tuple(tuple(spec2space(s, name) for s in spec))
    elif isinstance(spec, dict):
        return spaces.Dict({key: spec2space(value, name) for key, value in spec.items()})
    else:
        raise ValueError('Unexpected dmlab2d space: {}'.format(spec))

def space2spec(space: gym.Space, name: Optional[str] = None):
    """Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.
    Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
    specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
    Dict spaces are recursively converted to tuples and dictionaries of specs.
    Args:
      space: The Gym space to convert.
      name: Optional name to apply to all return spec(s).
    Returns:
      A dm_env spec or nested structure of specs, corresponding to the input
      space.
    """
    if isinstance(space, spaces.Discrete):
        return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

    elif isinstance(space, spaces.Box):
        return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                                  minimum=space.low, maximum=space.high, name=name)

    elif isinstance(space, spaces.MultiBinary):
        return specs.BoundedArray(shape=space.shape, dtype=space.dtype, minimum=0.0,
                                  maximum=1.0, name=name)

    elif isinstance(space, spaces.MultiDiscrete):
        return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                                  minimum=np.zeros(space.shape),
                                  maximum=space.nvec, name=name)

    elif isinstance(space, spaces.Tuple):
        return tuple(space2spec(s, name) for s in space.spaces)

    elif isinstance(space, spaces.Dict):
        return {key: space2spec(value, name) for key, value in space.spaces.items()}

    else:
        raise ValueError('Unexpected gym space: {}'.format(space))



class DMEnvFromGym(dm_env.Environment):
    """A wrapper to convert an OpenAI Gym environment to a dm_env.Environment."""

    def __init__(self, gym_env: gym.Env):
        self.gym_env = gym_env
        # Convert gym action and observation spaces to dm_env specs.
        self._observation_spec = space2spec(self.gym_env.observation_space,
                                            name='observations')
        self._action_spec = space2spec(self.gym_env.action_space, name='actions')
        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        observation = self.gym_env.reset()
        return dm_env.restart(observation)

    def step(self, action: int) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()

        # Convert the gym step result to a dm_env TimeStep.
        observation, reward, done, info = self.gym_env.step(action)
        self._reset_next_step = done

        if done:
            is_truncated = info.get('TimeLimit.truncated', False)
            if is_truncated:
                return dm_env.truncation(reward, observation)
            else:
                return dm_env.termination(reward, observation)
        else:
            return dm_env.transition(reward, observation)

    def close(self):
        self.gym_env.close()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

