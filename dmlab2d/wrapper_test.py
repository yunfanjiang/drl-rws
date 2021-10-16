from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import numpy as np
import pygame

import dmlab2d
from dmlab2d import runfiles_helper
from gym_wrapper import GymFromDMEnv

def create_environment(args):
  """Creates an environment.

  Args:
    args: See `main()` for description of args.

  Returns:
    dmlab2d.Environment with one observation.
  """
  args.settings['levelName'] = args.level_name
  lab2d = dmlab2d.Lab2d(runfiles_helper.find(), args.settings)
  print('HERE!!!!', lab2d.observation_names())
  return dmlab2d.Environment(lab2d, args.observation, args.env_seed)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--level_name', type=str, default='clean_up', help='Level name to load')
  parser.add_argument(
      '--observation',
      type=str,
      default='WORLD.RGB',
      help='Observation to render')
  parser.add_argument(
      '--settings', type=json.loads, default={}, help='Settings as JSON string')
  parser.add_argument(
      '--env_seed', type=int, default=0, help='Environment seed')
  parser.add_argument('--agent_seed', type=int, default=0, help='Agent seed')
  parser.add_argument(
      '--num_episodes', type=int, default=1, help='Number of episodes')
  parser.add_argument(
      '--scale', type=float, default=1, help='Scale to render screen')

  args = parser.parse_args()
  ## specify the observation name here
  args.observation = ['GLOBAL.TEXT', 'WORLD.RGB', '1.REWARD', '1.INVENTORY', '1.LAYER', '1.RGB']
  env = create_environment(args)

  print('dmlab2d action spec',env.action_spec())
  print('dmlab2d obs spec', env.observation_spec())
  gym_env = GymFromDMEnv(env,level_name=args.level_name,convert_action=True,observation_keys = args.observation)
  gym_env.reset()
  for i in range(10):
      action = gym_env.action_space.sample()
      print('action',action)
      obs, reward, done, info = gym_env.step(action)
      for key in obs.keys():
        print(key,obs[key].shape)
      print('reward',reward)




if __name__ == '__main__':
  main()

