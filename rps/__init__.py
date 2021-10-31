from ray.tune.registry import register_env

from .rps_env import RPSEnv


def create_env(env_config):
    return RPSEnv(**env_config)


register_env("rws", create_env)
