from ray.tune.registry import register_env

from .rps_env import RPSEnv, RPSEnvNoBot, RPSEnvLab2D


def create_env(env_config):
    return RPSEnv(**env_config)


def create_env_no_bot(env_config):
    return RPSEnv(**env_config)


def create_env_lab2d(env_config):
    return RPSEnvLab2D(**env_config)


register_env("rws", create_env)
register_env("rws_no_bot", create_env_no_bot)
register_env("rws_lab2d", create_env_lab2d)
