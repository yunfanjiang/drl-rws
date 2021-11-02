from ray.tune.registry import register_env

from .rps_env import RPSEnv, RPSEnvNoBot


def create_env(env_config):
    return RPSEnv(**env_config)


def create_env_no_bot(env_config):
    return RPSEnv(**env_config)


register_env("rws", create_env)
register_env("rws_no_bot", create_env_no_bot)
