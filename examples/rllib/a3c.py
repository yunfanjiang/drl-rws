import ray
from tqdm import tqdm
import ray.rllib.agents.a3c as a3c
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

import rps
import rps.models


def create_env(env_config):
    return rps.RPSEnv(**env_config)


if __name__ == "__main__":
    register_env("rws", create_env)

    ray.init()
    config = a3c.DEFAULT_CONFIG.copy()
    config["framework"] = "torch"
    config["num_workers"] = 1
    config["num_gpus"] = 0
    config["model"] = {
        "custom_model": "baseline_model",
        "custom_model_config": {},
    }
    config["env_config"] = {
        "scenario_name": "running_with_scissors_in_the_matrix_1",
        "state_obs": True,
        "centralized_critic": False,
        "seed": 1,
    }
    trainer = a3c.A3CTrainer(config=config, env="rws")

    for i in tqdm(range(1000)):
        result = trainer.train()
        print(pretty_print(result))
