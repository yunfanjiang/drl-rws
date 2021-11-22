import argparse

import ray
import ray.rllib.agents.a3c as a3c
from ray.tune import tune

import rps
import rps.models
from constants import POLICY_SCENARIO_MAP


def train(args):
    # parse bot policy
    scenario_name = POLICY_SCENARIO_MAP[args.bot_policy]

    # initialize ray and configuration
    ray.init()
    config = a3c.DEFAULT_CONFIG.copy()
    config["framework"] = "torch"

    # env related config
    config["env"] = "rws"
    config["env_config"] = {
        "scenario_name": scenario_name,
        "state_obs": True,
        "centralized_critic": True,
        "seed": args.seed,
    }

    # computation resource related config
    config["num_workers"] = args.num_workers
    config["num_envs_per_worker"] = args.num_envs_per_worker
    config["num_gpus"] = args.num_gpus

    # model related config
    config["model"] = {
        "custom_model": "centralized_critic_feed_forward_model",
        "custom_model_config": {},
    }

    # hyper-parameters related config
    config["lr"] = args.lr
    config["gamma"] = args.gamma
    config["entropy_coeff"] = args.entropy_coeff

    # run tune
    tune.run("A3C", config=config, name=args.exp_name, local_dir=args.local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("A2C RWS sanity check")

    # env related config
    parser.add_argument(
        "--bot_policy",
        type=str,
        choices=["pure", "semi_pure", "pure_rock", "pure_paper", "pure_scissor"],
        required=True,
    )

    # experiment name and save dir
    parser.add_argument(
        "--exp_name", type=str, required=True,
    )
    parser.add_argument(
        "--local_dir", type=str,
    )

    parser.add_argument(
        "--seed", type=int, default=0,
    )

    # computation resource related config
    parser.add_argument(
        "--num_workers", type=int, default=1,
    )
    parser.add_argument(
        "--num_envs_per_worker", type=int, default=1,
    )
    parser.add_argument(
        "--num_gpus", type=int, default=0, required=True,
    )

    # hyper-parameters
    parser.add_argument(
        "--lr", type=float, default=4e-4,
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
    )
    parser.add_argument(
        "--entropy_coeff", type=float, default=0.003,
    )

    args = parser.parse_args()

    train(args)
