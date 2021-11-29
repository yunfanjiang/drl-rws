import argparse

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune import tune
from ray.rllib.examples.models.rnn_model import RNNModel, TorchRNNModel
from ray.rllib.models import ModelCatalog
import rps
import rps.models
from constants import POLICY_SCENARIO_MAP


def train(args):
    # parse bot policy
    scenario_name = POLICY_SCENARIO_MAP[args.bot_policy]

    # initialize ray and configuration
    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["framework"] = "torch"
    ModelCatalog.register_custom_model(
        "rnn", TorchRNNModel if config["framework"] == "torch" else RNNModel
    )
    # env related config
    config["env"] = "rws"
    config["env_config"] = {
        "scenario_name": scenario_name,
        "state_obs": True,
        "centralized_critic": False,
        "seed": args.seed,
    }

    # computation resource related config
    config["num_workers"] = args.num_workers
    config["num_envs_per_worker"] = args.num_envs_per_worker
    config["num_gpus"] = args.num_gpus

    # model related config
    # config["model"] = {
    #     "custom_model": "baseline_model",
    #     "custom_model_config": {},
    # }

    # model related config
    if args.use_rnn:
        config["model"] = {
            "custom_model": "baseline_model",
            "custom_model_config": {},
        }
    else:
        config["model"] = {
            "custom_model": "rnn",
            "max_seq_len": 20,
        }
    # hyper-parameters related config
    config["lr"] = args.lr
    config["gamma"] = args.gamma
    config["entropy_coeff"] = args.entropy_coeff

    # run tune
    tune.run("PPO", config=config, name=args.exp_name, local_dir=args.local_dir)


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

    parser.add_argument(
        "--use_rnn", action="store_true",
    )
    args = parser.parse_args()

    train(args)
