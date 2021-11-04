import argparse

import ray
from tqdm import tqdm
import ray.rllib.agents.impala as impala

import rps
import rps.models
from constants import POLICY_SCENARIO_MAP


def train(args):
    # parse bot policy
    scenario_name = POLICY_SCENARIO_MAP[args.bot_policy]

    # initialize ray and configuration
    ray.init()
    config = impala.DEFAULT_CONFIG.copy()
    config["framework"] = "torch"

    # env related config
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
    config["model"] = {
        "custom_model": "baseline_model",
        "custom_model_config": {},
    }

    # hyper-parameters related config
    config["lr"] = args.lr
    config["gamma"] = args.gamma
    config["entropy_coeff"] = args.entropy_coeff
    config["replay_proportion"] = args.replay_proportion

    # timeout time
    config["learner_queue_timeout"] = args.learner_queue_timeout

    # instantiate trainer
    trainer = impala.ImpalaTrainer(config=config, env="rws")

    # start training
    for _ in tqdm(range(int(args.train_steps))):
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Impala RWS sanity check")

    # env related config
    parser.add_argument(
        "--bot_policy",
        type=str,
        choices=["pure", "semi_pure", "pure_rock", "pure_paper", "pure_scissor"],
        required=True,
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
        "--num_gpus", type=int, default=1,
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
        "--replay_proportion", type=float, default=0.2,
    )

    # timeout time
    parser.add_argument(
        "--learner_queue_timeout", type=int, default=999999,
    )

    # train steps
    parser.add_argument(
        "--train_steps", type=float, default=1e9,
    )
    args = parser.parse_args()

    train(args)
