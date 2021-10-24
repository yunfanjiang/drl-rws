import argparse

from stable_baselines3.a2c import A2C, MultiInputPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rps import RPSEnv
from rps.models import StateObsExtractor


def train(args):
    if args.bot_policy == "pure":
        scenario_name = "running_with_scissors_in_the_matrix_0"
    elif args.bot_policy == "semi_pure":
        scenario_name = "running_with_scissors_in_the_matrix_1"
    elif args.bot_policy == "rock":
        scenario_name = "running_with_scissors_in_the_matrix_2"
    elif args.bot_policy == "paper":
        scenario_name = "running_with_scissors_in_the_matrix_3"
    elif args.bot_policy == "scissors":
        scenario_name = "running_with_scissors_in_the_matrix_4"
    else:
        raise ValueError

    def env_fn():
        return Monitor(RPSEnv(scenario_name=scenario_name))

    vec_env = DummyVecEnv([env_fn for _ in range(args.n_envs)])

    model = A2C(
        policy=MultiInputPolicy,
        env=vec_env,
        verbose=1,
        policy_kwargs=dict(
            features_extractor_class=StateObsExtractor,
            features_extractor_kwargs=dict(embed_dim=8, feature_dim=256,),
        ),
        tensorboard_log=args.tb_log_dir,
        ent_coef=args.ent_coef,
        seed=args.seed,
        learning_rate=args.lr,
    )
    model.learn(total_timesteps=int(args.total_timesteps))

    if args.model_save_path:
        model.save(path=args.model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("A2C RWS sanity check")
    parser.add_argument(
        "--tb_log_dir", type=str, default=None,
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.003,
    )
    parser.add_argument(
        "--seed", type=int, default=0,
    )
    parser.add_argument(
        "--total_timesteps", type=float, default=1e6,
    )
    parser.add_argument(
        "--lr", type=float, default=4e-4,
    )
    parser.add_argument(
        "--bot_policy",
        type=str,
        choices=["pure", "semi_pure", "rock", "paper", "scissors"],
    )
    parser.add_argument(
        "--n_envs", type=int, default=1,
    )
    parser.add_argument(
        "--model_save_path", type=str,
    )
    args = parser.parse_args()
    train(args)
