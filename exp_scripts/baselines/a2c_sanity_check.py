import argparse

from stable_baselines3.a2c import A2C, MultiInputPolicy

from rps import RPSEnv
from rps.models import StateObsExtractor


def train(args):
    env = RPSEnv(scenario_name="running_with_scissors_in_the_matrix_2",)

    model = A2C(
        policy=MultiInputPolicy,
        env=env,
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
    args = parser.parse_args()
    train(args)
