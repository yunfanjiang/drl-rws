from stable_baselines3.ppo import PPO, MultiInputPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from argparse import ArgumentParser
from constants import (Policy,
                       POLICY_SCENARIO_MAP)

from rps import RPSEnv
from rps.models import StateObsExtractor

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
      "--oppo-policy",
      type=Policy,
      choices=list(Policy),
      default="semi_pure",
      help="policy for opponent."
    )
    parser.add_argument(
      "--lr",
      type=float,
      default=4e-4,
      help="learning rate."
    )
    parser.add_argument(
      "--tb-log-dir",
      type=str,
      default=None,
      help="tensorboard log directory."
    )
    parser.add_argument(
      "--timesteps",
      type=float,
      default=1e6,
      help="total number of timesteps to train the policy."
    )
    args = parser.parse_args()

    env = RPSEnv(scenario_name=POLICY_SCENARIO_MAP[args.oppo_policy])
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env for _ in range(1)])

    model = PPO(
        policy=MultiInputPolicy,
        env=env,
        verbose=1,
        policy_kwargs=dict(
            features_extractor_class=StateObsExtractor,
            features_extractor_kwargs=dict(embed_dim=8, feature_dim=256,),
        ),
        learning_rate=args.lr,
        tensorboard_log=args.tb_log_dir,
    )
    model.learn(total_timesteps=args.timesteps)
