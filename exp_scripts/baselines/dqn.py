from stable_baselines3.dqn import DQN, MultiInputPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import argparse

from rps import RPSEnv
from rps.models import StateObsExtractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=str,
        default="semi-pure",
        choices=["semi_pure", "pure_rock", "pure_paper", "pure_scissor"],
        help="policy of the opponenet player",
    )
    args = parser.parse_args()
    if args.policy == "semi-pure":
        scenario_name = "running_with_scissors_in_the_matrix_1"
    elif args.policy == "pure_rock":
        scenario_name = "running_with_scissors_in_the_matrix_2"
    elif args.policy == "pure_paper":
        scenario_name = "running_with_scissors_in_the_matrix_3"
    elif args.policy == "pure_scissor":
        scenario_name = "running_with_scissors_in_the_matrix_4"
    else:
        raise NotImplementedError
    env = RPSEnv(scenario_name=scenario_name)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env for _ in range(1)])

    model = DQN(
        policy=MultiInputPolicy,
        env=vec_env,
        verbose=1,
        policy_kwargs=dict(
            features_extractor_class=StateObsExtractor,
            features_extractor_kwargs=dict(embed_dim=8, feature_dim=256,),
        ),
        tensorboard_log="/content/gdrive/MyDrive/baseline_results/dqn/dqn_pure_rock_state_obs",
        seed=666,
    )
    model.learn(total_timesteps=int(1e8))
