from stable_baselines3.ppo import PPO, MultiInputPolicy

from rps import RPSEnv
from rps.models import StateObsExtractor


if __name__ == "__main__":
    env = RPSEnv()

    model = PPO(
        policy=MultiInputPolicy,
        env=env,
        verbose=1,
        policy_kwargs=dict(
            features_extractor_class=StateObsExtractor,
            features_extractor_kwargs=dict(embed_dim=8, feature_dim=256,),
        ),
    )
    model.learn(total_timesteps=int(1e6))
