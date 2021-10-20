from stable_baselines3.a2c import A2C, MultiInputPolicy

from rps import RPSEnv
from rps.models import StateObsExtractor


if __name__ == "__main__":
    env = RPSEnv()

    model = A2C(
        policy=MultiInputPolicy,
        env=env,
        verbose=1,
        policy_kwargs=dict(
            features_extractor_class=StateObsExtractor,
            features_extractor_kwargs=dict(embed_dim=8, feature_dim=256,),
        ),
        tensorboard_log="../../tblogs/a2c_state_obs"
    )
    model.learn(total_timesteps=int(1e6))
