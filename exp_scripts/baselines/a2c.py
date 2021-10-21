from stable_baselines3.a2c import A2C, MultiInputPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from rps import RPSEnv
from rps.models import StateObsExtractor


if __name__ == "__main__":
    env = RPSEnv()
    vec_env = DummyVecEnv([lambda: env for _ in range(32)])

    model = A2C(
        policy=MultiInputPolicy,
        env=vec_env,
        verbose=1,
        policy_kwargs=dict(
            features_extractor_class=StateObsExtractor,
            features_extractor_kwargs=dict(embed_dim=8, feature_dim=256,),
        ),
        tensorboard_log="../../tblogs/a2c_state_obs",
        ent_coef=1e-3,
        seed=666,
    )
    model.learn(total_timesteps=int(1e6))
