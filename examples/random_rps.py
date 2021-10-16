import cv2

from rps import RPSEnv


if __name__ == "__main__":
    env = RPSEnv(centralized_critic=True)
    env.reset()
    for _ in range(int(1e6)):
        obs, reward, done, info = env.step(env.action_space.sample())
        obs = obs["WORLD.RGB"]
        cv2.imshow("Random RPS", obs)
        cv2.waitKey(1)
        if done:
            env.reset()
