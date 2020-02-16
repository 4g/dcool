import gym
import pybullet_envs.bullet.racecarGymEnv as e
env = e.RacecarGymEnv()

observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
