import gym
from dc_rack_env import DataCenterEnv
import time

env = DataCenterEnv(render=True)
observation = env.reset()


for i in range(1000):
  for _ in range(1000):
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    # print(observation)
    if done == 1:
      env.reset()
      break
    time.sleep(1)
  print (reward, observation, action)
