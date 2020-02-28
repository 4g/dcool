import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, arg1, arg2):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(5)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(640, 480, 3), dtype=np.uint8)

  def step(self, action):
    observation = self.observation_space.sample()
    reward = 0
    done = 1
    info = {}
    return observation, reward, done, info

  def reset(self):
    return self.observation_space.sample()  # reward, done, info can't be included

  def render(self, mode='human'):
    print ("banana")

  def close (self):
    pass
