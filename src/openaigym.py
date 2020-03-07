import gym
from dc_env import DataCenterEnv
import time
from dc_events import DataCenterLife

def get_dc():
  dc = DataCenterEnv(800, 500)

  dc.add_wall(100, 150, 600)
  dc.add_wall(100, 350, 600)
  dc.add_wall(100, 150, 200, vertical=True)

  dc.add_fan(150, 200, rate=0)
  dc.add_fan(150, 300, rate=0)
  dc.add_fan(150, 250, rate=0)

  dc.add_heater(450, 250, rate=0)
  dc.add_heater(250, 200, rate=0)

  return dc

env = get_dc()
dlife = DataCenterLife(env)
env.start()
dlife.start()

observation = env.reset()
for _ in range(1000):
  action = env.action_space.sample() # your agent here (this takes random actions)
  print (env.action_space)
  observation, reward, done, info = env.step(action)
  time.sleep(1)
  print (reward, observation.shape)
env.close()
