from dc_env import DataCenterEnv
from dc_events import DataCenterLife

import time

def get_dc():
    dc = DataCenterEnv(800, 500)

    dc.add_wall(100, 150, 600)
    dc.add_wall(100, 350, 600)

    dc.add_fan(150, 200, rate=0)
    dc.add_fan(150, 300, rate=0)
    dc.add_fan(150, 250, rate=0)

    return dc


dc = get_dc()
dlife = DataCenterLife(dc)
dc.start()
dlife.start()

time_elapsed = 0
start_time = time.time()
for i in range(10000):
    state = dc.state()
    heat_ratio = state[0].get('heater', 0)
    sample_action = dc.action_space().sample()
    fan_speed = int(10*(heat_ratio/2000.0))
    # fan_speed = 10
    sample_action = [fan_speed for j in sample_action]
    dc.step(sample_action)
    time_elapsed = time.time() - start_time
    print(time_elapsed, "::", dc.state())
    time.sleep(1)

# dc.join()