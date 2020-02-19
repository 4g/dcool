from dc_env import DataCenterEnv
from dc_events import DataCenterLife
import matplotlib.pyplot as plt
import numpy as np
plt.ion() ## Note this correction
import time
import re


def get_dc():
    dc = DataCenterEnv(800, 500)

    dc.add_wall(100, 150, 600)
    dc.add_wall(100, 350, 600)

    dc.add_fan(150, 200, rate=0)
    dc.add_fan(150, 300, rate=0)
    dc.add_fan(150, 250, rate=0)

    return dc

def pad_left(l, size):
    lsize = len(l)
    if lsize >= size:
        return l

    z = list(range(size - lsize))
    l = z + l
    return l

def plot_metrics(metrics):
    maxsize = max([len(metrics[name]) for name in metrics])
    for n in metrics:
        metrics[n] = pad_left(metrics[n], maxsize)

    t = np.arange(0, maxsize, 1)

    legal_metrics = {"fan_speed_", "heater_speed_", "fan_energy"}
    allowed_names = set()
    key_groups = {}
    for name in metrics:
        key_group = re.sub('\d', '', name)
        if key_group in legal_metrics:
            allowed_names.add(name)
            key_groups[key_group] = key_groups.get(key_group, [])
            key_groups[key_group].append(name)

    plot_params = {}
    for group_index, key_group in enumerate(key_groups):
        for color_index, name in enumerate(key_groups[key_group]):
            plot_params[name] = [color_index, group_index]


    num_plots = len(legal_metrics)
    colors = ['r-', 'g-', 'b-']
    for name in metrics:
        if name not in allowed_names:
            continue
        color = colors[plot_params[name][0]]
        plot = plot_params[name][1] + 1
        ax = plt.subplot(num_plots, 1, plot)
        ax.set_title(name)
        plt.plot(t, metrics[name], color)

    plt.tight_layout(pad=1.0)
    plt.show()
    plt.pause(0.0001)
    plt.savefig("after_control.png")


dc = get_dc()
dlife = DataCenterLife(dc)
dc.start()
dlife.start()

time_elapsed = 0
start_time = time.time()

for i in range(150):
    sample_action = dc.action_space().sample()
    print(time_elapsed, "::", dc.state())
    heat_energy = dc.metrics.latest().get("heater_presence", 0)
    fan_speed = int(10*(heat_energy/2000.0))
    # fan_speed = 5
    sample_action = [fan_speed for j in sample_action]
    dc.step(sample_action)
    time_elapsed = int(time.time() - start_time)
    time.sleep(1)

plot_metrics(dc.metrics.data)

dlife.join()
dc.join()