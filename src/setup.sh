#!/bin/bash
set -e

python3 -m venv --system-site-packages ~/py3env
source ~/py3env/bin/activate
pip install tensorflow
pip install gym
pip install keras-rl
pip install keras
pip install pygame
pip install pymunk
pip install matplotlib
