======
dc-env
======
1. Game engine based emulator for a DC rack.
2. Run simple RL algorithms on the emulator

-----
Setup
-----
cd src/
chmod +x setup.sh
./setup.sh

----
Run
----
cd dc-env/
source ~/py3env/bin/activate
# Runs a dc emulator
python gym_test.py

# Runs dc emulator and trpo from keras-rl on emulator
python rl_train.py
