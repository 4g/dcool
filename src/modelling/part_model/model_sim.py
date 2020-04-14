from part_model.partlib import System, ParameterModel, Data
import dc_defs as configs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def run_model_simulation(sensor_csv_file, model_dir, debug):
    chiller = load_part(configs.chiller)
    pahu = load_part(configs.pahu)
    data_hall = load_part(configs.data_hall)

    oah = None
    oat = None
    datetime = None

    data_hall.set_state(data, datetime)
    pahu.set_state(data, datetime)
    chiller.set_state(data, datetime)

    for i in range(1000):
        supply_air, return_water = pahu.next()
        data_hall_state, return_air = data_hall.next()
        supply_water = chiller.next()

        chiller.set_var(return_water)
        pahu.set_var(supply_water, return_air)
        data_hall.set_var(supply_air)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=None, required=True)
    parser.add_argument("--models", default=None, required=True)
    parser.add_argument("--debug", default=None, required=True)

    args = parser.parse_args()
    test(args.infile, args.models, args.debug)

