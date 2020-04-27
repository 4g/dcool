from part_model.partlib import System, ParameterModel, Data
import dc_defs as configs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

class System:
    def __init__(self):
        self.history = None
        self.sensor_names = []
        self.parts = []

    def set_state(self, state):

    def get_state(self, size=20):

    def step(self):
        slice = self.get_state(size=20)
        output = []
        output_names = []
        for part in self.parts:
            predictions = part.predict(slice)
            output.extend(predictions)
            output_names.extend(part.output_names)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=None, required=True)
    parser.add_argument("--models", default=None, required=True)
    parser.add_argument("--debug", default=None, required=True)

    args = parser.parse_args()
    test(args.infile, args.models, args.debug)

