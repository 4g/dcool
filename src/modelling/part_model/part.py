import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import csv
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class System:
    def __init__(self):
        self.connection_graph = {}
        self.parts = []

    def add_part(self, part):
        self.parts.append(part)

    def train(self):
        for part in self.parts:
            part.train_model()

    def predict(self, inputs):
        outputs = inputs
        for part in self.parts:
            outputs = part.predict(outputs)

class Model:
    @staticmethod
    def create_model(indim, outdim):
        input_layer = keras.layers.Input(shape=(indim))
        x = keras.layers.Dense(32, activation='relu')(input_layer)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        output = keras.layers.Dense(outdim, activation='relu')(x)

        model = keras.Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=keras.optimizers.SGD(lr=0.01), loss='mse', metrics=['mse', 'mae'])
        return model

    @staticmethod
    def lrschedule(epoch, lr):
        if epoch < 5:
            lr = 0.001
        elif epoch < 10:
            lr = 0.0001
        elif epoch < 15:
            lr = 0.00001
        elif epoch < 25:
            lr = 0.000001
        else:
            lr = 0.0000001
        return lr


class Part:
    def __init__(self, name):
        self.name = name
        self.model = None

    def set_params(self, inparams, outparams):
        self.input_params = inparams
        self.output_params = outparams


    def set_data(self, data):
        # indexNames = data[data[self.output_params[0]] < 10].index
        # data = data.drop(indexNames)

        X = data[self.input_params]
        y = data[self.output_params]

        X, X_norm = Data.mean_normalize(X)
        y, y_norm = Data.minmax_normalize(y)

        sns.lineplot(data=X)
        sns.lineplot(data=y)
        plt.show()

        X = X.values
        y = y.values

        self.X = X
        self.y = y

    def train_model(self):
        if self.model is None:
            self.model = Model.create_model(len(self.input_params), len(self.output_params))

        reduce_lr = keras.callbacks.LearningRateScheduler(schedule=Model.lrschedule, verbose=True)
        for iteration in range(1):
            for batch_size in [4, 64]:
                Data.shuffle_together(self.X, self.y)
                self.model.fit(self.X, self.y, batch_size=batch_size,
                          epochs=10,
                          verbose=1,
                          callbacks=[reduce_lr],
                          shuffle=True,
                          validation_split=0.2)

    def predict(self, input):
        output = self.model.predict(input)
        output_dict = {}
        return output_dict


class Data:
    def __init__(self, f):
        self.infile = f

    def load(self):
        fhandler = csv.reader(open(self.infile, errors='ignore'))
        data = {}
        for line in tqdm(fhandler, desc="reading data ..."):
            timestamp, param, tags, value = line
            data[param] = data.get(param, [])
            data[param].append((timestamp, float(value)))

        data_dict = {}
        for param in tqdm(data, desc="Preparing by timestamps ..."):
            for ts, val in data[param]:
                o_ts = datetime.strptime(ts, '%d-%b-%y %H:%M:%S %p %Z')
                o_ts = o_ts.replace(second=0)
                data_dict[o_ts] = data_dict.get(o_ts, {})
                data_dict[o_ts][param] = max(val, data_dict[o_ts].get(param, -1))

        data_array = []
        for ts in tqdm(sorted(data_dict), desc="preparing DataFrame ..."):
            _d = {"timestamp": ts}
            for param in data_dict[ts]:
                _d[param] = data_dict[ts][param]
            data_array.append(_d)

        df = pd.DataFrame(data_array)
        df.dropna(thresh=7, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        self.df = df

    def partition(self, params):
        X = self.df[params]
        return X

    @staticmethod
    def mean_normalize(df):
        df_min = df.min()
        df_max = df.max()
        df_mean = df.mean()
        return ((df - df_mean) / (df_max - df_min)), list(zip(df_min.values, df_max.values))

    @staticmethod
    def minmax_normalize(df):
        df_min = df.min()
        df_max = df.max()
        return ((df - df_min) / (df_max - df_min)), list(zip(df_min.values, df_max.values))

    @staticmethod
    def shuffle_together(a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def preprocess(self):
        pass

def main():
    import dc_defs as configs
    param_tuples = configs.param_dict

    system = System()
    data = Data(configs.data_file)
    data.load()
    # print(list(data.df.columns))

    for part_name in param_tuples:
        _inparams, _outparams, part_number = param_tuples[part_name]
        part = Part(part_name)
        for pn in range(1, part_number + 1):
            inparams = [i.format(n=pn) for i in _inparams]
            outparams = [i.format(n=pn) for i in _outparams]
            part.set_params(inparams, outparams)
            part.set_data(data.df)

            desc = f"Training part {part_name} with input {inparams} and output {outparams}"
            print(desc)

            part.train_model()
            system.add_part(part)

if __name__ == "__main__":
    main()

