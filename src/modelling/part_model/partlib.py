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
from pathlib import Path

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
        if epoch < 15:
            lr = 0.001
        elif epoch < 30:
            lr = 0.0001
        else:
            lr = 0.00001
        return lr


class ParameterModel:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.X = None
        self.y = None

    def set_params(self, inparams, outparams):
        self.input_params = inparams
        self.output_params = outparams

    def add_data(self, data):
        # indexNames = data[data[self.output_params[0]] < 10].index
        # data = data.drop(indexNames)

        X = data[self.input_params]
        y = data[self.output_params]
        #
        # print (X)
        # print (y)

        X, X_norm = Data.mean_normalize(X)
        y, y_norm = Data.minmax_normalize(y)

        # sns.lineplot(data=X)
        # sns.lineplot(data=y)
        # plt.show()

        _X = X.to_numpy()
        _y = y.to_numpy()

        # print (_y)

        if self.X is None:
            self.X = _X
            self.y = _y
        else:
            self.X = np.concatenate((self.X, _X), axis=0)
            self.y = np.concatenate((self.y, _y), axis=0)

    def reset_data(self):
        self.X = None
        self.y = None

    def train_model(self):
        if self.model is None:
            self.model = Model.create_model(len(self.input_params), len(self.output_params))

        from sklearn.preprocessing import scale
        # self.X = self.X.to_numpy()
        # self.y = self.y.to_numpy()

        print (self.X.shape, self.y.shape)
        print (self.model.summary())
        #
        # sns.lineplot(data=self.X)
        # sns.lineplot(data=self.y)
        # plt.savefig(self.name + ".png")
        #
        # self.X = scale(self.X, axis=0)
        # self.y = scale(self.y, axis=0, with_mean=False)

        reduce_lr = keras.callbacks.LearningRateScheduler(schedule=Model.lrschedule, verbose=True)
        for iteration in range(1):
            for batch_size in [64, 16, 64]:
                Data.shuffle_together(self.X, self.y)
                self.model.fit(self.X, self.y, batch_size=batch_size,
                          epochs=30,
                          verbose=1,
                          callbacks=[reduce_lr],
                          shuffle=True,
                          validation_split=0.2)

    def predict(self, input):
        output = self.model.predict(input)
        output_dict = {}
        return output_dict

    def save_model(self, model_dir):
        model_file = Path(model_dir) / (self.name + ".hdf5")
        keras.models.save_model(self.model, model_file)

    def load_model(self, model_dir):
        model_file = Path(model_dir) / (self.name + ".hdf5")
        print (model_file)
        self.model = keras.models.load_model(model_file)


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
        for param in tqdm(data, desc="Preparing each param ..."):
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

    def is_present(self, param):
        return param in set(self.df.columns)

    def preprocess(self):
        pass