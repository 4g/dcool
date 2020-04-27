from part_model.partlib import Data
import dc_defs as configs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import random
import numpy as np
import json
from tqdm import tqdm

class Part:
    def __init__(self):
        self.input_params = []
        self.output_params = []
        self.model = None
        self.name = None
        self.type = None
        self.model_path = None

    def __str__(self):
        return self.__repr__()

    def to_json(self):
        s = {"IN": self.input_params,
             "OUT": self.output_params,
             "name": self.name,
             "path": str(self.model_path)}
        return s

    def __repr__(self):
        return json.dumps(self.to_json())

class System:
    def __init__(self, datafile, outdir):
        self.data_file = datafile
        self.history = None
        self.sensor_names = []
        self.parts = []
        self.epochs = 1
        self.past_size = 20
        self.rng_state = random.getstate()
        self.steps_per_epoch = 1000
        self.outdir = Path(outdir)
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 64

    def multivariate_data(self, dataset, target, split_ratio, history_size,
                          target_size, step, train=True, single_step=False):
        """
        Taken from https://www.tensorflow.org/tutorials/structured_data/time_series
        Chops into time slices of data
        splits the "dataset" into history_size chunks
        and "target" into "step" size chunks
        """
        random.setstate(self.rng_state)

        data = []
        labels = []

        start_index = history_size
        end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i - history_size, i, step)
            r = random.random()
            if r >= split_ratio and train:
                continue
            elif r < split_ratio and not train:
                continue

            data.append(dataset[indices])

            if single_step:
                labels.append(target[i + target_size])
            else:
                labels.append(target[i:i + target_size])

        return np.array(data), np.array(labels)


    def get_chillers(self):
        chillers = []
        flatten = lambda l: list([item for sublist in l for item in sublist])

        for c in [1, 2]:
            ch_t = ['CH_{c}_SUPPLY'.format(c=c)]
            ch_r = ['CH_{c}_RETURN'.format(c=c)]
            oat = ['oat1']
            oah = ['oah']
            pa_t, pa_fs, pa_st = [], [], []

            for z in [1, 2]:
                pa_t += ["SF/Z{z} PAHU {n}/RETURN_TEMP".format(z=z, n=n) for n in range(1, 9)]
                pa_fs += ['SF/Z{z} PAHU {n}/FAN_SPEED'.format(z=z, n=n) for n in range(1, 9)]
                pa_st += ['SF/Z{z} PAHU {n}/SUP_TEMP'.format(z=z, n=n) for n in range(1, 9)]

            input_params = [pa_fs, pa_t, pa_st, oat, oah]
            output_params = [ch_t, ch_r]

            input_params = flatten(input_params)
            output_params = flatten(output_params)
            part = Part()
            input_params = list(filter(self.isparam_valid, input_params))
            output_params = list(filter(self.isparam_valid, output_params))
            part.input_params = input_params
            part.output_params = output_params
            part.name = f"chiller_{c}"
            part.type = "CHILLER"
            part.model_path = self.outdir / (part.name + ".hdf5")
            chillers.append(part)

        return chillers

    def get_pahus(self):
        def _iter():
            res = []
            for z in [1, 2]:
                for p in [1, 2, 3, 4, 5, 6, 7, 8]:
                    res.append((z, p))
            return res

        pahus = []
        for z, n in _iter():
            dh_t = ['TEMP_SENSOR/Z{z}S1_PDU_TEMP_{p}'.format(z=z, p=pdu) for pdu in range(1,8)]
            dh_t += ['TEMP_SENSOR/Z{z}S2_PDU_TEMP_{p}'.format(z=z, p=pdu) for pdu in range(1,8)]
            ch_t = ['CH_1_SUPPLY', 'CH_2_SUPPLY']

            pa_t = ["SF/Z{z} PAHU {n}/RETURN_TEMP".format(z=z, n=n)]
            pa_fs = ['SF/Z{z} PAHU {n}/FAN_SPEED'.format(z=z, n=n)]
            flatten = lambda l: list([item for sublist in l for item in sublist])

            input_params = [dh_t, ch_t]
            output_params = [pa_t, pa_fs]

            input_params = flatten(input_params)
            output_params = flatten(output_params)
            part = Part()
            input_params = list(filter(self.isparam_valid, input_params))
            output_params = list(filter(self.isparam_valid, output_params))
            part.input_params = input_params
            part.output_params = output_params
            part.name = f"pahu_Z{z}_N{n}"
            part.type = "PAHU"
            part.model_path = self.outdir / (part.name + ".hdf5")
            pahus.append(part)

        return pahus

    def get_pdus(self):
        def _iter():
            res = []
            for z in [1, 2]:
                for s in [1, 2]:
                    for p in [1, 2, 3, 4, 5, 6, 7]:
                        res.append((z, s, p))
            return res

        pdus = []
        for z, s, p in _iter():
            dh_h = ['HUMIDITY_SENSOR/Z{z}S{s}_PDU_HUMI_{p}'.format(z=z, s=s, p=p)]
            dh_t = ['TEMP_SENSOR/Z{z}S{s}_PDU_TEMP_{p}'.format(z=z, s=s, p=p)]
            dh_power = ['PDU/Z{z}S{s} PDU {p}/TOTAL_POWER'.format(z=s, s=s,p=p)]

            pa_t = ['SF/Z{z} PAHU {n}/SUP_TEMP'.format(z=z, n=n) for n in range(1, 9)]
            pa_f = ['SF/Z{z} PAHU {n}/FAN_SPEED'.format(z=z, n=n) for n in range(1, 9)]

            flatten = lambda l: list([item for sublist in l for item in sublist])

            input_params = [dh_t, pa_t, pa_f, dh_power]
            output_params = [dh_t]

            input_params = flatten(input_params)
            output_params = flatten(output_params)
            part = Part()
            input_params = list(filter(self.isparam_valid, input_params))
            output_params = list(filter(self.isparam_valid, output_params))
            part.input_params = input_params
            part.output_params = output_params
            part.name = f"pdu_Z{z}_S{s}_P{p}"
            part.type = "PDU"
            part.model_path = self.outdir / (part.name + ".hdf5")
            pdus.append(part)

        return pdus

    def create_timeseries_model(self, input_params, output_params):
        input_shape = (self.past_size, len(input_params))
        output_shape = len(output_params)

        multi_step_model = tf.keras.models.Sequential()
        multi_step_model.add(tf.keras.layers.LSTM(32,
                                                  return_sequences=True,
                                                  input_shape=input_shape))

        multi_step_model.add(tf.keras.layers.LSTM(16, return_sequences=False, activation='relu'))
        multi_step_model.add(tf.keras.layers.Dense(output_shape, activation='relu'))

        multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
        return multi_step_model

    def get_timeseries_data(self, input_params, output_params):

        di = self.history[input_params]
        do = self.history[output_params]
        di = di.values
        do = do.values

        past_history = self.past_size
        future_target = 1
        STEP = 1
        TRAIN_SPLIT = 0.8

        x_train, y_train = self.multivariate_data(di, do,
                                             TRAIN_SPLIT, past_history,
                                             future_target, STEP, train=True)

        x_val, y_val = self.multivariate_data(di, do,
                                         TRAIN_SPLIT, past_history,
                                         future_target, STEP, train=False)

        return x_train, y_train, x_val, y_val

    def setup_data(self):
        data = Data(self.data_file)
        data.load()
        data = data.df
        print ("Loaded data of shape ", data.shape)
        data = data.drop(columns=["timestamp"])
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data = (data - data_min) / (data_max - data_min)
        self.history = data
        self.history_min = data_min
        self.history_max = data_max
        self.valid_params = set(self.history.columns)

    def isparam_valid(self, param):
        return param in self.valid_params

    @staticmethod
    def lrschedule(epoch, lr):
        if epoch < 2:
            lr = 0.001
        elif epoch < 5:
            lr = 0.0001
        else:
            lr = 0.00001
        return lr

    def create_DC_json(self):
        dc_dict = {}
        for part in self.parts:
            dc_dict[part.name] = part.to_json()

        json.dump(dc_dict, open(self.outdir/'dc.json', 'w'), indent=True)

    def setup(self):
        self.setup_data()
        pdus = self.get_pdus()
        pahus = self.get_pahus()
        chillers = self.get_chillers()
        self.parts = pdus + pahus + chillers
        self.create_DC_json()

    def train(self):
        reduce_lr = tf.keras.callbacks.LearningRateScheduler(schedule=System.lrschedule, verbose=True)

        for part in self.parts:
            print ("Training ", part.name)
            part.model = self.create_timeseries_model(part.input_params, part.output_params)
            part_X, part_y, x_val, y_val = self.get_timeseries_data(part.input_params, part.output_params)

            train_data_multi = tf.data.Dataset.from_tensor_slices((part_X, part_y))
            train_data_multi = train_data_multi.cache().shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE).repeat()

            val_data_multi = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            val_data_multi = val_data_multi.batch(self.BATCH_SIZE).repeat()

            print (part.model.summary())

            part.model.fit(train_data_multi, epochs=self.epochs,
                                                  steps_per_epoch=self.steps_per_epoch,
                                                  validation_data=val_data_multi,
                                                  validation_steps=self.steps_per_epoch//5,
                                                  callbacks=[reduce_lr])


            tf.keras.models.save_model(part.model, part.model_path)

    def load(self):
        for part in tqdm(self.parts, desc="Loading models ... "):
            part.model = tf.keras.models.load_model(part.model_path)

    def run(self, steps=100):
        slice = self.history.head(self.past_size)
        next_slice = slice.copy(deep=True)
        for step in tqdm(range(steps), desc="STEPPING..."):
            for part in self.parts:
                part_slice = slice[part.input_params]
                part_slice = np.expand_dims(part_slice, axis=0)
                res = part.model.predict(part_slice)
                next_slice[part.output_params] = res
            print (next_slice[['TEMP_SENSOR/Z1S1_PDU_TEMP_1']])
            slice = next_slice


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=None, required=True)
    parser.add_argument("--output", default=None, required=True)

    args = parser.parse_args()
    system = System(args.infile, args.output)

    system.setup()
    system.train()
    system.load()
    system.run()
