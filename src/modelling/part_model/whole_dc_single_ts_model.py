from part_model.partlib import System, ParameterModel, Data
import dc_defs as configs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate

rng_state = random.getstate()
cache = {}


def multivariate_data(dataset, target, split_ratio, history_size,
                      target_size, step, train=True, single_step=False):
    """
    Taken from https://www.tensorflow.org/tutorials/structured_data/time_series
    Chops into time slices of data
    splits the "dataset" into history_size chunks
    and "target" into "step" size chunks
    """
    random.setstate(rng_state)

    data = []
    labels = []

    start_index = history_size
    end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        r = random.random()
        if r >= split_ratio and train:
            continue
        elif r < split_ratio and not train:
            continue


        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])


    return np.array(data), np.array(labels)

def normalize(dataset):
    data_mean = dataset.mean(axis=0)
    data_std = dataset.std(axis=0)
    dataset = (dataset - data_mean) / data_std
    return dataset

def normalize_minmax(dataset):
    data_min = dataset.min(axis=0)
    data_max = dataset.max(axis=0)
    dataset = (dataset - data_min) / (data_max - data_min)
    return dataset

def train_all(sensor_csv_file, model_dir):
    model = train(sensor_csv_file)
    tf.keras.models.save_model(model, Path(model_dir) / "full_dc.hdf5")

def get_data(sensor_csv_file):
    # input_params, output_params = get_params(z, s, p)
    if "data" not in cache:
        data = Data(sensor_csv_file)
        data.load()
        cache['data'] = data
    data = cache["data"]

    names = set(data.df.columns.tolist())
    fil = lambda l,y : sorted(list(filter(lambda x: y in x.lower(), l)))
    sps =  fil(names, 'ra_sp') + fil(names, 'sa_sp')
    pahus = fil(names, 'pahu')
    pdus = fil(names, 'pdu')
    chiller = fil(names, 'ch') + fil(names, 'ct') + fil(names, 'cw') + fil(names, 'pump')
    outer = fil(names, 'oa')
    # print (pahus, '\n\n', pdus, '\n\n', chiller, '\n\n', outer)
    remaining = names - set(pahus) - set(chiller) - set(pdus) - set(outer)

    parameters = sorted(set(pahus + pdus + chiller + outer) - set(sps))

    data = data.partition(parameters)

    data = normalize_minmax(dataset=data)
    data = data.dropna(axis=1, how='all')
    cache["params"] = data.columns.tolist()
    data = data.values

    past_history = 20
    future_target = 20
    STEP = 1
    TRAIN_SPLIT = 0.8

    x_train, y_train = multivariate_data(data, data,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP, train=True)

    x_val, y_val = multivariate_data(data, data,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP, train=False)

    print (x_train.shape, y_train.shape)
    print (x_val.shape, y_val.shape)

    return x_train, y_train, x_val, y_val

def train(sensor_csv_file):
    x_train, y_train, x_val, y_val = get_data(sensor_csv_file)
    print (x_train.shape)
    BATCH_SIZE = 32
    BUFFER_SIZE = 32

    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    #
    # for x,y in train_data_multi.take(10):
    #     print(x, y)

    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(16,
                                              return_sequences=True,
                                              input_shape=x_train.shape[-2:], activation='relu'))

    multi_step_model.add(tf.keras.layers.LSTM(16, return_sequences=True, activation='relu'))
    multi_step_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(x_train.shape[-1], activation='relu')))

    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
    multi_step_model.summary()

    def lrschedule(epoch, lr):
        if epoch < 5:
            lr = 0.001
        elif epoch < 10:
            lr = 0.0001
        elif epoch < 20:
            lr = 0.00001
        elif epoch < 30:
            lr = 0.000001
        else:
            lr = 0.0000001
        return lr

    reduce_lr = keras.callbacks.LearningRateScheduler(schedule=lrschedule, verbose=True)

    EVALUATION_INTERVAL = 200
    EPOCHS = 30

    multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                              steps_per_epoch=4000,
                                              validation_data=val_data_multi,
                                              validation_steps=1000,
                                              callbacks=[reduce_lr])

    return multi_step_model

def test(sensor_csv_file, model_dir):
    models = {}
    states = {}
    time_index = 0

    model_name = "full_dc.hdf5"
    model = tf.keras.models.load_model(Path(model_dir) / model_name)
    x_train, y_train, x_val, y_val = get_data(sensor_csv_file)

    print (cache["params"])
    states[model_name] = x_train

    x = x_train[0:1]
    for i in range(5):
        print(x[0, :, 1])
        x = model.predict(x)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=None, required=True)
    parser.add_argument("--output", default=None, required=True)

    args = parser.parse_args()
    # train_all(args.infile, args.output)
    test(args.infile, args.output)
