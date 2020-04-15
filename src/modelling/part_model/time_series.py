from part_model.partlib import System, ParameterModel, Data
import dc_defs as configs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from pathlib import Path
from tqdm import tqdm

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


def _iter():
    for z in [1]:
        for s in [1]:
            for p in [1, 2, 3, 4, 5, 6, 7]:
                yield z, s, p

def train_all(sensor_csv_file, model_dir):
    for z,s,p in _iter():
        model = train(sensor_csv_file, z, s, p)
        tf.keras.models.save_model(model, Path(model_dir) / "data_hall_ts_Z{z}S{s}P{p}.hdf5".format(z=z,s=s,p=p))

def get_params(z, s, p):
    dh_h = ['HUMIDITY_SENSOR/Z{z}S{s}_PDU_HUMI_{p}'.format(z=z, s=s, p=p)]
    dh_t = ['TEMP_SENSOR/Z{z}S{s}_PDU_TEMP_{p}'.format(z=z, s=s, p=p)]

    pa_t = ['SF/Z{z} PAHU {n}/SUP_TEMP'.format(z=z, n=n) for n in range(1, 9)]
    pa_f = ['SF/Z{z} PAHU {n}/FAN_SPEED'.format(z=z, n=n) for n in range(1, 9)]

    flatten = lambda l: list([item for sublist in l for item in sublist])

    input_params = [dh_t, dh_h, pa_t, pa_f]
    output_params = [dh_t]

    input_params = flatten(input_params)
    output_params = flatten(output_params)
    return input_params, output_params

def get_data(sensor_csv_file, z, s, p):
    input_params, output_params = get_params(z, s, p)
    if "data" not in cache:
        data = Data(sensor_csv_file)
        data.load()
        cache['data'] = data
    data = cache["data"]

    input_params = list(filter(data.is_present, input_params))
    output_params = list(filter(data.is_present, output_params))
    print (input_params)

    for i, inp in enumerate(input_params):
        if "TEMP_SENSOR" in inp:
            print(i)


    # print(input_params)
    # print(output_params)

    data_hall_i = data.partition(input_params).values
    data_hall_o = data.partition(output_params).values

    data_hall_i = normalize(dataset=data_hall_i)
    data_hall_o = normalize_minmax(dataset=data_hall_o)

    past_history = 20
    future_target = 20
    STEP = 1
    TRAIN_SPLIT = 0.8

    x_train, y_train = multivariate_data(data_hall_i, data_hall_o,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP, train=True)

    x_val, y_val = multivariate_data(data_hall_i, data_hall_o,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP, train=False)

    # print (x_train.shape, y_train.shape)
    # print (x_val.shape, y_val.shape)

    return x_train, y_train, x_val, y_val

def train(sensor_csv_file, z, s, p):
    x_train, y_train, x_val, y_val = get_data(sensor_csv_file, z, s, p)
    BATCH_SIZE = 64
    BUFFER_SIZE = 64

    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(32,
                                              return_sequences=True,
                                              input_shape=x_train.shape[-2:]))

    multi_step_model.add(tf.keras.layers.LSTM(16, return_sequences=True, activation='relu'))
    multi_step_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='relu')))

    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
    multi_step_model.summary()

    def lrschedule(epoch, lr):
        if epoch < 5:
            lr = 0.001
        elif epoch < 10:
            lr = 0.0001
        else:
            lr = 0.00001
        return lr

    reduce_lr = keras.callbacks.LearningRateScheduler(schedule=lrschedule, verbose=True)

    EVALUATION_INTERVAL = 200
    EPOCHS = 15

    multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                              steps_per_epoch=2000,
                                              validation_data=val_data_multi,
                                              validation_steps=100,
                                              callbacks=[reduce_lr])

    return multi_step_model

def test(sensor_csv_file, model_dir):
    models = {}
    states = {}
    time_index = 0
    for z, s, p in tqdm(list(_iter()), "Loading models"):
        model_name = "data_hall_ts_Z{z}S{s}P{p}.hdf5".format(z=z,s=s,p=p)
        models[model_name] = tf.keras.models.load_model(Path(model_dir) / model_name)
        x_train, y_train, x_val, y_val = get_data(sensor_csv_file, z, s, p)
        states[model_name] = x_train
        print (x_train[0][0])

    prev_temp = None
    temps = []
    initial_states = {m:states[m][0:1] for m in models}

    np.set_printoptions(precision=3)

    for iter in range(1000):
        for model_name in models:
            output = models[model_name].predict(initial_states[model_name]).flatten()
            x = initial_states[model_name]
            x[0, :, 0] = output
            initial_states[model_name] = x

        temps = np.asarray([initial_states[model_name][0, 0, 0] for model_name in models])
        distance = np.linalg.norm(prev_temp - temps) if prev_temp is not None else 1

        print(temps, distance)
        if distance < 0.0001:
            print ("STEADY STATE ACHIEVED, CHANGING TEMPS ==============")
            time_index += 20
            for model_name in models:
                initial_states[model_name] = states[model_name][time_index:time_index+1]

        prev_temp = temps

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=None, required=True)
    parser.add_argument("--output", default=None, required=True)

    args = parser.parse_args()
    # train_all(args.infile, args.output)
    test(args.infile, args.output)