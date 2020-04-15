from part_model.partlib import System, ParameterModel, Data
import dc_defs as configs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from pathlib import Path

rng_state = random.getstate()


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

def train_all(sensor_csv_file, model_dir):
    def _iter():
        for z in [1, 2]:
            for s in [1, 2]:
                for p in [1, 2, 3, 4, 5, 6, 7]:
                    yield z, s, p

    for z,s,p in _iter():
        model = train(sensor_csv_file, z, s, p)
        tf.keras.models.save_model(model, Path(model_dir) / "data_hall_ts_Z{z}S{s}P{p}.hdf5".format(z=z,s=s,p=p))


def train(sensor_csv_file, z, s, p):
    dh_h = ['HUMIDITY_SENSOR/Z{z}S{s}_PDU_HUMI_{p}'.format(z=z, s=s, p=p)]
    dh_t = ['TEMP_SENSOR/Z{z}S{s}_PDU_TEMP_{p}'.format(z=z, s=s, p=p)]

    pa_t = ['SF/Z{z} PAHU {n}/SUP_TEMP'.format(z=z, n=n) for n in range(1, 9)]
    pa_f = ['SF/Z{z} PAHU {n}/FAN_SPEED'.format(z=z, n=n) for n in range(1, 9)]

    flatten = lambda l:list(set([item for sublist in l for item in sublist]))

    input_params = [dh_h, dh_t, pa_t, pa_f]
    output_params = [dh_t]

    input_params = flatten(input_params)
    output_params = flatten(output_params)

    data = Data(sensor_csv_file)
    data.load()

    input_params = list(filter(data.is_present, input_params))
    output_params = list(filter(data.is_present, output_params))

    print(input_params)
    print(output_params)

    data_hall_i = data.partition(input_params).values
    data_hall_o = data.partition(output_params).values

    data_hall_i = normalize(dataset=data_hall_i)
    data_hall_o = normalize(dataset=data_hall_o)

    # data_hall_i.plot(subplots=False)
    # plt.show()

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

    print (x_train.shape, y_train.shape)
    print (x_val.shape, y_val.shape)

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
    multi_step_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(output_params))))

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
                                              steps_per_epoch=3000,
                                              validation_data=val_data_multi,
                                              validation_steps=750,
                                              callbacks=[reduce_lr])

    return multi_step_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=None, required=True)
    parser.add_argument("--output", default=None, required=True)

    args = parser.parse_args()
    train_all(args.infile, args.output)

