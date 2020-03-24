import pandas
import logging
import numpy as np
from sklearn.preprocessing import normalize
import keras

log = logging.getLogger()

def get_data(infile, input_params, output_params, shuffle=True):
    data = pandas.read_csv(infile)
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)

    data = data.fillna(method='backfill')

    input_data = data[input_params]
    input_data = input_data.values

    output_data = data[output_params]
    output_data = output_data.values
    rows = len(input_data)

    log.debug("Read {infile} {rows}".format(infile=infile, rows=rows))
    return input_data, output_data

def create_model(indim, outdim, timesteps):
    input_layer = keras.layers.Input(shape=(timesteps, indim))
    x = keras.layers.Conv1D(filters=64, kernel_size=5, strides=5,)(input_layer)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(outdim, activation='relu', use_bias=True)(x)

    model = keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse', metrics=['mse'])
    return model

def split(l, ratio):
    test_size = int(len(l) * ratio)
    return l[test_size:], l[:test_size]

def test(test_data, model):
    return number

def lr_schedule():
    def schedule(epoch):
        return

    return schedule

def main():
    infile = "../data/AHU_dataset_ami/AHU1_Office.csv"

    model_input_params = [""]
    model_output_params = ["Avg_ZoneTemp"]

    X, y = get_data(infile, model_input_params, model_output_params, shuffle=False)

    X = normalize(X)
    y = normalize(y)

    # X = np.expand_dims(X, axis=-1)

    test_ratio = 0.2
    time_steps = 10
    # test_size = len(X) * test_ratio
    # trainX, testX = split(X, test_ratio)
    # trainY, testY = split(y, test_ratio)

    data_gen = keras.preprocessing.sequence.TimeseriesGenerator(X, y,
                        length=time_steps, sampling_rate=1,
                        batch_size=32)


    model = create_model(X.shape[1], y.shape[1], timesteps=time_steps)
    model.summary()

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.00001, verbose=1)

    model.fit(data_gen, epochs=500, verbose=1, callbacks=[reduce_lr], shuffle=True)
    # score = test(testX, testY, model)

if __name__ == '__main__':
    main()
