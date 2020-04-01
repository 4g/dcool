import csv
import json
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import keras
import dc_defs as DC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(infile, tagset=None):
    fhandler = csv.reader(open(infile, errors='ignore'))
    data = {}
    tagmap = {}
    for line in tqdm(fhandler, desc="reading data.."):
        timestamp, param, tags, value = line
        tags = (set(json.loads(tags)))
        common_tag = tags.intersection(tagset)
        if common_tag:
            tagmap[param] = common_tag
            data[param] = data.get(param, [])
            data[param].append((timestamp, float(value)))

    return data, tagmap

def preprocess(data):
    data_dict = {}
    for param in data:
        for ts, val in data[param]:
            o_ts = datetime.strptime(ts, '%d-%b-%y %H:%M:%S %p %Z')
            o_ts = o_ts.replace(second=0)
            data_dict[o_ts] = data_dict.get(o_ts, {})
            data_dict[o_ts][param] = max(val, data_dict[o_ts].get(param, -1))

    data_array = []
    for ts in sorted(data_dict):
        _d = {"timestamp": ts}
        for param in data_dict[ts]:
            _d[param] = data_dict[ts][param]
        data_array.append(_d)

    df = pd.DataFrame(data_array)
    df.dropna(thresh=7, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

# def create_model(indim, outdim, timesteps):
#     input_layer = keras.layers.Input(shape=(timesteps, indim))
#     x = keras.layers.Conv1D(filters=64, kernel_size=5, strides=1,activation='relu')(input_layer)
#     x = keras.layers.Flatten()(x)
#     # x = keras.layers.Dense(64, activation='relu')(x)
#     output = keras.layers.Dense(outdim, activation='sigmoid', use_bias=True)(x)
#
#     model = keras.Model(inputs=input_layer, outputs=output)
#     model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss='mse', metrics=['mse'])
#     return model


def create_model(indim, outdim):
    input_layer = keras.layers.Input(shape=(indim,1))
    x = keras.layers.Dense(32, activation='relu')(input_layer)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dense(32, activation='relu')(x)

    x = keras.layers.Flatten()(x)
    output = keras.layers.Dense(outdim, activation='sigmoid')(x)

    model = keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.01), loss='mse', metrics=['mse', 'mae'])
    return model


def normalize(df):
    return (df - df.min()) / (df.max() - df.min())


def partition_data(infile, tags, input_params, output_params):
    data, tagmap = load_data(infile, tags)

    print("Input params:", input_params)
    print("Output params:", output_params)

    df = preprocess(data)

    # indexNames = df[df[output_params[0]] < 10].index
    # df.drop(indexNames, inplace=True)

    X = df[input_params]
    X = normalize(X)


    y = df[output_params]
    y = normalize(y)

    sns.lineplot(data=X)
    sns.lineplot(data=y)
    plt.show()

    X = X.values
    y = y.values


    # # Chop off the points when power is less than 10
    # z = np.where(y > 10)
    #
    # X = X[z[0]]
    # y = y[z[0]]
    #

    #
    # X, _xnorms = normalize(X, norm='max', axis=1, return_norm=True)
    # y, _ynorms = normalize(y, norm='max', axis=0, return_norm=True)

    # print (_xnorms, _ynorms)

    X = np.expand_dims(X, axis=-1)

    print(X.shape, y.shape)
    return X, y

def lrschedule(epoch, lr):
    if epoch % 5 == 0:
        lr = lr / 10.0
    return lr


def main(infile):
    chiller1_model_tags = {DC.chiller1, DC.a_humidity, DC.a_temperature}
    chiller2_model_tags = {DC.chiller2, DC.a_humidity, DC.a_temperature}

    chiller1_input_params = [x.format(n="1") for x in DC.chiller_input_params]
    chiller1_output_params = [x.format(n="1") for x in DC.chiller_output_params]

    X, y = partition_data(infile,
                          chiller1_model_tags,
                          chiller1_input_params,
                          chiller1_output_params)

    chiller2_input_params = [x.format(n="2") for x in DC.chiller_input_params]
    chiller2_output_params = [x.format(n="2") for x in DC.chiller_output_params]

    test_X, test_y = partition_data(infile,
                          chiller2_model_tags,
                          chiller2_input_params,
                          chiller2_output_params)


    model = create_model(X.shape[1], y.shape[1])
    model.summary()


    reduce_lr = keras.callbacks.LearningRateScheduler(schedule=lrschedule, verbose=True)

    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                                               patience=5, min_lr=0.000001, verbose=1)
    #
    for batch_size in [4,8,16,32,64,128][::-1]:
        model.fit(X, y, batch_size=batch_size, epochs=10, verbose=1, callbacks=[reduce_lr], shuffle=True, validation_split=0.2)
        model.fit(test_X, test_y, batch_size=batch_size, epochs=10, verbose=1, callbacks=[reduce_lr], shuffle=True, validation_split=0.2)
        model.save("saved_model.hdf5", overwrite=True)

    model = keras.models.load_model("saved_model.hdf5")

    for _x, _y in [(X, y), (test_X, test_y)]:
        pred_y = model.predict(_x).flatten()
        print (pred_y.shape, _y.shape)

        plt.scatter(_y, pred_y)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.show()




    # model.evaluate(test_X, test_y)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=None, required=True)
    args = parser.parse_args()
    main(args.infile)