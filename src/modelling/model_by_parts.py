import csv
import json
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from tensorflow import keras
import dc_defs as DC
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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

def mean_normalize(df):
    df_min = df.min()
    df_max = df.max()
    df_mean = df.mean()
    return ((df - df_mean) / (df_max - df_min)), list(zip(df_min.values, df_max.values))

def minmax_normalize(df):
    df_min = df.min()
    df_max = df.max()
    return ((df - df_min) / (df_max - df_min)), list(zip(df_min.values, df_max.values))

def partition_data(infile, tags, input_params, output_params):
    data, tagmap = load_data(infile, tags)

    print("Input params:", input_params)
    print("Output params:", output_params)

    df = preprocess(data)

    indexNames = df[df[output_params[0]] < 10].index
    df.drop(indexNames, inplace=True)

    X = df[input_params]
    y = df[output_params]

    # sns.lineplot(data=X)
    # sns.lineplot(data=y)
    # plt.show()

    X, X_norm = mean_normalize(X)
    y, y_norm = minmax_normalize(y)

    X = X.values
    y = y.values

    print(X.shape, y.shape)
    return X, y, X_norm, y_norm

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

def shuffle_together(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def get_data(infile):
    X = None
    y = None

    for chiller_index in [1, 2]:
        chiller_model_tags = {DC.chiller.format(n=chiller_index), DC.a_humidity, DC.a_temperature}

        chiller_input_params = [x.format(n=chiller_index) for x in DC.chiller_input_params]
        chiller_output_params = [x.format(n=chiller_index) for x in DC.chiller_output_params]

        _X, _y, X_norm, y_norm = partition_data(infile,
                                          chiller_model_tags,
                                          chiller_input_params,
                                          chiller_output_params)

        print (X_norm, y_norm)

        if X is None:
            X = _X
            y = _y
        else:
            X = np.concatenate((X, _X), axis=0)
            y = np.concatenate((y, _y), axis=0)

    return X, y

def main(infile, model_file, mode):
    num_input_params = len(DC.chiller_input_params)
    num_output_params = len(DC.chiller_output_params)
    model = create_model(num_input_params, num_output_params)
    model.summary()


    if mode == "train":
        X, y = get_data(infile)
        model = keras.models.load_model(model_file)
        reduce_lr = keras.callbacks.LearningRateScheduler(schedule=lrschedule, verbose=True)
        for batch_size in [4, 16, 64, 128]:
            shuffle_together(X,  y)
            model.fit(X, y, batch_size=batch_size,
                      epochs=10,
                      verbose=1,
                      callbacks=[reduce_lr],
                      shuffle=True,
                      validation_split=0.2)

            model.save(model_file, overwrite=True)


    if mode == "test":
        test_model_variation(model_file)
        # model = keras.models.load_model(model_file)
        # X, y = get_data(infile)
        # y = y.flatten()
        # pred_y = model.predict(X).flatten()
        # print (pred_y.shape, y.shape)
        # data = {'predicted': pred_y, 'real': y}
        # df = pd.DataFrame.from_dict(data)
        # print (df)
        # sns.lineplot(data=df)
        # plt.show()

    # model.evaluate(test_X, test_y)

def test_model_variation(model_file):
    """
    (20.3, 21.4), (25.3, 32.5), (34.1, 82.1), (20.1, 24.5), (14.9, 22.2)]
     [(24.0, 131.0)
    """

    """
    ['CH_{n}_RETURN',
        'oat1',
        'oah',
        'CW_{n}_RETURN',
        'CH_{n}/Evaporator Saturation Temp']
    """
    model = keras.models.load_model(model_file)
    minx, maxx = zip(*[(20.3, 21.4), (25.3, 32.5), (34.1, 82.1), (20.1, 24.5), (14.9, 22.2)])
    minx = np.asarray(minx)
    maxx = np.asarray(maxx)
    num_samples = 20
    xa = []
    ya = []

    fval = 0.3
    x_index = 1
    y_index = 2
    X = []
    for x in range(num_samples):
        for y in range(num_samples):
            v = [fval, fval, fval, fval, fval]
            x_val = x / num_samples
            y_val = y / num_samples
            v[x_index] = x_val
            v[y_index] = y_val
            X.append(v)
            xa.append(x_val)
            ya.append(y_val)

    X = np.asarray(X)
    z = model.predict(X, batch_size=1024, verbose=True)
    z = z[:, 0]
    print(X.shape, z.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(xa, ya, z, cmap=plt.cm.jet, linewidth=0.2)
    ax.set_xlabel(DC.chiller_input_params[x_index])
    ax.set_ylabel(DC.chiller_input_params[y_index])
    ax.set_zlabel('Power')

    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=None, required=True)
    parser.add_argument("--model", default=None, required=True)
    parser.add_argument("--mode", default=None, required=True)

    args = parser.parse_args()
    main(args.infile, args.model, args.mode)