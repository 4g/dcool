import os
from pathlib import Path
import csv
from tqdm import tqdm
from datetime import datetime
import math
import json
import pandas as pd

def get_dfpath(directory):
    dffile = Path(directory) / 'flat_df.csv'
    return dffile

def get_plotdir(directory):
    return Path(directory) / 'plots/'

def list_files(directory):
    for path, dirs, files in os.walk(directory):
        dir_path = Path(path)
        for f in files:
            yield dir_path, f

def dir_to_json(path):
    d = {'name': os.path.basename(path), 'full_path': path}
    if os.path.isdir(path):
        d['type'] = "directory"
        d['children'] = [dir_to_json(os.path.join(path, x)) for x in os.listdir(path)]
    else:
        d['type'] = "file"
    return d

def csv_header(csv_file):
    data = csv.reader(open(csv_file, errors='ignore'))
    header = next(data)
    return header

def read_all_csvs(directory):
    for dir_path, file_name in list_files(directory):
        file_path = dir_path / file_name
        data = csv.reader(open(file_path, errors='ignore'))
        for line in data:
            yield file_name, file_path, line


def serialize_directory(directory, dffile):
    param_files = {}
    data = {}

    for dir_path, file_name in tqdm(list(list_files(directory)), "Parsing all files ... "):
        file_path = dir_path / file_name

        _d = csv.reader(open(file_path, errors='ignore'))
        header = next(_d)
        param_name = header[1]
        if not param_name:
            param_name = file_name.replace(".csv", "").split("_", maxsplit=1)[1]

        param_name = param_name.split('(')[0].strip()
        param_name = param_name.replace("CTRLS/", "")
        param_name = param_name.replace("points/", "")

        param_files[param_name] = param_files.get(param_name, [])
        param_files[param_name].append(file_path)

        for line in _d:
            param_name = param_name.strip()
            timestamp = line[0].strip()
            value = float(line[1])
            if math.isnan(value):
                continue

            data[param_name] = data.get(param_name, [])
            data[param_name].append((timestamp, float(value)))

    data_dict = {}
    time_cache = {}
    for param in tqdm(data, desc="Preparing each param ..."):
        for ts, val in data[param]:
            if ts not in time_cache:
                o_ts = datetime.strptime(ts, '%d-%b-%y %H:%M:%S %p %Z')
                o_ts = o_ts.replace(second=0)
                time_cache[ts] = o_ts

            o_ts = time_cache[ts]
            data_dict[o_ts] = data_dict.get(o_ts, {})
            data_dict[o_ts][param] = max(val, data_dict[o_ts].get(param, -1))

    data_array = []
    for ts in tqdm(sorted(data_dict), desc="Preparing DataFrame ..."):
        _d = {"timestamp": ts}
        for param in data_dict[ts]:
            _d[param] = data_dict[ts][param]
        data_array.append(_d)

    df = pd.DataFrame(data_array)

    # df.dropna(thresh=7, inplace=True)
    # df.fillna(method='ffill', inplace=True)
    # df.fillna(method='bfill', inplace=True)

    df.to_csv(dffile, index=False)

def load_df(dffile, clean=True):
    df = pd.read_csv(dffile)
    if clean:
        df.dropna(thresh=7, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
    return df

def to_filename(s):
    return s.replace("/","_").replace(" ", "_")

def plots(dffile, plot_dir):
    import seaborn as sns
    import matplotlib.pyplot as plt
    df = load_df(dffile)
    columns = df.columns.tolist()

    # skip the timestamp column
    columns = columns[1:]

    for col in tqdm(columns, "Saving plots ... "):
        x = df[col]
        sns.lineplot(data=x, ci=None)
        plt_path = plot_dir / (to_filename(col) + ".png")
        plt.savefig(plt_path)
        plt.clf()

def analysis(dffile):
    df = load_df(dffile, clean=False)
    print(f"Data shape = {df.shape}")
    print("Null counts ====== \n", df.isna().sum(), "\n===============")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default=None, required=True)
    parser.add_argument("--outdir", default=None, required=True)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dffile = get_dfpath(args.outdir)
    serialize_directory(args.indir, dffile)

    plot_dir = get_plotdir(args.outdir)
    os.makedirs(plot_dir, exist_ok=True)
    plots(dffile, plot_dir)

    analysis(dffile)
