import os
from pathlib import Path
import csv
from tqdm import tqdm
from datetime import datetime
import math
import json

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


def serialize_directory(directory, output):
    param_files = {}
    outfile = open(output, 'w')
    writer = csv.writer(outfile)

    for dir_path, file_name in tqdm(list(list_files(directory))):
        file_path = dir_path / file_name
        tags = str(dir_path).replace(directory, "").split("/")

        data = csv.reader(open(file_path, errors='ignore'))
        header = next(data)
        param_name = header[1]
        if not param_name:
            param_name = file_name.replace(".csv", "").split("_", maxsplit=1)[1]

        param_name = param_name.split('(')[0].strip()
        param_name = param_name.replace("CTRLS/", "")
        param_name = param_name.replace("points/", "")

        param_files[param_name] = param_files.get(param_name, [])
        param_files[param_name].append(file_path)

        for line in data:
            param_name = param_name.strip()
            timestamp = line[0].strip()
            # datetime_object = datetime.strptime(timestamp, '%d-%b-%y %H:%M:%S %p %Z')
            value = float(line[1])
            if math.isnan(value):
                continue

            o = [timestamp, param_name, json.dumps(tags), value]
            writer.writerow(o)

    outfile.close()

def create_datafile(flatcsv, output):
    reader = csv.reader(open(flatcsv, errors='ignore'))
    data = dict()
    params = set()

    outfile = open(output, 'w')
    writer = csv.writer(outfile)

    for line in tqdm(reader, desc="Reading " + flatcsv):
        timestamp, param, value = line
        data[timestamp] = data.get(timestamp, {})
        data[timestamp][param] = value
        params.add(param)

    params = list(params)
    header = ["timestamp"] + params
    writer.writerow(header)

    for timestamp in tqdm(data, desc="Writing " + output):
        values = []
        for param in params:
            value = data[timestamp].get(param, None)
            values.append(value)
        row = [timestamp] + values
        writer.writerow(row)

    outfile.close()

if __name__ == "__main__":
    import sys
    serialize_directory(sys.argv[1], sys.argv[2])
    # create_datafile(sys.argv[1], sys.argv[2])
