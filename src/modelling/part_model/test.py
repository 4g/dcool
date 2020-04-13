from part_model.partlib import System, ParameterModel, Data
import dc_defs as configs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def model_test(model, X, y, idname, debug_dir):
    # model = part.model
    score = model.evaluate(X, y)
    print (score)

    y = y.flatten()
    pred_y = model.predict(X).flatten()

    print (pred_y.shape, y.shape)
    data = {'predicted': pred_y, 'real': y}
    df = pd.DataFrame.from_dict(data)
    sns.lineplot(data=df)
    img_path = Path(debug_dir) / (idname + ".png")
    plt.savefig(img_path)
    plt.clf()
    return score

def test(sensor_csv_file, model_dir, debug):
    part_configs = configs.parts
    data = Data(sensor_csv_file)
    data.load()

    for part_name in part_configs:
        part = ParameterModel(part_name)
        part.load_model(model_dir)

        for part_config in part_configs[part_name]:
            inparams = part_config["input"]
            outparams = part_config["output"]
            part_id = part_config["name"]

            inparams = list(filter(data.is_present, inparams))

            part.set_params(inparams, outparams)
            part.add_data(data.df)
            print (f"Testing part {part_name} {part_id}")

            model_test(part.model, part.X, part.y, part_id, debug_dir=debug)
            part.reset_data()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=None, required=True)
    parser.add_argument("--models", default=None, required=True)
    parser.add_argument("--debug", default=None, required=True)

    args = parser.parse_args()
    test(args.infile, args.models, args.debug)

