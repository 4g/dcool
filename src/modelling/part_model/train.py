from part_model.partlib import System, ParameterModel, Data
import dc_defs as configs

def train(sensor_csv_file, model_dir):
    part_configs = configs.parts
    data = Data(sensor_csv_file)
    data.load()

    for part_name in part_configs:
        part = ParameterModel(part_name)
        for part_config in part_configs[part_name]:
            inparams = part_config["input"]
            outparams = part_config["output"]

            inparams = list(filter(data.is_present, inparams))

            part.set_params(inparams, outparams)
            desc = f"Adding part {part_name} with input {inparams} and output {outparams}"
            print(desc)
            part.add_data(data.df)

        print (f"Training part {part_name}")
        part.train_model()
        part.save_model(model_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default=None, required=True)
    parser.add_argument("--output", default=None, required=True)

    args = parser.parse_args()
    train(args.infile, args.output)

