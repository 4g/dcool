from tensorflow import keras
from tqdm import tqdm

class System:
    def __init__(self):
        self.connection_graph = {}
        self.parts = []

    def add_part(self, part):
        self.parts.append(part)

    def train(self):
        for part in self.parts:
            part.train_model()

    def predict(self, inputs):
        outputs = inputs
        for part in self.parts:
            outputs = part.predict(outputs)



class Part:
    def __init__(self, name, input_params, output_params):
        self.input_params = input_params
        self.output_params = output_params
        self.model = Part.create_model(len(self.input_params), len(self.output_params))

    @staticmethod
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

    def load_data(self, indata, outdata):
        pass

    def train_model(self):
        pass

    def predict(self, input):
        output = self.model.predict(input)
        output_dict = {}
        return output_dict


class Data:
    def __init__(self):
        pass

    def load_data(self):
        pass

    def partition(self):
        pass

    def preprocess(self):
        pass

def main():
    param_tuples = load_part_params()
    system = System()

    for part_name, inparams, outparams in param_tuples:
        desc = F"""Training part {part_name}"""
        print (desc)
        part = Part(part_name, inparams, outparams)
        input_data = load_param_data(inparams)
        output_data = load_param_data(outparams)
        part.load_data(input_data, output_data)
        part.train_model()
        system.add_part(part)

    system_map = load_system_part_map()
