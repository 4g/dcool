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
    def __init__(self, input_params, output_params):
        self.input_params = input_params
        self.output_params = output_params
        self.model = None

    def load_data(self, infile):
        pass

    def train_model(self):
        pass

    def predict(self, input):
        output = self.model.predict(input)
        output_dict = {}
        return output_dict