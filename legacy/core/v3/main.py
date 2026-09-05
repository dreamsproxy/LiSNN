import neurons

class network:
    def __init__(self, input_shape, hidden_size, output_shape) -> None:
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.output_shape = output_shape
        self.neurons = neurons.initialize_population
