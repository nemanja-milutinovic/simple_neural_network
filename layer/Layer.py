# base class
class Layer:

    def __init__(self):
        self.input = None
        self.output = None

    # computing output of the layer for given inpuut
    def forward_propagation(self, input):
        raise NotImplementedError

    # computing dE/dX for given dE/dY (and updating parameters if any)
    # dE/dX, dE/dY- derivative of total error in respect to input i.e in respect to output
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
