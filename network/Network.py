class Network:

    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer
    def add_layer(self, layer):
        self.layers.append(layer)

    # set loss
    def set_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for a given input
    def predict(self, input_data):
        samples = len(input_data)
        result = []

        # run network
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output);
            result.append(output)

        return result

    # train the network
    def train(self, x_train, y_train, epochs, learning_rate):
        # dimension
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                #compute loss for display purpose only
                err += self.loss(y_train[j], output)

                #backward propagation
                error = self.loss_prime(y_train[j], output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error
            error /= samples
            print('epoch %d/%d      error=%f' % (i+1, epochs, err));
