import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):

        # TODO
        # layer_sizes example: [4, 10, 2]
        input_size, hidden_size, output_size  = layer_sizes[0], layer_sizes[1], layer_sizes[2]
        mu, sigma = 0, 1
        self.w_in_hidden = np.random.normal(mu, sigma, size=(hidden_size, input_size))
        self.w_hidden_out = np.random.normal(mu, sigma, size=(output_size, hidden_size))
        self.b_hidden = np.zeros((hidden_size, 1))
        self.b_output = np.zeros((output_size, 1))


    def activation(self, x):
        
        # TODO
        def sigmoid(z):
            return 1/(1 + np.exp(-z))

        return sigmoid(x)

    def forward(self, x):
        
        # TODO
        # x example: np.array([[0.1], [0.2], [0.3]])
        hidden_layer_input = np.dot(self.w_in_hidden, x) + self.b_hidden
        hidden_layer_output = self.activation(hidden_layer_input)

        output_layer_input = np.dot(self.w_hidden_out ,hidden_layer_output) + self.b_output
        return self.activation(output_layer_input)
