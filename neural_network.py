import matrix

'''
    File name: main.py
    Author: Michael Berge
    Date created: 7/19/2018
    Python Version: 3.8.1
'''

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.__input_nodes = input_nodes
        self.__hidden_nodes = hidden_nodes
        self.__output_nodes = output_nodes

        # weights
        self.__weights_ih = matrix.Matrix(self.__hidden_nodes, self.__input_nodes)
        self.__weights_ho = matrix.Matrix(self.__output_nodes, self.__hidden_nodes)
        self.__weights_ih.randomize()
        self.__weights_ho.randomize()

        # bias
        self.__bias_h = matrix.Matrix(self.__hidden_nodes, 1)
        self.__bias_o = matrix.Matrix(self.__output_nodes, 1)
        self.__bias_h.randomize()
        self.__bias_o.randomize()

        # Learning rate
        self.lr = 0.1

    def feed_forward(self, i):
        input_ = matrix.Matrix.from_array(i)
        hidden = matrix.Matrix.multiply(self.__weights_ih, input_)
        hidden.add(self.__bias_h)
        hidden.map(matrix.Matrix.sigmoid)
        output = matrix.Matrix.multiply(self.__weights_ho, hidden)
        output.add(self.__bias_o)
        output.map(matrix.Matrix.sigmoid)
        return matrix.Matrix.to_array(output)

    def train(self, i, target_, gr):
        # Feed-forward
        input_ = matrix.Matrix.from_array(i)
        hidden = matrix.Matrix.multiply(self.__weights_ih, input_)
        hidden.add(self.__bias_h)
        hidden.map(matrix.Matrix.sigmoid)
        output = matrix.Matrix.multiply(self.__weights_ho, hidden)
        output.add(self.__bias_o)
        output.map(matrix.Matrix.sigmoid)

        ###################################################################
        ######################### Backpropagation #########################
        ###################################################################

        # Calculate output errors
        target = matrix.Matrix.from_array(target_)
        output_errors = matrix.Matrix.subtract(target, output)

        # Calculate output gradient
        output_gradient = matrix.Matrix.map_(output, matrix.Matrix.d_sigmoid)
        output_gradient.multiply_(output_errors)
        output_gradient.multiply_(self.lr)

        ###################################################################

        # Calculate hidden --> output  weight deltas
        hidden_t = matrix.Matrix.transpose(hidden)
        weights_ho_deltas = matrix.Matrix.multiply(output_gradient, hidden_t)

        # Adjust hidden --> output weights
        self.__weights_ho.add(weights_ho_deltas)

        ###################################################################

        # Adjust output bias
        self.__bias_o.add(output_gradient)

        ###################################################################

        # Calculate hidden layer errors
        weights_ho_transpose = matrix.Matrix.transpose(self.__weights_ho)
        hidden_errors = matrix.Matrix.multiply(weights_ho_transpose, output_errors)
        # Calculate hidden gradient
        hidden_gradient = matrix.Matrix.map_(hidden, matrix.Matrix.d_sigmoid)
        hidden_gradient.multiply_(hidden_errors)
        hidden_gradient.multiply_(self.lr)

        # Calculate input --> hidden weight deltas
        input_t = matrix.Matrix.transpose(input_)
        weights_ih_deltas = matrix.Matrix.multiply(hidden_gradient, input_t)

        # Adjust output --> hidden weights
        self.__weights_ih.add(weights_ih_deltas)

        ###################################################################

        # Adjust hidden bias
        self.__bias_h.add(hidden_gradient)

        ###################################################################

        # Graphics
        gr.draw(input_, hidden, output, self.__weights_ih, self.__weights_ho)