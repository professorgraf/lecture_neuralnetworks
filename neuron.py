##
# (c) 2024 Prof. Dr. sc. hum. Markus Graf
#
# base classes for neural network simulating app
#

import random
import math


class Activation:
    def __init__(self):
        pass

    def calculate(self, value):
        return value


class Heaviside(Activation):
    def __init__(self):
        super().__init__()

    def calculate(self, value):
        if value > 0.0:
            return 1.0
        else:
            return 0.0


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def calculate(self, value):
        return max(0, value)


class WeakReLU(ReLU):
    def __init__(self, alpha=0.25):
        super().__init__()
        self.alpha = alpha

    def calculate(self, value):
        return max(self.alpha*value, value)


class TanH(Activation):
    def __init__(self):
        super().__init__()

    def calculate(self, value):
        return math.tanh(value)


class Neuron:
    def __init__(self, activation_class=None):
        if activation_class is None:
            activation_class = ReLU
        self.in_neurons = []
        self.out_neurons = []
        self.weights = []
        self.activation = activation_class()
        self.v = 0.0
        self.invalid = True
        self.item = None
        self.text_item = None

    def link_to(self, neuron):
        self.out_neurons.append(neuron)

        neuron.in_neurons.append(self)
        neuron.weights.append((random.random() * 2 - 1)/10)

        # would be linking this neuron to next neuron (given as parameter)
        #neuron.out_neurons.append(self)
        #self.in_neurons.append(neuron)
        #self.weights.append(random.random() * 2 - 1)

    def invalidate(self):
        self.invalid = True

    def backpropagation_error(self, adaption_value, use_pseudo_gradient=False):
        for j in range(0, len(self.in_neurons)):
            val = self.in_neurons[j].value()
            if val != 0.0:
                if use_pseudo_gradient:
                    self.weights[j] += adaption_value * val # * math.fabs(val-self.value()+0.00001)
                else:
                    self.weights[j] += adaption_value
            self.in_neurons[j].backpropagation_error(adaption_value, use_pseudo_gradient)

    def value(self):
        if self.invalid:
            self.v = 0.0
            for i in range(0, len(self.in_neurons)):
                self.v += self.in_neurons[i].value() * self.weights[i]
            self.invalid = False

        return self.activation.calculate(self.v)

    def is_active(self):
        return self.value() > 0.1


class InputNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.v = 0.0

    def set(self, value):
        self.v = value

    def backpropagation_error(self, adaption_value, use_pseudo_gradient=False):
        # do nothing
        pass

    def value(self):
        return self.v


class NeuralNetwork:
    # params in form: int, int, 1d list with number of neurons within the layers (besides the input)
    # example: nn = NeuralNetwork(3, [4, 2])
    #          --> creates fully connected nn with 3 inputs and 2 output neurons
    #          --> adds another "hidden" layer with 4 elements
    #          alpha is the regularization parameter
    #          epsilon is the learning rate
    def __init__(self, no_inputs, layer_specs=None, alpha=0.00001, epsilon=0.2, activation_class=Heaviside):
        if layer_specs is None:
            layer_specs = [1]
        self.n_in = no_inputs
        self.n_out = layer_specs[len(layer_specs)-1]
        self.n_layers = 1 + len(layer_specs)
        self.alpha = alpha
        self.epsilon = epsilon

        self.layers = []
        for j in range(0, self.n_layers):
            layer = []
            if j == 0:
                for i in range(0, no_inputs):
                    layer.append(InputNeuron())
            else:
                for i in range(0, layer_specs[j-1]):
                    n = Neuron(activation_class=activation_class)
                    layer.append(n)
                    for k in range(0, len(self.layers[j-1])):
                        self.layers[j-1][k].link_to(n)
                        # link from all neurons from previous layer to this neuron

            self.layers.append(layer)

    def train_on(self, inputs, labels):
        if len(inputs) != len(self.layers[0]):
            print("[ERROR] //please raise it as exception// number of inputs != number of input neurons error!")
        if len(labels) != len(self.layers[len(self.layers)-1]):
            print("[ERROR] //please raise it as exception// number of labels != number of outputs!")

        results = self.feed_forward(inputs)
        diffs = results.copy()

        error = 0.0
        # do backpropagation

        sum_weight = self.get_squared_weight_sum()      # for regularization
        sum_weight = 0

        for j in range(0, len(results)):
            diffs[j] = labels[j] - results[j]
            if diffs[j] < 0:
                diffs_regularized = diffs[j] - self.alpha * sum_weight
            else:
                diffs_regularized = diffs[j] + self.alpha * sum_weight
            self.layers[len(self.layers)-1][j].backpropagation_error(self.epsilon * diffs_regularized,
                                                                     len(self.layers)>2 )
            error += math.fabs(diffs[j])
        return error

    def feed_forward(self, inputs):
        if len(inputs) != len(self.layers[0]):
            print("[ERROR] //please raise it as exception// number of inputs != number of input neurons error!")

        for i in range(0, len(self.layers[0])):
            self.layers[0][i].set(inputs[i])

        for j in range(1, len(self.layers)):
            for i in range(0, len(self.layers[j])):
                self.layers[j][i].invalidate()

        # for j in range(0, len(self.layers)):
        #    for i in range(0, len(self.layers[j])):
        #        n =
        # somehow it is made recursive by calling value of a neuron
        # so just iterate through last layers and get values
        results = []
        last_layer = len(self.layers)-1
        for i in range(0, len(self.layers[last_layer])):
            results.append(self.layers[last_layer][i].value())
        return results

    def get_squared_weight_sum(self):
        sum = 0.0
        for k in range(1, len(self.layers)):
            for j in range(0, len(self.layers[k])):
                for i in range(0, len(self.layers[k][j].weights)):
                    sum += self.layers[k][j].weights[i]**2
        return sum

class PerceptronOrNetwork(NeuralNetwork):
    def __init__(self, no_inputs=2):
        super().__init__(no_inputs)
        for neuron in self.layers[1]:
            for j in range(0, len(neuron.weights)):
                neuron.weights[j] = 0.4
