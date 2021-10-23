'''
Created on Aug 15, 2021

@author: Syed.Ausaf.Hussain
'''

from math import exp
from random import random
import pandas
import numpy as np


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:       # for hidden layer
            for j in range(len(layer)):
                error = 0.0
                neuronTemp = network[i + 1][0]
                error += (neuron['weights'][j] * neuronTemp['error'])
                errors.append(error)
        else:                           # for output layer
            neuronTemp = layer[0]
            errors.append(expected - neuronTemp['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['error'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:  # for output layer hidden layer output become input
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['error'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['error']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = row[-1]
            print(expected, outputs)
            sum_error += (expected - outputs[0]) ** 2
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
            print("updated weights", network)
        print('epoch=%d' % (epoch))
    return network


l_rate = 0.3
n_epoch = 1000
n_hidden = 2
#2 input and 1 output
dataset = [[0, 0, 0],
           [0, 1, 1],
           [1, 0, 1],
           [1, 1, 0]]
network = [[{'weights': [0.2, 0.4, -0.4]},  #hidden layer neuron 3
            {'weights': [-0.3, 0.1, 0.2]}], #hidden layer neuron 4
           [{'weights': [-0.3, -0.2, 0.1]}]] #output layer neuron 5

n_inputs = len(dataset[0]) - 1
print(train_network(network, dataset, l_rate, n_epoch))