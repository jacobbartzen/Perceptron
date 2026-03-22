## Overview

The main file is perceptron.c. perceptronRunners.c is the same code, but with data from runners inputted as a test. As of now, it just contains one neuron. The user is easily able to adjust:
- All input data
- Number of epochs
- Prints
- Learning Rate

The code is commented and gives a basic overview of how a perceptron learns using weights and adjustments for data. 

## How a neuron works

In a neural network, a neuron basically a function that weights data. The basic steps are:

1. Weight all inputs and add a bias
   - y = w1x1 + w2x2... + wnxn + b = ∑wx + b

2. Apply a nonlinearization / activation function such as the sigmoid function. This is currently not included in my code since it only contains one neuron, but may be added in the future

## Training

A neuron learngs through adjusting weights to account for error. The most common method is gradient disent, where the derivative of how the error changes based on changing weights is calculated. With many layers of neurons, more complex algorithms such as backpropogation are needed but with only one neuron it is much simpler.

Weights are increased if the output was too high, and decreased if the output was too low proportionally to the input size and learning rate.

  weights[j] += LEARNING_RATE * eTotal * inputs[i][j];
