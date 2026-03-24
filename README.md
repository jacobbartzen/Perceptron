## Perceptron.c

The main file is perceptron.c. perceptronRunners.c uses the same implementation with runner data as a test dataset. The project currently implements a single neuron. The user can adjust:
- All input data
- Number of epochs
- Print settings
- Learning Rate

The code is commented and gives a basic overview of how a perceptron learns using weights and adjustments for data. 

## How a neuron works

In a neural network, a neuron basically a function that weights data. The basic steps are:

1. Weight all inputs and add a bias
   - y = w1x1 + w2x2... + wnxn + b = ∑wx + b

2. This value is then passed through an activation function. perceptron.c uses the reLU activation function, which simply changes negative values to 0

## Training

A neuron learns by adjusting its weights to reduce error. In a simple model, this can be done by updating each weight proportionally to the input value, the error, and the learning rate. For multi-layer neural networks, more advanced methods such as backpropagation are used.

In my example, weights are increased if the output was too high, and decreased if the output was too low proportionally to the input size and learning rate.

  weights[j] += LEARNING_RATE * eTotal * inputs[i][j];

## neuralNetwork.c

This will start to build a more complete neural network with many neurons and layers.