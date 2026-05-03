## Machine Learning from Scratch

This project features two main code files, both coded in C: perceptron.c and neuralNetwork.c. neuralNetwork builds on perceptron.c by adding multiple layers, while perceptron.c is a single neuron.

## perceptron.c

This file has one perceptron. It is able to fit one linear line to the data provided. The user is able to adjust settings such as the learning rate, epochs, all data, and features such as early stopping and data normalization. The code tracks runtime, error, and prints all weights

## How a neuron works

In a neural network, a neuron basically a function that weights data. The basic steps are:

1. Weight all inputs and add a bias
   - y = w1x1 + w2x2... + wnxn + b = ∑wx + b

2. This value is then passed through an activation function. perceptron.c uses the reLU activation function, which simply changes negative values to 0

## Training

A neuron learns by adjusting its weights to reduce error. In a simple model, this can be done by updating each weight proportionally to the input value, the error, and the learning rate. For multi-layer neural networks, more advanced methods such as backpropagation are used.

In my example, weights are increased if the output was too high, and decreased if the output was too low proportionally to the input size and learning rate.

  weights[j] += LEARNING_RATE * eTotal * inputs[i][j];

## Neuron vs. Perceptron

While the file is called perceptron.c, there is one main difference between a perceptron and the neuron coded: A perceptron either outputs 1 or 0 (or sometimes 1 / -1). A perceptron was one of the very early approaches to machine learning. It aimed to mimic a human brain, where neurons either fire or don't fire - binary states. In my file perceptron.c, the neuron outputs a range of values.

## neuralNetwork.c

neuralNetwork.c builds on perceptron.c by adding multiple layers of neurons - currently the user is able to adjust the amount of neurons in layer 2, as well as all features included in perceptron.c.
