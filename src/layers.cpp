#include "layers.h"

#include <cmath>
#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size, float (*activation)(float)) {
    this->input_size = input_size;
    this->output_size = output_size;
    this->activation = activation;

    if (activation == sigmoid) {
        this->derivative = sigmoid_derivative;
    } else if (activation == relu) {
        this->derivative = relu_derivative;
    }

    this->weights =
        std::vector<std::vector<float>>(output_size, std::vector<float>(input_size + 1, 0.0));
    this->gradients =
        std::vector<std::vector<float>>(output_size, std::vector<float>(input_size + 1, 0.0));
    this->errors = std::vector<float>(output_size, 0.0);
    this->outputs = std::vector<float>(output_size, 0.0);

    // Initialize weights with random values between -1 and 1
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size + 1; j++) {
            this->weights[i][j] = (float)rand() / RAND_MAX * 2 - 1;
        }
    }
}

std::vector<float> DenseLayer::predict(std::vector<float> input) {
    // Calculate output for each neuron
    for (int i = 0; i < this->output_size; i++) {
        this->outputs[i] = this->weights[i][0];

        for (int j = 0; j < this->input_size; j++) {
            this->outputs[i] += this->weights[i][j + 1] * input[j];
        }

        this->outputs[i] = this->activation(this->outputs[i]);
    }

    return this->outputs;
}

float sigmoid(float x) { return 1 / (1 + exp(-x)); }

float sigmoid_derivative(float x) { return x * (1 - x); }

float relu(float x) { return x > 0 ? x : 0; }

float relu_derivative(float x) { return x > 0 ? 1 : 0; }
