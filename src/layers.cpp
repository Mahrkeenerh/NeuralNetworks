#include "layers.h"

#include <iostream>
#include <vector>

DenseLayer::DenseLayer(int input_size, int output_size, float (*activation)(float)) {
    this->input_size = input_size;
    this->output_size = output_size;
    this->activation = activation;

    this->weights = std::vector<std::vector<float>>(output_size, std::vector<float>(input_size, 0.0));
    this->biases = std::vector<float>(output_size, 0.0);

    // Initialize weights and biases with random values between -1 and 1
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            this->weights[i][j] = (float)rand() / RAND_MAX * 2 - 1;
        }
        this->biases[i] = (float)rand() / RAND_MAX * 2 - 1;
    }
}

std::vector<float> DenseLayer::predict(std::vector<float> input) {
    std::vector<float> output(this->output_size, 0.0);

    // Calculate output for each neuron
    for (int i = 0; i < this->output_size; i++) {
        for (int j = 0; j < this->input_size; j++) {
            output[i] += this->weights[i][j] * input[j];
        }
        output[i] += this->biases[i];
    }

    // Apply activation function
    for (int i = 0; i < this->output_size; i++) {
        output[i] = this->activation(output[i]);
    }

    return output;
}
