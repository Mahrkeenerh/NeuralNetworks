#include "layers.h"

#include <iostream>
#include <vector>

DenseLayer::DenseLayer(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;

    this->weights = std::vector<std::vector<double>>(output_size, std::vector<double>(input_size, 0.0));
    this->biases = std::vector<double>(output_size, 0.0);

    // Initialize weights and biases with random values between -1 and 1
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            this->weights[i][j] = (double)rand() / RAND_MAX * 2 - 1;
        }
        this->biases[i] = (double)rand() / RAND_MAX * 2 - 1;
    }
}

std::vector<double> DenseLayer::predict(std::vector<double> input) {
    std::vector<double> output(this->output_size, 0.0);

    for (int i = 0; i < this->output_size; i++) {
        for (int j = 0; j < this->input_size; j++) {
            output[i] += this->weights[i][j] * input[j];
        }
        output[i] += this->biases[i];
    }

    // TODO apply activation function
    return output;
}
