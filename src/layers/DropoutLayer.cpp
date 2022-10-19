#include "DropoutLayer.hpp"

DropoutLayer::DropoutLayer(double dropout_chance) { this->dropout_chance = dropout_chance; }

DropoutLayer::DropoutLayer(int width, double dropout_chance) {
    this->input_size = width;
    this->output_size = width;

    this->dropout_chance = dropout_chance;

    this->weights = std::vector<std::vector<double>>(width, std::vector<double>(input_size + 1, 1.0));
}

void DropoutLayer::setup(int input_size) {
    this->input_size = input_size;
    this->output_size = input_size;

    this->weights =
        std::vector<std::vector<double>>(input_size, std::vector<double>(input_size + 1, 1.0));
}

std::vector<double> DropoutLayer::forwardpropagate(std::vector<double> input) {
    std::vector<double> output(this->output_size, 0.0);

    // Calculate output for each neuron
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        if (rand() / (double)RAND_MAX > this->dropout_chance) {
            output[n_i] = input[n_i] * (1.0 / (1.0 - this->dropout_chance));
        } else {
            output[n_i] = 0.0;
        }
    }

    return output;
};
