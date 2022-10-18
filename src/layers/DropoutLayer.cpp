#include "DropoutLayer.hpp"

DropoutLayer::DropoutLayer(int output_size, double dropout_chance) {
    this->input_size = output_size;
    this->output_size = output_size;

    this->dropout_chance = dropout_chance;

    this->weights =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 1.0));
    this->gradients = std::vector<double>(output_size, 0.0);
    this->outputs = std::vector<double>(output_size, 0.0);
}

std::vector<double> DropoutLayer::forwardpropagate(std::vector<double> input) {
    // Calculate output for each neuron
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        if (rand() / (double)RAND_MAX > this->dropout_chance) {
            this->outputs[n_i] = input[n_i] * (1.0 / (1.0 - this->dropout_chance));
        } else {
            this->outputs[n_i] = 0.0;
        }
    }

    return this->outputs;
};
