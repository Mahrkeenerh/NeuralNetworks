#include "DropoutLayer.h"

DropoutLayer::DropoutLayer(int output_size, float dropout_chance) {
    this->input_size = output_size;
    this->output_size = output_size;

    this->dropout_chance = dropout_chance;

    this->weights =
        std::vector<std::vector<float>>(output_size, std::vector<float>(input_size + 1, 1.0));
    this->errors = std::vector<float>(output_size, 0.0);
    this->outputs = std::vector<float>(output_size, 0.0);
}

std::vector<float> DropoutLayer::forwardpropagate(std::vector<float> input) {
    // Calculate output for each neuron
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        if (rand() / (float)RAND_MAX > this->dropout_chance) {
            this->outputs[n_i] = input[n_i] * (1.0 / (1.0 - this->dropout_chance));
        } else {
            this->outputs[n_i] = 0.0;
        }
    }

    return this->outputs;
};
