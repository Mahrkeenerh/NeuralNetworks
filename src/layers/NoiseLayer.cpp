#include "NoiseLayer.hpp"

NoiseLayer::NoiseLayer(int output_size, double noise_chance, double noise_scale) {
    this->input_size = output_size;
    this->output_size = output_size;

    this->noise_chance = noise_chance;
    this->noise_scale = noise_scale;

    this->weights =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 1.0));
    this->errors = std::vector<double>(output_size, 0.0);
    this->outputs = std::vector<double>(output_size, 0.0);
}

std::vector<double> NoiseLayer::forwardpropagate(std::vector<double> input) {
    // Calculate output for each neuron
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->outputs[n_i] = input[n_i];
        if (rand() / (double)RAND_MAX > this->noise_chance) {
            this->outputs[n_i] += (randn() * this->noise_scale);
        }
    }

    return this->outputs;
};
