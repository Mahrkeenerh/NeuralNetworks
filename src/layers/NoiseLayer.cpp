#include "NoiseLayer.h"

NoiseLayer::NoiseLayer(int output_size, float noise_chance, float noise_scale) {
    this->input_size = output_size;
    this->output_size = output_size;

    this->noise_chance = noise_chance;
    this->noise_scale = noise_scale;

    this->weights =
        std::vector<std::vector<float>>(output_size, std::vector<float>(input_size + 1, 1.0));
    this->errors = std::vector<float>(output_size, 0.0);
    this->outputs = std::vector<float>(output_size, 0.0);
}

std::vector<float> NoiseLayer::forwardpropagate(std::vector<float> input) {
    // Calculate output for each neuron
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        if (rand() / (float)RAND_MAX > this->noise_chance) {
            this->outputs[n_i] = input[n_i] + (randn() * this->noise_scale);
        } else {
            this->outputs[n_i] = 0.0;
        }
    }

    return this->outputs;
};
