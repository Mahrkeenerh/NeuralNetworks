#include "networks.h"

#include <cmath>

#include "layers.h"

DenseNetwork::DenseNetwork(std::vector<int> layer_sizes) {
    this->layer_sizes = layer_sizes;

    // Create layers
    for (int i = 0; i < layer_sizes.size() - 2; i++) {
        this->layers.push_back(
            DenseLayer(
                layer_sizes[i],
                layer_sizes[i + 1],
                relu));
    }

    // Create output layer
    this->layers.push_back(
        DenseLayer(
            layer_sizes[layer_sizes.size() - 2],
            layer_sizes[layer_sizes.size() - 1],
            sigmoid));
}

std::vector<float> DenseNetwork::predict(std::vector<float> input) {
    std::vector<float> output = input;

    for (int i = 0; i < this->layers.size(); i++) {
        output = this->layers[i].predict(output);
    }

    return output;
}

float mse(std::vector<float> output, std::vector<float> target) {
    float error = 0.0;

    for (int i = 0; i < output.size(); i++) {
        error += pow(output[i] - target[i], 2);
    }

    return error / output.size();
}
