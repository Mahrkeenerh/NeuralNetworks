#include "networks.h"

#include <cmath>
#include <iostream>

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

void DenseNetwork::fit(
    std::vector<std::vector<float>> inputs,
    std::vector<int> targets,
    int epochs,
    float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float error = this->error(inputs, targets);
        std::cout << "Epoch " << epoch << ": " << error << std::endl;

        for (int i = 0; i < this->layers.size(); i++) {
            // Calculate weight error gradients
            for (int n_i = 0; n_i < this->layers[i].output_size; n_i++) {
                for (int w_i = 0; w_i < this->layers[i].input_size + 1; w_i++) {
                    this->layers[i].weights[n_i][w_i] += learning_rate;

                    float error_after = this->error(inputs, targets);
                    float delta_error = error_after - error;

                    this->layers[i].weights[n_i][w_i] -= learning_rate;
                    this->layers[i].gradients_w[n_i][w_i] = delta_error / learning_rate;
                }
            }
        }

        // Update weights
        for (int i = 0; i < this->layers.size(); i++) {
            for (int n_i = 0; n_i < this->layers[i].output_size; n_i++) {
                for (int w_i = 0; w_i < this->layers[i].input_size + 1; w_i++) {
                    this->layers[i].weights[n_i][w_i] -= this->layers[i].gradients_w[n_i][w_i] * learning_rate;
                }
            }
        }
    }
}

float DenseNetwork::error(std::vector<std::vector<float>> inputs, std::vector<int> targets) {
    float error = 0;

    for (int i = 0; i < inputs.size(); i++) {
        std::vector<float> target_vector(this->layer_sizes[this->layer_sizes.size() - 1], 0);
        target_vector[targets[i]] = 1;

        error += mse(this->predict(inputs[i]), target_vector);
    }

    return error / inputs.size();
}

float mse(float output, float target) {
    return pow(output - target, 2);
}

float mse(std::vector<float> output, std::vector<float> target) {
    float error = 0.0;

    for (int i = 0; i < output.size(); i++) {
        error += mse(output[i], target[i]);
    }

    return error / output.size();
}
