#include "SoftmaxLayer.hpp"

#include <algorithm>
#include <iostream>

SoftmaxLayer::SoftmaxLayer(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;

    this->weights =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 1.0));
    this->gradients = std::vector<double>(output_size, 0.0);

    // Momentum value
    this->beta1 = 0.2;
    this->weight_delta =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));

    // Initialize weights
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size + 1; j++) {
            // He initialization with normal distribution
            // this->weights[i][j] = randn() * sqrt(2.0 / input_size);
            // Initialize weights with random values with uniform distribution
            // [-(1 / sqrt(input_size)), 1 / sqrt(input_size)]
            this->weights[i][j] =
                (rand() / (double)RAND_MAX) * 2.0 / sqrt(input_size) - 1.0 / sqrt(input_size);
        }
    }

    // Adam settings
    // this->momentum =
    //     std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    // this->variance =
    //     std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    // this->beta1 = 0.1;
    // this->beta2 = 0.999;
    // this->eta = 0.01;
    // this->epsilon = 1e-8;
}

std::vector<double> SoftmaxLayer::predict(std::vector<double> input) {
    std::vector<double> output(this->output_size, 0.0);

    // Calculate output for each neuron
    double sum = 0;
    // double max = *std::max_element(std::begin(this->outputs), std::end(this->outputs));
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        output[n_i] = this->weights[n_i][0];

        for (int i = 0; i < this->input_size; i++) {
            output[n_i] += this->weights[n_i][i + 1] * input[i];
        }
        // sum += exp(this->outputs[n_i] - max);
        sum += exp(output[n_i]);
    }
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        // this->outputs[n_i] = exp(this->outputs[n_i] - max) / sum;
        output[n_i] = exp(output[n_i]) / sum;
    }

    return output;
}

void SoftmaxLayer::out_errors(std::vector<double> output, std::vector<double> target_vector) {
    // Derivative of cross entropy loss
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->gradients[n_i] = output[n_i] - target_vector[n_i];
    }
}

void SoftmaxLayer::calculate_updates(std::vector<std::vector<double>>* updates,
                                     std::vector<double> input, double learning_rate) {
    double update;
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        update = this->gradients[0] * learning_rate + this->beta1 * this->weight_delta[n_i][0];
        (*updates)[n_i][0] += update;

        for (int w_i = 1; w_i < this->input_size + 1; w_i++) {
            update = this->gradients[n_i] * learning_rate * input[w_i - 1] +
                     this->beta1 * this->weight_delta[n_i][w_i];
            (*updates)[n_i][w_i] += update;
        }
    }
}

void SoftmaxLayer::apply_updates(std::vector<std::vector<double>> updates, int minibatch_size) {
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        for (int w_i = 0; w_i < this->input_size + 1; w_i++) {
            this->weights[n_i][w_i] -= updates[n_i][w_i];
            this->weight_delta[n_i][w_i] = updates[n_i][w_i] / minibatch_size;
        }
    }
}
