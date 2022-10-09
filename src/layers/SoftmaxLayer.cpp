#include "SoftmaxLayer.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

SoftmaxLayer::SoftmaxLayer(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;

    this->activation = softmax;
    this->derivative = softmax_derivative;

    this->weights =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 1.0));
    this->errors = std::vector<double>(output_size, 0.0);
    this->outputs = std::vector<double>(output_size, 0.0);

    // Initialize weights
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size + 1; j++) {
            this->weights[i][j] = randn() * sqrt(2.0 / input_size);
        }
    }
}

std::vector<double> SoftmaxLayer::predict(std::vector<double> input) {
    // #pragma omp parallel for
    // Calculate output for each neuron
    double sum = 0;
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->outputs[n_i] = this->weights[n_i][0];

        for (int i = 0; i < this->input_size; i++) {
            this->outputs[n_i] += this->weights[n_i][i + 1] * input[i];
        }
        sum += exp(this->outputs[n_i]);
    }

    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->outputs[n_i] = exp(this->outputs[n_i] - 0.0001) / sum;
    }

    return this->outputs;
}

void SoftmaxLayer::out_errors(std::vector<double> target_vector) {
    // Calculate errors - MSE and apply activation function
    // for (int n_i = 0; n_i < this->output_size; n_i++) {
    //    this->errors[n_i] = (this->outputs[n_i] - target_vector[n_i]) * this->derivative(this->outputs[n_i]);
    // }

    // Derivative of cross entropy loss
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->errors[n_i] = this->outputs[n_i];
        if (target_vector[n_i] == 1) {
            this->errors[n_i] -= 1;
        }
    }
}

void SoftmaxLayer::backpropagate(Layer* connected_layer, std::vector<double> target_vector) {
    // #pragma omp parallel for
    return;
}

void SoftmaxLayer::update_weights(std::vector<double> input, double learning_rate) {
    // #pragma omp parallel for
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        for (int w_i = 1; w_i < this->input_size + 1; w_i++) {
            this->weights[n_i][w_i] -= this->errors[n_i] * learning_rate * input[w_i - 1];
        }
        this->weights[n_i][0] -= this->errors[n_i] * learning_rate;
    //}
}
