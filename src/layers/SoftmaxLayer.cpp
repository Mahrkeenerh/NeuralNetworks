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
}

std::vector<double> SoftmaxLayer::predict(std::vector<double> input) {
    // #pragma omp parallel for
    // Calculate output for each neuron
    double sum = 0;
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->outputs[n_i] = input[n_i];
        sum += exp(this->outputs[n_i]);
    }

    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->outputs[n_i] = exp(this->outputs[n_i] - *std::max_element(std::begin(this->outputs), std::end(this->outputs))) / sum;
    }

    return this->outputs;
}

void SoftmaxLayer::out_errors(std::vector<double> target_vector) {
    // Calculate errors - MSE
     for (int n_i = 0; n_i < this->output_size; n_i++) {
         this->errors[n_i] = (this->outputs[n_i] - target_vector[n_i]);
     }

    // Calculate errors - Cross entropy
    //for (int n_i = 0; n_i < this->output_size; n_i++) {
    //    this->errors[n_i] = - target_vector[n_i] * log(this->outputs[n_i]);
    //}

    // Apply activation function
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->errors[n_i] *= this->derivative(this->outputs[n_i]);
    }
}

void SoftmaxLayer::backpropagate(Layer* connected_layer, std::vector<double> target_vector) {
    // #pragma omp parallel for
    return;
}

void SoftmaxLayer::update_weights(std::vector<double> input, double learning_rate) {
    // #pragma omp parallel for
    return;
}
