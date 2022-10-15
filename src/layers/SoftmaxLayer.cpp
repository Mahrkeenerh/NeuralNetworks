#include "SoftmaxLayer.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

SoftmaxLayer::SoftmaxLayer(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;

    this->weights =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 1.0));
    this->errors = std::vector<double>(output_size, 0.0);
    this->outputs = std::vector<double>(output_size, 0.0);

    // Momentum value
    this->beta1 = 0.2;
    this->weight_delta =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));

    // Initialize weights
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size + 1; j++) {
            this->weights[i][j] = randn() * sqrt(2.0 / input_size);
        }
    }

    // Adam settings
    this->momentum =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    this->variance =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    this->beta1 = 0.1;
    this->beta2 = 0.999;
    this->eta = 0.01;
    this->epsilon = 1e-8;
}

std::vector<double> SoftmaxLayer::predict(std::vector<double> input) {
    // #pragma omp parallel for
    // Calculate output for each neuron
    double sum = 0;
    double max = *std::max_element(std::begin(this->outputs), std::end(this->outputs));
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->outputs[n_i] = this->weights[n_i][0];

        for (int i = 0; i < this->input_size; i++) {
            if (! std::abs(std::isnan(this->outputs[n_i] + this->weights[n_i][i + 1] * input[i]))) {
                this->outputs[n_i] += this->weights[n_i][i + 1] * input[i];
            }

            //std::cout << std::endl << this->outputs[n_i] << std::isnan(this->outputs[n_i]);
            
        }
        sum += exp(this->outputs[n_i] - max);
        //std::cout << std::endl << sum;
    }
    //std::cout << std::endl << sum;
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->outputs[n_i] = exp(this->outputs[n_i] - max) / sum;
        //std::cout << std::endl << this->outputs[n_i];
    }

    return this->outputs;
}

void SoftmaxLayer::out_errors(std::vector<double> target_vector) {
    // Calculate errors - MSE and apply activation function
    // for (int n_i = 0; n_i < this->output_size; n_i++) {
    //    this->errors[n_i] = (this->outputs[n_i] - target_vector[n_i]) *
    //    this->derivative(this->outputs[n_i]);
    // }

    // Derivative of cross entropy loss
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->errors[n_i] = this->outputs[n_i] - target_vector[n_i];
    }
}

void SoftmaxLayer::backpropagate(Layer* connected_layer, std::vector<double> target_vector) {
    // #pragma omp parallel for
    return;
}

void SoftmaxLayer::update_weights(std::vector<double> input, double learning_rate, int t) {
    // #pragma omp parallel for
    double update;
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        update = this->errors[0] * learning_rate + this->beta1 * this->weight_delta[n_i][0];
        this->weights[n_i][0] -= update;
        this->weight_delta[n_i][0] = update;

        for (int w_i = 1; w_i < this->input_size + 1; w_i++) {
            update = this->errors[n_i] * learning_rate * input[w_i - 1] +
                     this->beta1 * this->weight_delta[n_i][w_i];
            this->weights[n_i][w_i] -= update;
            this->weight_delta[n_i][w_i] = update;
        }
    }
}
