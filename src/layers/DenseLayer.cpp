#include "DenseLayer.hpp"

#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size, double (*activation)(double)) {
    this->input_size = input_size;
    this->output_size = output_size;
    this->activation = activation;

    if (activation == sigmoid) {
        this->derivative = sigmoid_derivative;
    } else if (activation == relu) {
        this->derivative = relu_derivative;
    } else if (activation == leaky_relu) {
        this->derivative = leaky_relu_derivative;
    } else if (activation == swish) {
        this->derivative = swish_derivative;
    }

    this->weights =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    this->updates =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    this->gradients = std::vector<double>(output_size, 0.0);
    // this->batch_errors = std::vector<double>(output_size, 0.0);
    this->outputs = std::vector<double>(output_size, 0.0);

    // Momentum value
    this->beta1 = 0.3;
    this->weight_delta =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));

    // Adam settings
    /* this->momentum =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    this->variance =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    this->beta1 = 0.9;
    this->beta2 = 0.999;
    this->eta = 0.01;
    this->epsilon = 1e-8; */

    // Initialize weights
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size + 1; j++) {
            if (activation == sigmoid) {
                // Initialize weights with random values with uniform distribution
                // [-(1 / sqrt(input_size)), 1 / sqrt(input_size)]
                this->weights[i][j] =
                    (rand() / (double)RAND_MAX) * 2.0 / sqrt(input_size) - 1.0 / sqrt(input_size);
            } else {
                // He initialization with normal distribution
                this->weights[i][j] = randn() * sqrt(2.0 / input_size);
            }
        }
    }
}

std::vector<double> DenseLayer::predict(std::vector<double> input) {
    // #pragma omp parallel for
    // Calculate output for each neuron
    for (int n_i = 0; n_i < this->output_size; n_i += consts::MAT_MAX) {
        for (int n_j = 0; n_j < consts::MAT_MAX && n_i + n_j < this->output_size; n_j++) {
            this->outputs[n_i + n_j] = this->weights[n_i + n_j][0];
        }

        for (int i = 0; i < this->input_size; i++) {
            for (int n_j = 0; n_j < consts::MAT_MAX && n_i + n_j < this->output_size; n_j++) {
                this->outputs[n_i + n_j] += this->weights[n_i + n_j][i + 1] * input[i];
            }
        }
    }

    // #pragma omp parallel for
    // Apply activation function
    for (int i = 0; i < this->output_size; i++) {
        this->outputs[i] = this->activation(this->outputs[i]);
    }

    return this->outputs;
}

void DenseLayer::out_errors(std::vector<double> target_vector) {
    // Calculate errors - MSE
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->gradients[n_i] = this->outputs[n_i] - target_vector[n_i];
    }

    // Apply activation function
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->gradients[n_i] *= this->derivative(this->outputs[n_i]);
    }
}

void DenseLayer::backpropagate(Layer* connected_layer, std::vector<double> target_vector) {
    // #pragma omp parallel for
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->gradients[n_i] = 0;

        for (int o_i = 0; o_i < connected_layer->output_size; o_i++) {
            this->gradients[n_i] +=
                connected_layer->gradients[o_i] * connected_layer->weights[o_i][n_i + 1];
        }
    }

    // Apply activation function
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->gradients[n_i] *= this->derivative(this->outputs[n_i]);
    }
}

void DenseLayer::calculate_updates(std::vector<double> input, double learning_rate) {
    // #pragma omp parallel for
    double update;
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        update = this->gradients[0] * learning_rate + this->beta1 * this->weight_delta[n_i][0];
        this->updates[n_i][0] += update;

        for (int w_i = 1; w_i < this->input_size + 1; w_i++) {
            update = this->gradients[n_i] * learning_rate * input[w_i - 1] +
                     this->beta1 * this->weight_delta[n_i][w_i];
            this->updates[n_i][w_i] += update;
        }
    }

    // Adam
    /* double grad, alpha;
    #pragma omp parallel for
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        //#pragma omp parallel for
        for (int w_i = 0; w_i < this->input_size + 1; w_i++) {
            grad = this->errors[n_i];

            this->momentum[n_i][w_i] = this->beta1 * this->momentum[n_i][w_i] + (1 - this->beta1) * grad;
            this->variance[n_i][w_i] =
                this->beta2 * this->variance[n_i][w_i] + (1 - this->beta2) * pow(grad, 2);

            alpha = this->eta * sqrt((1 - pow(this->beta2, t + 1)) / (1 - pow(this->beta1, t + 1)));

            // Bias
            if (w_i == 0) {
                this->weights[n_i][w_i] -= learning_rate * alpha * this->momentum[n_i][w_i] /
                                           (sqrt(this->variance[n_i][w_i]) + this->epsilon);
            }
            // Weight
            else {
                this->weights[n_i][w_i] -= learning_rate * input[w_i - 1] * alpha *
                                           this->momentum[n_i][w_i] /
                                           (sqrt(this->variance[n_i][w_i]) + this->epsilon);
            }
        }
    } */
}

void DenseLayer::apply_updates(int minibatch_size) {
    // #pragma omp parallel for
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        for (int w_i = 0; w_i < this->input_size + 1; w_i++) {
            this->weights[n_i][w_i] -= this->updates[n_i][w_i];
            this->weight_delta[n_i][w_i] = this->updates[n_i][w_i] / minibatch_size;
        }
    }
}
