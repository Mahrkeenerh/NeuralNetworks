#include "layers.h"

#include <cmath>

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
    this->errors = std::vector<double>(output_size, 0.0);
    // this->batch_errors = std::vector<double>(output_size, 0.0);
    this->outputs = std::vector<double>(output_size, 0.0);

    // Initialize weights
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size + 1; j++) {
            if (activation == sigmoid) {
                // Initialize weights with random values with uniform distribution
                // [-(1 / sqrt(input_size)), 1 / sqrt(input_size)]
                this->weights[i][j] =
                    (rand() / (double)RAND_MAX) * 2.0 / sqrt(input_size) - 1.0 / sqrt(input_size);
            } else if (activation == relu || activation == leaky_relu || activation == swish) {
                // He initialization with normal distribution
                this->weights[i][j] = randn() * sqrt(2.0 / input_size);
            }
        }
    }
}

std::vector<double> DenseLayer::predict(std::vector<double> input) {
    // #pragma omp parallel for
    // Calculate output for each neuron
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->outputs[n_i] = this->weights[n_i][0];

        for (int i = 0; i < this->input_size; i++) {
            this->outputs[n_i] += this->weights[n_i][i + 1] * input[i];
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
        this->errors[n_i] = (this->outputs[n_i] - target_vector[n_i]);
    }

    // Apply activation function
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->errors[n_i] *= this->derivative(this->outputs[n_i]);
    }
}

void DenseLayer::backpropagate(Layer* connected_layer, std::vector<double> target_vector) {
    // #pragma omp parallel for
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->errors[n_i] = 0;

        for (int o_i = 0; o_i < connected_layer->output_size; o_i++) {
            this->errors[n_i] += connected_layer->errors[o_i] * connected_layer->weights[o_i][n_i + 1];
        }
    }

    // Apply activation function
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->errors[n_i] *= this->derivative(this->outputs[n_i]);
    }
}

void DenseLayer::update_weights(std::vector<double> input, double learning_rate) {
    // #pragma omp parallel for
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        for (int w_i = 1; w_i < this->input_size + 1; w_i++) {
            this->weights[n_i][w_i] -= this->errors[n_i] * learning_rate * input[w_i - 1];
        }
        this->weights[n_i][0] -= this->errors[n_i] * learning_rate;
    }
}

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
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->outputs[n_i] = input[n_i];
    }

    // Apply softmax preprocess
    double sum = 0;
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        sum += exp(this->outputs[n_i]);
    }

    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->outputs[n_i] = exp(this->outputs[n_i]) / sum;
    }

    // #pragma omp parallel for
    // Apply activation function
    for (int i = 0; i < this->output_size; i++) {
        this->outputs[i] = this->activation(this->outputs[i]);
    }

    return this->outputs;
}

void SoftmaxLayer::out_errors(std::vector<double> target_vector) {
    // Calculate errors - MSE
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->errors[n_i] = (this->outputs[n_i] - target_vector[n_i]);
    }

    // Softmax preprocess
    double sum = 0;
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        sum += exp(this->outputs[n_i]);
    }

    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->outputs[n_i] = exp(this->outputs[n_i]) / sum;
    }

    // Apply activation function
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->errors[n_i] *= this->derivative(this->outputs[n_i]);
    }
}

void SoftmaxLayer::backpropagate(Layer* connected_layer, std::vector<double> target_vector) {
    // #pragma omp parallel for
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->errors[n_i] = 0;

        for (int o_i = 0; o_i < connected_layer->output_size; o_i++) {
            this->errors[n_i] += connected_layer->errors[o_i] * connected_layer->weights[o_i][n_i + 1];
        }
    }

    // Softmax preprocess
    double sum = 0;
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        sum += exp(this->outputs[n_i]);
    }

    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->outputs[n_i] = exp(this->outputs[n_i]) / sum;
    }

    // Apply activation function
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->errors[n_i] *= this->derivative(this->outputs[n_i]);
    }
}

void SoftmaxLayer::update_weights(std::vector<double> input, double learning_rate) { return; }

// Random value from normal distribution using Box-Muller transform
double randn() {
    double u1 = rand() / (double)RAND_MAX;
    double u2 = rand() / (double)RAND_MAX;
    double out = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    // Avoid infinite values
    while (out == INFINITY || out == -INFINITY) {
        u1 = rand() / (double)RAND_MAX;
        u2 = rand() / (double)RAND_MAX;
        out = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    }

    return out;
}

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

double sigmoid_derivative(double x) { return x * (1 - x); }

double relu(double x) { return x > 0 ? x : 0; }

double relu_derivative(double x) { return x > 0 ? 1 : 0; }

double leaky_relu(double x) { return x > 0 ? x : 0.001 * x; }

double leaky_relu_derivative(double x) { return x > 0 ? 1 : 0.001; }

double swish(double x) { return x / (1 + exp(-x)); }

double swish_derivative(double x) { return (1 + exp(-x) + x * exp(-x)) / pow(1 + exp(-x), 2); }

double softmax(double x) { return x; }

double softmax_derivative(double x) { return x * (1 - x); }
