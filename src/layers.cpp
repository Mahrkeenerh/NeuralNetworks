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
    } else if (activation == softmax) {
        this->derivative = softmax_derivative;
    }

    this->weights =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    this->gradients =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    this->errors = std::vector<double>(output_size, 0.0);
    this->outputs = std::vector<double>(output_size, 0.0);

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size + 1; j++) {
            if (activation == sigmoid || activation == softmax) {
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
    // Calculate output for each neuron
    for (int i = 0; i < this->output_size; i++) {
        this->outputs[i] = this->weights[i][0];

        for (int j = 0; j < this->input_size; j++) {
            this->outputs[i] += this->weights[i][j + 1] * input[j];
        }
    }

    // Apply softmax preprocess
    if (this->activation == softmax) {
        double sum = 0;
        for (int i = 0; i < outputs.size(); i++) {
            sum += exp(outputs[i]);
        }

        for (int i = 0; i < outputs.size(); i++) {
            this->outputs[i] = exp(this->outputs[i]) / sum;
        }
    }

    // Apply activation function
    for (int i = 0; i < this->output_size; i++) {
        this->outputs[i] = this->activation(this->outputs[i]);
    }

    return this->outputs;
}

void DenseLayer::backpropagate(DenseLayer* connected_layer, std::vector<double> outputs,
                               std::vector<double> target_vector, bool last_layer) {
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->errors[n_i] = 0;
        if (last_layer) {
            this->errors[n_i] = (outputs[n_i] - target_vector[n_i]);
        } else {
            for (int o_i = 0; o_i < connected_layer->output_size; o_i++) {
                this->errors[n_i] +=
                    connected_layer->errors[o_i] * connected_layer->weights[o_i][n_i + 1];
            }
        }

        if (this->derivative == softmax_derivative) {
            double sum = 0;
            for (int idx = 0; idx < outputs.size(); idx++) {
                sum += exp(outputs[idx]);
            }
            this->outputs[n_i] = exp(this->outputs[n_i]) / sum;
        }
        this->errors[n_i] *= this->derivative(this->outputs[n_i]);
    }
}

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
