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
    }

    this->weights =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    this->gradients =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    this->errors = std::vector<double>(output_size, 0.0);
    this->outputs = std::vector<double>(output_size, 0.0);

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size + 1; j++) {
            if (activation == sigmoid) {
                // Initialize weights with random values with uniform distribution
                // [-(1 / sqrt(input_size)), 1 / sqrt(input_size)]
                this->weights[i][j] =
                    (rand() / (double)RAND_MAX) * 2.0 / sqrt(input_size) - 1.0 / sqrt(input_size);
            } else if (activation == relu) {
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

        this->outputs[i] = this->activation(this->outputs[i]);
    }

    return this->outputs;
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
