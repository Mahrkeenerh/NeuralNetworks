#include "layers.h"

#include <cmath>
#include <iostream>

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

    // Initialize weights with random values between -1 and 1
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size + 1; j++) {
            if (activation == sigmoid) {
                // Initialize weights with random values between -1 and 1
                this->weights[i][j] = (double)rand() / RAND_MAX * 2 - 1;
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

double randn() {
    double u = (double)rand() / RAND_MAX;
    double v = (double)rand() / RAND_MAX;
    return sqrt(-2 * log(u)) * cos(2 * M_PI * v);
}

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

double sigmoid_derivative(double x) { return x * (1 - x); }

double relu(double x) { return x > 0 ? x : 0; }

double relu_derivative(double x) { return x > 0 ? 1 : 0; }
