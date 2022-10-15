#include "BaseLayer.hpp"

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
