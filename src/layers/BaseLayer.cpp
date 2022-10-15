#include "BaseLayer.h"

// Random value from normal distribution using Box-Muller transform
float randn() {
    float u1 = rand() / (float)RAND_MAX;
    float u2 = rand() / (float)RAND_MAX;
    float out = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    // Avoid infinite values
    while (out == INFINITY || out == -INFINITY) {
        u1 = rand() / (float)RAND_MAX;
        u2 = rand() / (float)RAND_MAX;
        out = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    }

    return out;
}

float sigmoid(float x) { return 1 / (1 + exp(-x)); }

float sigmoid_derivative(float x) { return x * (1 - x); }

float relu(float x) { return x > 0 ? x : 0; }

float relu_derivative(float x) { return x > 0 ? 1 : 0; }

float leaky_relu(float x) { return x > 0 ? x : 0.001 * x; }

float leaky_relu_derivative(float x) { return x > 0 ? 1 : 0.001; }

float swish(float x) { return x / (1 + exp(-x)); }

float swish_derivative(float x) { return (1 + exp(-x) + x * exp(-x)) / pow(1 + exp(-x), 2); }

float softmax(float x) { return x; }

float softmax_derivative(float x) { return x * (1 - x); }
