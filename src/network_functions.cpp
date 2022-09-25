#include "network_functions.h"

#include <cmath>

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float relu(float x) {
    return x > 0 ? x : 0;
}

float mse(std::vector<float> output, std::vector<float> target) {
    float error = 0.0;

    for (int i = 0; i < output.size(); i++) {
        error += pow(output[i] - target[i], 2);
    }

    return error / output.size();
}
