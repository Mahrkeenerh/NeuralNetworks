#include "activations.h"

#include <cmath>

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float relu(float x) {
    return x > 0 ? x : 0;
}
