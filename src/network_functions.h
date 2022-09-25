#ifndef network_functions_h
#define network_functions_h

#include <vector>

float sigmoid(float x);

float relu(float x);

float mse(std::vector<float> output, std::vector<float> target);

#endif
