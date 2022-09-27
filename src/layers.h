#ifndef layers_h
#define layers_h

#include <vector>

class DenseLayer {
   public:
    int input_size, output_size;

    DenseLayer(int input_size, int output_size, float (*activation)(float));

    std::vector<float> predict(std::vector<float> input);

    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> gradients;
    std::vector<float> errors;
    std::vector<float> outputs;

    float (*activation)(float);
    float (*derivative)(float);
};

float sigmoid(float x);
float sigmoid_derivative(float x);

float relu(float x);
float relu_derivative(float x);

#endif
