#ifndef layers_h
#define layers_h

#include <vector>

class DenseLayer {
   public:
    int input_size, output_size;

    DenseLayer(int input_size, int output_size, float (*activation)(float));

    std::vector<float> predict(std::vector<float> input);

    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    std::vector<std::vector<float>> gradients_w;
    std::vector<float> gradients_b;
    float (*activation)(float);
};

float sigmoid(float x);

float relu(float x);

#endif
