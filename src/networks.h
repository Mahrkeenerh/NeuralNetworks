#ifndef networks
#define networks

#include <vector>

#include "layers.h"

class DenseNetwork {
   public:
    DenseNetwork(std::vector<int> layer_sizes);

    std::vector<float> predict(std::vector<float> input);
    void fit(std::vector<std::vector<float>> inputs, std::vector<int> targets, int epochs,
             float learning_rate);

    float error(std::vector<std::vector<float>> inputs, std::vector<int> targets);
    std::vector<float> output_errors(std::vector<float> inputs, int target, float (*derivative)(float));

   private:
    std::vector<int> layer_sizes;
    std::vector<DenseLayer> layers;
};

float mse(float output, float target);
float mse(std::vector<float> outputs, std::vector<float> target);

#endif
