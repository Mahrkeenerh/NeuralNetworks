#ifndef networks
#define networks

#include <vector>

#include "layers.h"

class DenseNetwork {
   public:
    DenseNetwork(std::vector<int> layer_sizes);

    std::vector<float> predict(std::vector<float> input);

   private:
    std::vector<int> layer_sizes;
    std::vector<DenseLayer> layers;
};

float mse(std::vector<float> output, std::vector<float> target);

#endif
