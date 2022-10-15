#ifndef networks
#define networks

#include <vector>

#include "datasets.h"
#include "layers/layers.h"

class DenseNetwork {
   public:
    DenseNetwork(std::vector<int> layer_sizes);

    std::vector<float> predict(std::vector<float> input);
    std::vector<float> forwardpropagate(std::vector<float> input);
    void backpropagate(std::vector<float> target_vector);
    void update_weights(std::vector<float> input_data, float learning_rate, int epoch);
    void fit(Dataset1D dataset, int epochs, float learning_rate, bool verbose);
    // void fit(Dataset1D dataset, int epochs, float learning_rate, int batch_size, bool verbose);

    float accuracy(std::vector<std::vector<float>> inputs, std::vector<int> targets);

   private:
    std::vector<int> layer_sizes;
    std::vector<Layer*> layers;
};

float mse(float output, float target);
float mse(std::vector<float> outputs, std::vector<float> target);

#endif
