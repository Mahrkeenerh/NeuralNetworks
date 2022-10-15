#ifndef networks
#define networks

#include <vector>

#include "datasets.hpp"
#include "layers/layers.hpp"

class DenseNetwork {
   public:
    DenseNetwork(std::vector<int> layer_sizes);

    std::vector<double> predict(std::vector<double> input);
    std::vector<double> forwardpropagate(std::vector<double> input);
    void backpropagate(std::vector<double> target_vector);
    void update_weights(std::vector<double> input_data, double learning_rate);
    void fit(Dataset1D dataset, int epochs, double learning_rate_start, double learning_rate_end,
             bool verbose = true);
    // void fit(Dataset1D dataset, int epochs, double learning_rate, int batch_size, bool verbose);

    double accuracy(std::vector<std::vector<double>> inputs, std::vector<int> targets);

   private:
    std::vector<int> layer_sizes;
    std::vector<Layer*> layers;
};

double mse(double output, double target);
double mse(std::vector<double> outputs, std::vector<double> target);

#endif
