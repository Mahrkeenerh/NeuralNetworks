#ifndef networks
#define networks

#include <vector>

#include "datasets.hpp"
#include "layers/layers.hpp"

class DenseNetwork {
   public:
    DenseNetwork(std::vector<int> layer_sizes);

    std::vector<double> predict(std::vector<double> input);

    void fit(Dataset1D dataset, int epochs, int minibatch_size, double learning_rate_start,
             double learning_rate_end, bool verbose = true);

    double accuracy(std::vector<std::vector<double>> inputs, std::vector<int> targets);

   private:
    std::vector<int> layer_sizes;
    std::vector<Layer*> layers;
    std::vector<std::vector<double>> outputs;
    std::vector<std::vector<std::vector<double>>> updates;

    void forwardpropagate(std::vector<double> input);
    void backpropagate(std::vector<double> target_vector);

    void calculate_updates(double learning_rate);
    void apply_updates(int minibatch_size);
    void clear_updates();
};

double mse(double output, double target);
double mse(std::vector<double> outputs, std::vector<double> target);

#endif
