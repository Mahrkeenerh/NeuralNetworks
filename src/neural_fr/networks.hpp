#ifndef networks
#define networks

#include <vector>

#include "datasets.hpp"
#include "layers.hpp"

class DenseNetwork {
   public:
    DenseNetwork(int input_size);

    DenseNetwork add_layer(Layer* layer);

    std::vector<double> predict(std::vector<double> input);

    void fit(Dataset1D dataset, int epochs, int minibatch_size, double learning_rate_start,
             double learning_rate_end = -1, bool verbose = true);

    double accuracy(std::vector<std::vector<double>> inputs, std::vector<int> targets);

   private:
    int input_size, size = 0;
    std::vector<Layer*> layers;
    std::vector<std::vector<std::vector<double>>> outputs;
    std::vector<std::vector<std::vector<double>>> gradients;
    std::vector<std::vector<std::vector<double>>> updates;

    void forwardpropagate(int thread_id, std::vector<double> input);
    void backpropagate(int thread_id, std::vector<double> target_vector);

    void calculate_updates(int thread_id, double learning_rate);
    void apply_updates(int minibatch_size);
    void clear_updates();
};

#endif