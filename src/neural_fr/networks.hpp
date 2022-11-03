#ifndef networks
#define networks

#include <vector>

#include "datasets.hpp"
#include "layers.hpp"

class DenseNetwork {
   public:
    DenseNetwork(std::vector<layers::Layer*> layers);

    DenseNetwork add_layer(layers::Layer* layer);

    std::vector<double> predict(std::vector<double> input, int thread_id = 0);

    void fit(Dataset1D dataset, double split, int epochs, int minibatch_size, double learning_rate_start,
             double learning_rate_end = -1, bool verbose = true);

    double accuracy(std::vector<std::vector<double>> inputs, std::vector<int> targets);

   private:
    int size = 0;
    std::vector<layers::Layer*> layers;

    void forwardpropagate(std::vector<double> input, int thread_id);
    void backpropagate(int thread_id, std::vector<double> target_vector);

    void calculate_updates(int thread_id, double learning_rate);
    void apply_updates(int minibatch_size);
    void clear_updates();

    void before_batch();

    double valid_accuracy(std::vector<std::vector<double>> inputs, std::vector<int> targets,
                          std::vector<int> valid_i);
};

#endif
