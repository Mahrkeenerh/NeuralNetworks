#ifndef networks
#define networks

#include <vector>

#include "datasets.h"
#include "layers.h"

class DenseNetwork {
   public:
    DenseNetwork(std::vector<int> layer_sizes);

    std::vector<double> predict(std::vector<double> input);
    void backpropagate(std::vector<double> outputs, std::vector<double> target_vector);
    void update_weights(std::vector<double> input_data, double learning_rate);
    void fit(Dataset1D dataset, int epochs, double learning_rate, bool verbose = true);

    double error(std::vector<std::vector<double>> inputs, std::vector<int> targets);
    double accuracy(std::vector<std::vector<double>> inputs, std::vector<int> targets);
    std::vector<double> output_errors(std::vector<double> inputs, int target,
                                      double (*derivative)(double));

   private:
    std::vector<int> layer_sizes;
    std::vector<DenseLayer> layers;
};

double mse(double output, double target);
double mse(std::vector<double> outputs, std::vector<double> target);

#endif
