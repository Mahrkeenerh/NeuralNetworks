#ifndef DenseLayer_hpp
#define DenseLayer_hpp

#include "BaseLayer.hpp"

class DenseLayer : public Layer {
   public:
    DenseLayer(int input_size, int output_size, double (*activation)(double));

    std::vector<std::vector<double>> momentum;
    std::vector<std::vector<double>> variance;
    double beta1, beta2, eta, epsilon;

    std::vector<std::vector<double>> weight_delta;

    std::vector<double> predict(std::vector<double> input) override;
    std::vector<double> forwardpropagate(std::vector<double> input) override {
        return this->predict(input);
    };
    void out_errors(std::vector<double> target_vector) override;
    void backpropagate(Layer* connected_layer, std::vector<double> target_vector) override;
    void update_weights(std::vector<double> input_data, double learning_rate) override;
};

#endif