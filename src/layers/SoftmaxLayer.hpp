#ifndef SoftmaxLayer_hpp
#define SoftmaxLayer_hpp

#include <stdexcept>

#include "BaseLayer.hpp"

class SoftmaxLayer : public Layer {
   public:
    SoftmaxLayer(int input_size, int output_size);

    std::vector<std::vector<double>> momentum;
    std::vector<std::vector<double>> variance;
    double beta1, beta2, eta, epsilon;

    std::vector<std::vector<double>> weight_delta;

    std::vector<double> predict(std::vector<double> input) override;
    std::vector<double> forwardpropagate(std::vector<double> input) override {
        return this->predict(input);
    };
    void out_errors(std::vector<double> output, std::vector<double> target_vector,
                    std::vector<double>* gradients) override;
    void backpropagate(Layer* connected_layer, std::vector<double> output,
                       std::vector<double> target_vector, std::vector<double>* gradients,
                       std::vector<double> connected_gradients) override {
        throw std::runtime_error("SoftmaxLayer::backpropagate() is not valid");
    }
    void calculate_updates(std::vector<std::vector<double>>* updates, std::vector<double> gradients,
                           std::vector<double> input, double learning_rate) override;
    void apply_updates(std::vector<std::vector<double>> updates, int minibatch_size) override;
};

#endif
