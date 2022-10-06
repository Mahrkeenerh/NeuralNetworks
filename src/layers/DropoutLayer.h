#ifndef DropoutLayer_h
#define DropoutLayer_h

#include <stdexcept>

#include "BaseLayer.h"

class DropoutLayer : public Layer {
   public:
    DropoutLayer(int output_size, double dropout_chance);

    std::vector<double> predict(std::vector<double> input) override;
    std::vector<double> forwardpropagate(std::vector<double> input) override;
    void out_errors(std::vector<double> target_vector) override {
        throw std::runtime_error("DropoutLayer::out_errors() is not valid");
    };
    void backpropagate(Layer* connected_layer, std::vector<double> target_vector) override;
    void update_weights(std::vector<double> input_data, double learning_rate) override { return; };

   private:
    double dropout_chance;
};

#endif
