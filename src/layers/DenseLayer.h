#ifndef DenseLayer_h
#define DenseLayer_h

#include "BaseLayer.h"

class DenseLayer : public Layer {
   public:
    DenseLayer(int input_size, int output_size, double (*activation)(double));

    std::vector<double> predict(std::vector<double> input) override;
    std::vector<double> forwardpropagate(std::vector<double> input) override {
        return this->predict(input);
    };
    void out_errors(std::vector<double> target_vector) override;
    void backpropagate(Layer* connected_layer, std::vector<double> target_vector) override;
    void update_weights(std::vector<double> input_data, double learning_rate) override;
};

#endif
