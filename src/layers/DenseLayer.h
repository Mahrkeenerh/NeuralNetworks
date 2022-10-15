#ifndef DenseLayer_h
#define DenseLayer_h

#include "BaseLayer.h"

class DenseLayer : public Layer {
   public:
    DenseLayer(int input_size, int output_size, float (*activation)(float));

    std::vector<std::vector<float>> momentum;
    std::vector<std::vector<float>> variance;
    float beta1, beta2, eta, epsilon;

    std::vector<std::vector<float>> weight_delta;

    std::vector<float> predict(std::vector<float> input) override;
    std::vector<float> forwardpropagate(std::vector<float> input) override {
        return this->predict(input);
    };
    void out_errors(std::vector<float> target_vector) override;
    void backpropagate(Layer* connected_layer, std::vector<float> target_vector) override;
    void update_weights(std::vector<float> input_data, float learning_rate, int t) override;
};

#endif
