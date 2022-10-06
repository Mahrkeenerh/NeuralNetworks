#ifndef SoftmaxLayer_h
#define SoftmaxLayer_h

#include "BaseLayer.h"

class SoftmaxLayer : public Layer {
   public:
    SoftmaxLayer(int input_size, int output_size);

    std::vector<double> predict(std::vector<double> input) override;
    void out_errors(std::vector<double> target_vector) override;
    void backpropagate(Layer* connected_layer, std::vector<double> target_vector) override;
    void update_weights(std::vector<double> input_data, double learning_rate) override;
};

#endif
