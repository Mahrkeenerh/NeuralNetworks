#ifndef DropoutLayer_h
#define DropoutLayer_h

#include <stdexcept>

#include "BaseLayer.h"

class DropoutLayer : public Layer {
   public:
    DropoutLayer(int output_size, float dropout_chance = 0.5);

    std::vector<float> forwardpropagate(std::vector<float> input) override;
    void out_errors(std::vector<float> target_vector) override {
        throw std::runtime_error("DropoutLayer::out_errors() is not valid");
    };
    void update_weights(std::vector<float> input_data, float learning_rate) override { return; };

   private:
    float dropout_chance;
};

#endif
