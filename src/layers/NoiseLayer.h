#ifndef NoiseLayer_h
#define NoiseLayer_h

#include <stdexcept>

#include "BaseLayer.h"

class NoiseLayer : public Layer {
   public:
    NoiseLayer(int output_size, float noise_chance, float noise_scale = 0.1);

    std::vector<float> forwardpropagate(std::vector<float> input) override;
    void out_errors(std::vector<float> target_vector) override {
        throw std::runtime_error("NoiseLayer::out_errors() is not valid");
    };

   private:
    float noise_chance;
    float noise_scale;
};

#endif
