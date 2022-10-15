#ifndef NoiseLayer_h
#define NoiseLayer_h

#include <stdexcept>

#include "BaseLayer.h"

class NoiseLayer : public Layer {
   public:
    NoiseLayer(int output_size, double noise_chance, double noise_scale = 0.1);

    std::vector<double> forwardpropagate(std::vector<double> input) override;
    void out_errors(std::vector<double> target_vector) override {
        throw std::runtime_error("NoiseLayer::out_errors() is not valid");
    };

   private:
    double noise_chance;
    double noise_scale;
};

#endif
