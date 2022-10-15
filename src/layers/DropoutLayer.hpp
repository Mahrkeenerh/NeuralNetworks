#ifndef DropoutLayer_hpp
#define DropoutLayer_hpp

#include <stdexcept>

#include "BaseLayer.hpp"

class DropoutLayer : public Layer {
   public:
    DropoutLayer(int output_size, double dropout_chance = 0.5);

    std::vector<double> forwardpropagate(std::vector<double> input) override;
    void out_errors(std::vector<double> target_vector) override {
        throw std::runtime_error("DropoutLayer::out_errors() is not valid");
    };

   private:
    double dropout_chance;
};

#endif
