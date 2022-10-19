#ifndef DropoutLayer_hpp
#define DropoutLayer_hpp

#include <stdexcept>

#include "BaseLayer.hpp"

class DropoutLayer : public Layer {
   public:
    DropoutLayer(double dropout_chance = 0.5);
    DropoutLayer(int width, double dropout_chance = 0.5);
    void setup(int input_size) override;

    std::vector<double> forwardpropagate(std::vector<double> input) override;
    void out_errors(std::vector<double> output, std::vector<double> target_vector,
                    std::vector<double>* gradients) override {
        throw std::runtime_error("DropoutLayer::out_errors() is not valid");
    };

   private:
    double dropout_chance;
};

#endif
