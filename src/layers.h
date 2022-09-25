#ifndef layers_h
#define layers_h

#include <vector>

class DenseLayer {
   public:
    int input_size, output_size;

    DenseLayer(int input_size, int output_size, float (*activation)(float));

    std::vector<float> predict(std::vector<float> input);

   private:
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    float (*activation)(float);
};

#endif
