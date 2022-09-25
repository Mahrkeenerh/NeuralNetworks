#ifndef layers_h
#define layers_h

#include <vector>

class DenseLayer {
   public:
    int input_size;
    int output_size;

    DenseLayer(int input_size, int output_size);

    std::vector<double> predict(std::vector<double> input);

   private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

#endif
