#ifndef layers_h
#define layers_h

#include <vector>

class DenseLayer {
   public:
    int input_size, output_size;

    DenseLayer(int input_size, int output_size, double (*activation)(double));

    std::vector<double> predict(std::vector<double> input);

    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> gradients;
    std::vector<double> errors;
    std::vector<double> outputs;

    double (*activation)(double);
    double (*derivative)(double);
};

double randn();

double sigmoid(double x);
double sigmoid_derivative(double x);

double relu(double x);
double relu_derivative(double x);

double leaky_relu(double x);
double leaky_relu_derivative(double x);

double swish(double x);
double swish_derivative(double x);

double softmax(double x);
double softmax_derivative(double x);

#endif
