#ifndef BaseLayer_h
#define BaseLayer_h

#include <cmath>
#include <vector>

class Layer {
   public:
    int input_size, output_size;

    virtual std::vector<double> predict(std::vector<double> input) { return this->outputs; }
    virtual void out_errors(std::vector<double> target_vector) {}
    virtual void backpropagate(Layer* connected_layer, std::vector<double> target_vector) {}
    virtual void update_weights(std::vector<double> input_data, double learning_rate) {}

    std::vector<std::vector<double>> weights;
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
