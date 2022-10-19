#ifndef BaseLayer_hpp
#define BaseLayer_hpp

#include <cmath>
#include <vector>

#include "../optimizations.hpp"

class Layer {
   public:
    virtual void setup(int input_size){};

    int input_size, output_size;

    std::vector<std::vector<double>> weights;

    virtual std::vector<double> predict(std::vector<double> input) { return input; }
    virtual std::vector<double> forwardpropagate(std::vector<double> input) { return input; }

    virtual void out_errors(std::vector<double> output, std::vector<double> target_vector,
                            std::vector<double>* gradients) {}
    virtual void backpropagate(Layer* connected_layer, std::vector<double> output,
                               std::vector<double> target_vector, std::vector<double>* gradients,
                               std::vector<double> connected_gradients) {
        for (int n_i = 0; n_i < this->output_size; n_i++) {
            (*gradients)[n_i] = connected_gradients[n_i];
        }
    }

    virtual void calculate_updates(std::vector<std::vector<double>>* updates,
                                   std::vector<double> gradients, std::vector<double> input,
                                   double learning_rate) {}
    virtual void apply_updates(std::vector<std::vector<double>> updates, int minibatch_size) {}

   protected:
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
