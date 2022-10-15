#ifndef BaseLayer_hpp
#define BaseLayer_hpp

#include <cmath>
#include <vector>

class Layer {
   public:
    int input_size, output_size;

    virtual std::vector<double> predict(std::vector<double> input) {
        // Calculate output for each neuron
        for (int n_i = 0; n_i < this->output_size; n_i++) {
            this->outputs[n_i] = input[n_i];
        }

        return this->outputs;
    }
    virtual std::vector<double> forwardpropagate(std::vector<double> input) { return this->outputs; }

    virtual void out_errors(std::vector<double> target_vector) {}
    virtual void backpropagate(Layer* connected_layer, std::vector<double> target_vector) {
        for (int n_i = 0; n_i < this->output_size; n_i++) {
            this->errors[n_i] = connected_layer->errors[n_i];
        }
    }

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
