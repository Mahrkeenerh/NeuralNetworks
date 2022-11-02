#ifndef layers_hpp
#define layers_hpp

#include <cmath>
#include <stdexcept>
#include <vector>

#include "optimizations.hpp"

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
                               std::vector<double>* gradients, std::vector<double> connected_gradients) {
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

class DenseLayer : public Layer {
   public:
    DenseLayer(int width, double (*activation)(double));
    DenseLayer(int input_size, int output_size, double (*activation)(double));
    void setup(int input_size) override;

    std::vector<std::vector<double>> momentum;
    std::vector<std::vector<double>> variance;
    double beta1, beta2, eta, epsilon;

    std::vector<std::vector<double>> weight_delta;

    std::vector<double> predict(std::vector<double> input) override;
    std::vector<double> forwardpropagate(std::vector<double> input) override {
        return this->predict(input);
    };
    void out_errors(std::vector<double> output, std::vector<double> target_vector,
                    std::vector<double>* gradients) override;
    void backpropagate(Layer* connected_layer, std::vector<double> output,
                       std::vector<double>* gradients, std::vector<double> connected_gradients) override;
    void calculate_updates(std::vector<std::vector<double>>* updates, std::vector<double> gradients,
                           std::vector<double> input, double learning_rate) override;
    void apply_updates(std::vector<std::vector<double>> updates, int minibatch_size) override;
};

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

class SoftmaxLayer : public Layer {
   public:
    SoftmaxLayer(int width);
    SoftmaxLayer(int input_size, int output_size);
    void setup(int input_size) override;

    std::vector<std::vector<double>> momentum;
    std::vector<std::vector<double>> variance;
    double beta1, beta2, eta, epsilon;

    std::vector<std::vector<double>> weight_delta;

    std::vector<double> predict(std::vector<double> input) override;
    std::vector<double> forwardpropagate(std::vector<double> input) override {
        return this->predict(input);
    };
    void out_errors(std::vector<double> output, std::vector<double> target_vector,
                    std::vector<double>* gradients) override;
    void backpropagate(Layer* connected_layer, std::vector<double> output,
                       std::vector<double>* gradients,
                       std::vector<double> connected_gradients) override {
        throw std::runtime_error("SoftmaxLayer::backpropagate() is not valid");
    }
    void calculate_updates(std::vector<std::vector<double>>* updates, std::vector<double> gradients,
                           std::vector<double> input, double learning_rate) override;
    void apply_updates(std::vector<std::vector<double>> updates, int minibatch_size) override;
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
