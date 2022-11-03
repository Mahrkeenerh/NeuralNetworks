#ifndef layers_hpp
#define layers_hpp

#include <cmath>
#include <stdexcept>
#include <vector>

#include "optimizations.hpp"

class Layer {
   public:
    virtual void setup(Layer* previous, Layer* next, int max_threads){};

    int input_size, output_size;
    Layer *previous, *next;

    virtual std::vector<double> predict(std::vector<double> input, int thread_id) { return input; }
    virtual std::vector<double> predict(int thread_id) {
        return this->previous->get_outputs({thread_id});
    }
    virtual std::vector<double> forwardpropagate(std::vector<double> input, int thread_id) {
        return this->predict(input, thread_id);
    }
    virtual std::vector<double> forwardpropagate(int thread_id) { return this->predict(thread_id); }

    virtual void out_errors(int thread_id, std::vector<double> target_vector) {}
    virtual void backpropagate(int thread_id) {}

    virtual void calculate_updates(int thread_id, double learning_rate) {}
    virtual void apply_updates(int minibatch_size) {}
    virtual void clear_updates() {}

    virtual std::vector<std::vector<double>> get_weights() { return {}; }
    virtual std::vector<double> get_outputs(std::vector<int> loc) { return std::vector<double>(); }
    virtual std::vector<double> get_gradients(std::vector<int> loc) { return std::vector<double>(); }

   protected:
    double (*activation)(double);
    double (*derivative)(double);
};

class InputLayer : public Layer {
   public:
    InputLayer(int size);
    void setup(Layer* previous, Layer* next, int max_threads) override;

    std::vector<std::vector<double>> outputs;
    std::vector<std::vector<double>> gradients;
    std::vector<std::vector<double>> updates;

    std::vector<double> predict(std::vector<double> input, int thread_id) override;

    std::vector<double> get_outputs(std::vector<int> loc) override;

   private:
};

class DenseLayer : public Layer {
   public:
    DenseLayer(int width, double (*activation)(double));
    void setup(Layer* previous, Layer* next, int max_threads) override;

    // std::vector<std::vector<double>> momentum;
    // std::vector<std::vector<double>> variance;
    // double beta1, beta2, eta, epsilon;
    double beta1;

    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> weight_delta;

    std::vector<std::vector<double>> outputs;
    std::vector<std::vector<double>> gradients;
    std::vector<std::vector<double>> updates;

    // network functions
    std::vector<double> predict(int thread_id) override;
    void out_errors(int thread_id, std::vector<double> target_vector) override;
    void backpropagate(int thread_id) override;
    void calculate_updates(int thread_id, double learning_rate) override;
    void apply_updates(int minibatch_size) override;
    void clear_updates() override;

    // layer functions
    std::vector<std::vector<double>> get_weights() override;
    std::vector<double> get_outputs(std::vector<int> loc) override;
    std::vector<double> get_gradients(std::vector<int> loc) override;
};

class DropoutLayer : public Layer {
   public:
    DropoutLayer(double dropout_chance = 0.5);
    void setup(Layer* previous, Layer* next, int max_threads) override;

    std::vector<std::vector<double>> weights;

    std::vector<std::vector<double>> outputs;
    std::vector<std::vector<double>> gradients;
    std::vector<std::vector<double>> updates;

    // network functions
    std::vector<double> forwardpropagate(int thread_id) override;
    void out_errors(int thread_id, std::vector<double> target_vector) override {
        throw std::runtime_error("DropoutLayer::out_errors() is not valid");
    };

    // layer functions
    std::vector<std::vector<double>> get_weights() override;
    std::vector<double> get_outputs(std::vector<int> loc) override;
    std::vector<double> get_gradients(std::vector<int> loc) override;

   private:
    double dropout_chance;
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
