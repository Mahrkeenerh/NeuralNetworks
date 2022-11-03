#ifndef layers_hpp
#define layers_hpp

#include <cmath>
#include <stdexcept>
#include <vector>

#include "optimizations.hpp"

namespace layers {
class Layer {
   public:
    virtual void setup(Layer* previous, Layer* next, int max_threads){};

    std::vector<int> input_shape, output_shape;

    // network functions
    virtual void predict(std::vector<double> input, int thread_id) {}
    virtual void predict(int thread_id) {}
    virtual void forwardpropagate(std::vector<double> input, int thread_id) {
        this->predict(input, thread_id);
    }
    virtual void forwardpropagate(int thread_id) { this->predict(thread_id); }

    virtual void out_errors(int thread_id, std::vector<double> target_vector) {}
    virtual void backpropagate(int thread_id) {}

    virtual void calculate_updates(int thread_id, double learning_rate) {}
    virtual void apply_updates(int minibatch_size) {}
    virtual void clear_updates() {}

    virtual void before_batch(int thread_id) {}

    // layer functions
    virtual std::vector<std::vector<double>> get_weights() { return {}; }
    virtual std::vector<double> get_outputs(std::vector<int> loc) { return std::vector<double>(); }
    virtual std::vector<double> get_gradients(std::vector<int> loc) { return std::vector<double>(); }

   protected:
    Layer *previous, *next;

    double (*activation)(double);
    double (*derivative)(double);
};

class Input1D : public Layer {
   public:
    Input1D(int size);
    void setup(Layer* previous, Layer* next, int max_threads) override;

    // network functions
    void predict(std::vector<double> input, int thread_id) override;

    // layer functions
    std::vector<double> get_outputs(std::vector<int> loc) override;

   private:
    std::vector<std::vector<double>> outputs;
};

class Dense : public Layer {
   public:
    Dense(int width, double (*activation)(double));
    void setup(Layer* previous, Layer* next, int max_threads) override;

    // network functions
    void predict(int thread_id) override;
    void out_errors(int thread_id, std::vector<double> target_vector) override;
    void backpropagate(int thread_id) override;
    void calculate_updates(int thread_id, double learning_rate) override;
    void apply_updates(int minibatch_size) override;
    void clear_updates() override;

    // layer functions
    std::vector<std::vector<double>> get_weights() override;
    std::vector<double> get_outputs(std::vector<int> loc) override;
    std::vector<double> get_gradients(std::vector<int> loc) override;

   private:
    double beta1;
    // std::vector<std::vector<double>> momentum;
    // std::vector<std::vector<double>> variance;
    // double beta1, beta2, eta, epsilon;

    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> weight_delta;

    std::vector<std::vector<double>> outputs;
    std::vector<std::vector<double>> gradients;
    std::vector<std::vector<double>> updates;
};

class Dropout : public Layer {
   public:
    Dropout(double dropout_chance = 0.5);
    void setup(Layer* previous, Layer* next, int max_threads) override;

    // network functions
    void predict(int thread_id) override;
    void forwardpropagate(int thread_id) override;
    void out_errors(int thread_id, std::vector<double> target_vector) override {
        throw std::runtime_error("Dropout::out_errors() is not valid");
    };
    void backpropagate(int thread_id) override;

    void before_batch(int thread_id) override;

    // layer functions
    std::vector<std::vector<double>> get_weights() override;
    std::vector<double> get_outputs(std::vector<int> loc) override;
    std::vector<double> get_gradients(std::vector<int> loc) override;

   private:
    double dropout_chance;

    std::vector<std::vector<double>> weights;

    std::vector<std::vector<double>> outputs;
    std::vector<std::vector<double>> gradients;

    std::vector<std::vector<bool>> dropout_mask;
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
}  // namespace layers

#endif
