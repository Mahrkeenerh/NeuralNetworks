#ifndef layers_h
#define layers_h

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

class DenseLayer : public Layer {
   public:
    DenseLayer(int input_size, int output_size, double (*activation)(double));

    std::vector<double> predict(std::vector<double> input) override;
    void out_errors(std::vector<double> target_vector) override;
    void backpropagate(Layer* connected_layer, std::vector<double> target_vector) override;
    void update_weights(std::vector<double> input_data, double learning_rate) override;

    // std::vector<std::vector<double>> weights;
    // std::vector<double> errors;
    // std::vector<double> batch_errors;
    // std::vector<double> outputs;
};

class SoftmaxLayer : public Layer {
   public:
    SoftmaxLayer(int input_size, int output_size);

    std::vector<double> predict(std::vector<double> input) override;
    void out_errors(std::vector<double> target_vector) override;
    void backpropagate(Layer* connected_layer, std::vector<double> target_vector) override;
    void update_weights(std::vector<double> input_data, double learning_rate) override;
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
