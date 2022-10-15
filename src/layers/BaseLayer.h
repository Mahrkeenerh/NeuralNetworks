#ifndef BaseLayer_h
#define BaseLayer_h

#include <cmath>
#include <vector>

class Layer {
   public:
    int input_size, output_size;

    virtual std::vector<float> predict(std::vector<float> input) {
        // Calculate output for each neuron
        for (int n_i = 0; n_i < this->output_size; n_i++) {
            this->outputs[n_i] = input[n_i];
        }

        return this->outputs;
    }
    virtual std::vector<float> forwardpropagate(std::vector<float> input) { return this->outputs; }

    virtual void out_errors(std::vector<float> target_vector) {}
    virtual void backpropagate(Layer* connected_layer, std::vector<float> target_vector) {
        for (int n_i = 0; n_i < this->output_size; n_i++) {
            this->errors[n_i] = connected_layer->errors[n_i];
        }
    }

    virtual void update_weights(std::vector<float> input_data, float learning_rate, int t) {}

    std::vector<std::vector<float>> weights;
    std::vector<float> errors;
    std::vector<float> outputs;

    float (*activation)(float);
    float (*derivative)(float);
};

float randn();

float sigmoid(float x);
float sigmoid_derivative(float x);

float relu(float x);
float relu_derivative(float x);

float leaky_relu(float x);
float leaky_relu_derivative(float x);

float swish(float x);
float swish_derivative(float x);

float softmax(float x);
float softmax_derivative(float x);

#endif
