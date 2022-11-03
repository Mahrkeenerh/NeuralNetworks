#include "layers.hpp"

// INPUT LAYER
InputLayer::InputLayer(int size) {
    this->output_size = size;
    this->input_size = size;
}

void InputLayer::setup(Layer* previous, Layer* next, int max_threads) {
    this->next = next;

    this->outputs =
        std::vector<std::vector<double>>(max_threads, std::vector<double>(this->output_size, 0.0));
}

std::vector<double> InputLayer::predict(std::vector<double> input, int thread_id) {
    this->outputs[thread_id] = input;

    return input;
}

std::vector<double> InputLayer::get_outputs(std::vector<int> loc) { return this->outputs[loc[0]]; }

// DENSE LAYER
DenseLayer::DenseLayer(int width, double (*activation)(double)) {
    this->output_size = width;
    this->activation = activation;

    if (activation == sigmoid) {
        this->derivative = sigmoid_derivative;
    } else if (activation == relu) {
        this->derivative = relu_derivative;
    } else if (activation == leaky_relu) {
        this->derivative = leaky_relu_derivative;
    } else if (activation == swish) {
        this->derivative = swish_derivative;
    } else if (activation == softmax) {
        this->derivative = softmax_derivative;
    } else {
        this->derivative = nullptr;
    }
}

void DenseLayer::setup(Layer* previous, Layer* next, int thread_count) {
    this->previous = previous;
    this->next = next;

    this->input_size = previous->output_size;

    this->weights = std::vector<std::vector<double>>(this->output_size,
                                                     std::vector<double>(this->input_size + 1, 0.0));
    this->outputs =
        std::vector<std::vector<double>>(thread_count, std::vector<double>(this->output_size, 0.0));
    this->gradients =
        std::vector<std::vector<double>>(thread_count, std::vector<double>(this->output_size, 0.0));
    this->updates = std::vector<std::vector<double>>(this->output_size,
                                                     std::vector<double>(this->input_size + 1, 0.0));

    // Momentum value
    this->beta1 = 0.3;
    this->weight_delta = std::vector<std::vector<double>>(
        this->output_size, std::vector<double>(this->input_size + 1, 0.0));

    // Adam settings
    /* this->momentum =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    this->variance =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    this->beta1 = 0.9;
    this->beta2 = 0.999;
    this->eta = 0.01;
    this->epsilon = 1e-8; */

    // Initialize weights
    for (int i = 0; i < this->output_size; i++) {
        for (int j = 0; j < this->input_size + 1; j++) {
            if (activation == sigmoid) {
                // Initialize weights with random values with uniform distribution
                // [-(1 / sqrt(input_size)), 1 / sqrt(input_size)]
                this->weights[i][j] = (rand() / (double)RAND_MAX) * 2.0 / sqrt(this->input_size) -
                                      1.0 / sqrt(this->input_size);
            } else {
                // He initialization with normal distribution
                this->weights[i][j] = randn() * sqrt(2.0 / this->input_size);
            }
        }
    }
}

std::vector<double> DenseLayer::predict(int thread_id) {
    std::vector<double> prev_output = this->previous->get_outputs({thread_id});

    // Calculate output for each neuron
    for (int n_i = 0; n_i < this->output_size; n_i += consts::MAT_MAX) {
        for (int n_j = 0; n_j < consts::MAT_MAX && n_i + n_j < this->output_size; n_j++) {
            this->outputs[thread_id][n_i + n_j] = this->weights[n_i + n_j][0];
        }

        for (int i = 0; i < this->input_size; i++) {
            for (int n_j = 0; n_j < consts::MAT_MAX && n_i + n_j < this->output_size; n_j++) {
                this->outputs[thread_id][n_i + n_j] += this->weights[n_i + n_j][i + 1] * prev_output[i];
            }
        }
    }

    double sum = 0;

    // Apply activation function
    for (int i = 0; i < this->output_size; i++) {
        this->outputs[thread_id][i] = this->activation(this->outputs[thread_id][i]);
        sum += this->outputs[thread_id][i];
    }

    // Softmax normalization
    if (this->activation == softmax) {
        for (int i = 0; i < this->output_size; i++) {
            this->outputs[thread_id][i] /= sum;
        }
    }

    return this->outputs[thread_id];
}

void DenseLayer::out_errors(int thread_id, std::vector<double> target_vector) {
    // Calculate errors - MSE
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->gradients[thread_id][n_i] = this->outputs[thread_id][n_i] - target_vector[n_i];
    }

    // Apply derivative activation function
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->gradients[thread_id][n_i] *= this->derivative(this->outputs[thread_id][n_i]);
    }
}

void DenseLayer::backpropagate(int thread_id) {
    std::vector<double> next_gradients = this->next->get_gradients({thread_id});
    std::vector<std::vector<double>> next_weights = this->next->get_weights();

    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->gradients[thread_id][n_i] = 0;

        for (int o_i = 0; o_i < this->next->output_size; o_i++) {
            this->gradients[thread_id][n_i] += next_gradients[o_i] * next_weights[o_i][n_i + 1];
        }
    }

    // Apply activation function
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        this->gradients[thread_id][n_i] *= this->derivative(this->outputs[thread_id][n_i]);
    }
}

void DenseLayer::calculate_updates(int thread_id, double learning_rate) {
    std::vector<double> prev_output = this->previous->get_outputs({thread_id});

    double update;
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        update =
            this->gradients[thread_id][0] * learning_rate + this->beta1 * this->weight_delta[n_i][0];
        this->updates[n_i][0] += update;

        for (int w_i = 1; w_i < this->input_size + 1; w_i++) {
            update = this->gradients[thread_id][n_i] * learning_rate * prev_output[w_i - 1] +
                     this->beta1 * this->weight_delta[n_i][w_i];
            this->updates[n_i][w_i] += update;
        }
    }

    // Adam
    /* double grad, alpha;
    #pragma omp parallel for
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        //#pragma omp parallel for
        for (int w_i = 0; w_i < this->input_size + 1; w_i++) {
            grad = this->errors[n_i];

            this->momentum[n_i][w_i] = this->beta1 * this->momentum[n_i][w_i] + (1 - this->beta1) * grad;
            this->variance[n_i][w_i] =
                this->beta2 * this->variance[n_i][w_i] + (1 - this->beta2) * pow(grad, 2);

            alpha = this->eta * sqrt((1 - pow(this->beta2, t + 1)) / (1 - pow(this->beta1, t + 1)));

            // Bias
            if (w_i == 0) {
                this->weights[n_i][w_i] -= learning_rate * alpha * this->momentum[n_i][w_i] /
                                           (sqrt(this->variance[n_i][w_i]) + this->epsilon);
            }
            // Weight
            else {
                this->weights[n_i][w_i] -= learning_rate * input[w_i - 1] * alpha *
                                           this->momentum[n_i][w_i] /
                                           (sqrt(this->variance[n_i][w_i]) + this->epsilon);
            }
        }
    } */
}

void DenseLayer::apply_updates(int minibatch_size) {
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        for (int w_i = 0; w_i < this->input_size + 1; w_i++) {
            this->weights[n_i][w_i] -= this->updates[n_i][w_i];
            this->weight_delta[n_i][w_i] = updates[n_i][w_i] / minibatch_size;
        }
    }
}

void DenseLayer::clear_updates() {
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        for (int w_i = 0; w_i < this->input_size + 1; w_i++) {
            this->updates[n_i][w_i] = 0;
        }
    }
}

std::vector<std::vector<double>> DenseLayer::get_weights() { return this->weights; }

std::vector<double> DenseLayer::get_outputs(std::vector<int> loc) { return this->outputs[loc[0]]; }

std::vector<double> DenseLayer::get_gradients(std::vector<int> loc) { return this->gradients[loc[0]]; }

// DROPOUT LAYER
DropoutLayer::DropoutLayer(double dropout_chance) { this->dropout_chance = dropout_chance; }

void DropoutLayer::setup(Layer* previous, Layer* next, int thread_count) {
    this->input_size = previous->output_size;
    this->output_size = this->input_size;

    this->previous = previous;
    this->next = next;

    this->weights = std::vector<std::vector<double>>(this->input_size,
                                                     std::vector<double>(this->input_size + 1, 1.0));
}

std::vector<double> DropoutLayer::forwardpropagate(int thread_id) {
    std::vector<double> prev_output = this->previous->get_outputs({thread_id});

    // Calculate output for each neuron
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        if (rand() / (double)RAND_MAX > this->dropout_chance) {
            this->outputs[thread_id][n_i] = prev_output[n_i] * (1.0 / (1.0 - this->dropout_chance));
        } else {
            this->outputs[thread_id][n_i] = 0.0;
        }
    }

    return this->outputs[thread_id];
};

std::vector<std::vector<double>> DropoutLayer::get_weights() { return this->weights; }

std::vector<double> DropoutLayer::get_outputs(std::vector<int> loc) { return this->outputs[loc[0]]; }

std::vector<double> DropoutLayer::get_gradients(std::vector<int> loc) { return this->gradients[loc[0]]; }

// Random value from normal distribution using Box-Muller transform
double randn() {
    double u1 = rand() / (double)RAND_MAX;
    double u2 = rand() / (double)RAND_MAX;
    double out = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    // Avoid infinite values
    while (out == INFINITY || out == -INFINITY) {
        u1 = rand() / (double)RAND_MAX;
        u2 = rand() / (double)RAND_MAX;
        out = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    }

    return out;
}

// Activations
double sigmoid(double x) { return 1 / (1 + exp(-x)); }

double sigmoid_derivative(double x) { return x * (1 - x); }

double relu(double x) { return x > 0 ? x : 0; }

double relu_derivative(double x) { return x > 0 ? 1 : 0; }

double leaky_relu(double x) { return x > 0 ? x : 0.001 * x; }

double leaky_relu_derivative(double x) { return x > 0 ? 1 : 0.001; }

double swish(double x) { return x / (1 + exp(-x)); }

double swish_derivative(double x) { return (1 + exp(-x) + x * exp(-x)) / pow(1 + exp(-x), 2); }

double softmax(double x) { return exp(x); }

double softmax_derivative(double x) { return 1; }
