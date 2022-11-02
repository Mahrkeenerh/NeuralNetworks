#include "layers.hpp"

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
    }
}

DenseLayer::DenseLayer(int input_size, int output_size, double (*activation)(double)) {
    this->input_size = input_size;
    this->output_size = output_size;
    this->activation = activation;

    if (activation == sigmoid) {
        this->derivative = sigmoid_derivative;
    } else if (activation == relu) {
        this->derivative = relu_derivative;
    } else if (activation == leaky_relu) {
        this->derivative = leaky_relu_derivative;
    } else if (activation == swish) {
        this->derivative = swish_derivative;
    }

    this->weights =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));

    // Momentum value
    this->beta1 = 0.3;
    this->weight_delta =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));

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
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size + 1; j++) {
            if (activation == sigmoid) {
                // Initialize weights with random values with uniform distribution
                // [-(1 / sqrt(input_size)), 1 / sqrt(input_size)]
                this->weights[i][j] =
                    (rand() / (double)RAND_MAX) * 2.0 / sqrt(input_size) - 1.0 / sqrt(input_size);
            } else {
                // He initialization with normal distribution
                this->weights[i][j] = randn() * sqrt(2.0 / input_size);
            }
        }
    }
}

void DenseLayer::setup(int input_size) {
    this->input_size = input_size;

    this->weights =
        std::vector<std::vector<double>>(this->output_size, std::vector<double>(input_size + 1, 0.0));

    // Momentum value
    this->beta1 = 0.3;
    this->weight_delta =
        std::vector<std::vector<double>>(this->output_size, std::vector<double>(input_size + 1, 0.0));

    // Initialize weights
    for (int i = 0; i < this->output_size; i++) {
        for (int j = 0; j < input_size + 1; j++) {
            if (activation == sigmoid) {
                // Initialize weights with random values with uniform distribution
                // [-(1 / sqrt(input_size)), 1 / sqrt(input_size)]
                this->weights[i][j] =
                    (rand() / (double)RAND_MAX) * 2.0 / sqrt(input_size) - 1.0 / sqrt(input_size);
            } else {
                // He initialization with normal distribution
                this->weights[i][j] = randn() * sqrt(2.0 / input_size);
            }
        }
    }
}

std::vector<double> DenseLayer::predict(std::vector<double> input) {
    std::vector<double> output(this->output_size, 0.0);

    // Calculate output for each neuron
    for (int n_i = 0; n_i < this->output_size; n_i += consts::MAT_MAX) {
        for (int n_j = 0; n_j < consts::MAT_MAX && n_i + n_j < this->output_size; n_j++) {
            output[n_i + n_j] = this->weights[n_i + n_j][0];
        }

        for (int i = 0; i < this->input_size; i++) {
            for (int n_j = 0; n_j < consts::MAT_MAX && n_i + n_j < this->output_size; n_j++) {
                output[n_i + n_j] += this->weights[n_i + n_j][i + 1] * input[i];
            }
        }
    }

    // Apply activation function
    for (int i = 0; i < this->output_size; i++) {
        output[i] = this->activation(output[i]);
    }

    return output;
}

void DenseLayer::out_errors(std::vector<double> output, std::vector<double> target_vector,
                            std::vector<double>* gradients) {
    // Calculate errors - MSE
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        (*gradients)[n_i] = output[n_i] - target_vector[n_i];
    }

    // Apply activation function
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        (*gradients)[n_i] *= this->derivative(output[n_i]);
    }
}

void DenseLayer::backpropagate(Layer* connected_layer, std::vector<double> output,
                               std::vector<double>* gradients, std::vector<double> connected_gradients) {
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        (*gradients)[n_i] = 0;

        for (int o_i = 0; o_i < connected_layer->output_size; o_i++) {
            (*gradients)[n_i] += connected_gradients[o_i] * connected_layer->weights[o_i][n_i + 1];
        }
    }

    // Apply activation function
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        (*gradients)[n_i] *= this->derivative(output[n_i]);
    }
}

void DenseLayer::calculate_updates(std::vector<std::vector<double>>* updates,
                                   std::vector<double> gradients, std::vector<double> input,
                                   double learning_rate) {
    double update;
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        update = gradients[0] * learning_rate + this->beta1 * this->weight_delta[n_i][0];
        (*updates)[n_i][0] += update;

        for (int w_i = 1; w_i < this->input_size + 1; w_i++) {
            update = gradients[n_i] * learning_rate * input[w_i - 1] +
                     this->beta1 * this->weight_delta[n_i][w_i];
            (*updates)[n_i][w_i] += update;
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

void DenseLayer::apply_updates(std::vector<std::vector<double>> updates, int minibatch_size) {
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        for (int w_i = 0; w_i < this->input_size + 1; w_i++) {
            this->weights[n_i][w_i] -= updates[n_i][w_i];
            this->weight_delta[n_i][w_i] = updates[n_i][w_i] / minibatch_size;
        }
    }
}

// DROPOUT LAYER
DropoutLayer::DropoutLayer(double dropout_chance) { this->dropout_chance = dropout_chance; }

DropoutLayer::DropoutLayer(int width, double dropout_chance) {
    this->input_size = width;
    this->output_size = width;

    this->dropout_chance = dropout_chance;

    this->weights = std::vector<std::vector<double>>(width, std::vector<double>(input_size + 1, 1.0));
}

void DropoutLayer::setup(int input_size) {
    this->input_size = input_size;
    this->output_size = input_size;

    this->weights =
        std::vector<std::vector<double>>(input_size, std::vector<double>(input_size + 1, 1.0));
}

std::vector<double> DropoutLayer::forwardpropagate(std::vector<double> input) {
    std::vector<double> output(this->output_size, 0.0);

    // Calculate output for each neuron
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        if (rand() / (double)RAND_MAX > this->dropout_chance) {
            output[n_i] = input[n_i] * (1.0 / (1.0 - this->dropout_chance));
        } else {
            output[n_i] = 0.0;
        }
    }

    return output;
};

// SOFTMAX LAYER
SoftmaxLayer::SoftmaxLayer(int width) { this->output_size = width; }

SoftmaxLayer::SoftmaxLayer(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;

    this->weights =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 1.0));

    // Momentum value
    this->beta1 = 0.2;
    this->weight_delta =
        std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));

    // Initialize weights
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size + 1; j++) {
            // He initialization with normal distribution
            // this->weights[i][j] = randn() * sqrt(2.0 / input_size);

            // Initialize weights with random values with uniform distribution
            // [-(1 / sqrt(input_size)), 1 / sqrt(input_size)]
            this->weights[i][j] =
                (rand() / (double)RAND_MAX) * 2.0 / sqrt(input_size) - 1.0 / sqrt(input_size);
        }
    }

    // Adam settings
    // this->momentum =
    //     std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    // this->variance =
    //     std::vector<std::vector<double>>(output_size, std::vector<double>(input_size + 1, 0.0));
    // this->beta1 = 0.1;
    // this->beta2 = 0.999;
    // this->eta = 0.01;
    // this->epsilon = 1e-8;
}

void SoftmaxLayer::setup(int input_size) {
    this->input_size = input_size;

    this->weights =
        std::vector<std::vector<double>>(this->output_size, std::vector<double>(input_size + 1, 1.0));

    // Momentum value
    this->beta1 = 0.2;
    this->weight_delta =
        std::vector<std::vector<double>>(this->output_size, std::vector<double>(input_size + 1, 0.0));

    // Initialize weights
    for (int i = 0; i < this->output_size; i++) {
        for (int j = 0; j < input_size + 1; j++) {
            // He initialization with normal distribution
            // this->weights[i][j] = randn() * sqrt(2.0 / input_size);

            // Initialize weights with random values with uniform distribution
            // [-(1 / sqrt(input_size)), 1 / sqrt(input_size)]
            this->weights[i][j] =
                (rand() / (double)RAND_MAX) * 2.0 / sqrt(input_size) - 1.0 / sqrt(input_size);
        }
    }
}

std::vector<double> SoftmaxLayer::predict(std::vector<double> input) {
    std::vector<double> output(this->output_size, 0.0);

    // Calculate output for each neuron
    double sum = 0;
    // double max = *std::max_element(std::begin(this->outputs), std::end(this->outputs));
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        output[n_i] = this->weights[n_i][0];

        for (int i = 0; i < this->input_size; i++) {
            output[n_i] += this->weights[n_i][i + 1] * input[i];
        }
        // sum += exp(this->outputs[n_i] - max);
        sum += exp(output[n_i]);
    }
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        // this->outputs[n_i] = exp(this->outputs[n_i] - max) / sum;
        output[n_i] = exp(output[n_i]) / sum;
    }

    return output;
}

void SoftmaxLayer::out_errors(std::vector<double> output, std::vector<double> target_vector,
                              std::vector<double>* gradients) {
    // Derivative of cross entropy loss
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        (*gradients)[n_i] = output[n_i] - target_vector[n_i];
    }
}

void SoftmaxLayer::calculate_updates(std::vector<std::vector<double>>* updates,
                                     std::vector<double> gradients, std::vector<double> input,
                                     double learning_rate) {
    double update;
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        update = gradients[0] * learning_rate + this->beta1 * this->weight_delta[n_i][0];
        (*updates)[n_i][0] += update;

        for (int w_i = 1; w_i < this->input_size + 1; w_i++) {
            update = gradients[n_i] * learning_rate * input[w_i - 1] +
                     this->beta1 * this->weight_delta[n_i][w_i];
            (*updates)[n_i][w_i] += update;
        }
    }
}

void SoftmaxLayer::apply_updates(std::vector<std::vector<double>> updates, int minibatch_size) {
    for (int n_i = 0; n_i < this->output_size; n_i++) {
        for (int w_i = 0; w_i < this->input_size + 1; w_i++) {
            this->weights[n_i][w_i] -= updates[n_i][w_i];
            this->weight_delta[n_i][w_i] = updates[n_i][w_i] / minibatch_size;
        }
    }
}

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

double softmax(double x) { return x; }

double softmax_derivative(double x) { return x * (1 - x); }