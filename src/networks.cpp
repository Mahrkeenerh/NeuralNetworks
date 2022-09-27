#include "networks.h"

#include <cmath>
#include <iostream>

DenseNetwork::DenseNetwork(std::vector<int> layer_sizes) {
    this->layer_sizes = layer_sizes;

    // Create layers
    for (int i = 0; i < layer_sizes.size() - 2; i++) {
        this->layers.push_back(DenseLayer(layer_sizes[i], layer_sizes[i + 1], relu));
    }

    // Create output layer
    this->layers.push_back(
        DenseLayer(layer_sizes[layer_sizes.size() - 2], layer_sizes[layer_sizes.size() - 1], sigmoid));
}

std::vector<float> DenseNetwork::predict(std::vector<float> input) {
    std::vector<float> output = input;

    for (int i = 0; i < this->layers.size(); i++) {
        output = this->layers[i].predict(output);
    }

    return output;
}

void DenseNetwork::fit(Dataset1D dataset, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        clock_t start, end;
        start = clock();
        std::cout << "..." << std::endl;

        for (int i = 0; i < dataset.train_data.size(); i++) {
            if (i % 100 == 0) {
                std::cout << "\033[F"
                          << "Input: " << i + 1 << "/" << dataset.train_data.size() << "\033[K"
                          << std::endl;
            }

            // Backpropagation
            std::vector<float> outputs = this->predict(dataset.train_data[i]);
            std::vector<float> target_vector(this->layer_sizes[this->layer_sizes.size() - 1], 0);
            target_vector[dataset.train_labels[i]] = 1;

            // Calculate output layer errors
            for (int n_i = 0; n_i < this->layers[this->layers.size() - 1].output_size; n_i++) {
                this->layers[this->layers.size() - 1].errors[n_i] =
                    (outputs[n_i] - target_vector[n_i]) *
                    this->layers[this->layers.size() - 1].derivative(outputs[n_i]);
            }

            // Calculate hidden layer errors
            for (int l_i = this->layers.size() - 2; l_i >= 0; l_i--) {
                for (int n_i = 0; n_i < this->layers[l_i].output_size; n_i++) {
                    this->layers[l_i].errors[n_i] = 0;

                    for (int o_i = 0; o_i < this->layers[l_i + 1].output_size; o_i++) {
                        this->layers[l_i].errors[n_i] += this->layers[l_i + 1].errors[o_i] *
                                                         this->layers[l_i + 1].weights[o_i][n_i + 1];
                    }

                    this->layers[l_i].errors[n_i] *=
                        this->layers[l_i].derivative(this->layers[l_i].outputs[n_i]);
                }
            }

            // Print outputs, targets, and errors
            // std::cout << "Outputs: ";
            // for (int i = 0; i < outputs.size(); i++) {
            //     std::cout << outputs[i] << " ";
            // }
            // std::cout << std::endl;

            // std::cout << "Targets: ";
            // for (int i = 0; i < target_vector.size(); i++) {
            //     std::cout << target_vector[i] << " ";
            // }
            // std::cout << std::endl;

            // std::cout << "Errors: ";
            // for (int i = 0; i < this->layers[this->layers.size() - 1].errors.size(); i++) {
            //     std::cout << this->layers[this->layers.size() - 1].errors[i] << " ";
            // }
            // std::cout << std::endl;

            // std::cout << "-----------------------------------------" << std::endl;

            // Update weights
            for (int l_i = 0; l_i < this->layers.size(); l_i++) {
                std::vector<float> l_inputs =
                    (l_i == 0) ? dataset.train_data[i] : this->layers[l_i - 1].outputs;

                for (int n_i = 0; n_i < this->layers[l_i].output_size; n_i++) {
                    for (int w_i = 0; w_i < this->layers[l_i].input_size + 1; w_i++) {
                        float input = (w_i == 0) ? 1 : l_inputs[w_i - 1];
                        this->layers[l_i].weights[n_i][w_i] -=
                            this->layers[l_i].errors[n_i] * learning_rate * input;
                    }
                }
            }
        }

        float error = this->error(dataset.test_data, dataset.test_labels);
        float accuracy = this->accuracy(dataset.test_data, dataset.test_labels);

        end = clock();
        float epoch_time = (float)(end - start) / CLOCKS_PER_SEC;

        std::cout << "\033[FEpoch " << epoch + 1 << "/" << epochs << ": " << error
                  << " Accuracy: " << accuracy << " Epoch time: " << epoch_time
                  << "s ETA: " << epoch_time * (epochs - epoch - 1) << "s\033[K" << std::endl;
    }
}

float DenseNetwork::error(std::vector<std::vector<float>> inputs, std::vector<int> targets) {
    float error = 0;

    for (int i = 0; i < inputs.size(); i++) {
        std::vector<float> target_vector(this->layer_sizes[this->layer_sizes.size() - 1], 0);
        target_vector[targets[i]] = 1;

        error += mse(this->predict(inputs[i]), target_vector);
    }

    return error / inputs.size() * 2;
}

float DenseNetwork::accuracy(std::vector<std::vector<float>> inputs, std::vector<int> targets) {
    int correct = 0;

    for (int i = 0; i < inputs.size(); i++) {
        std::vector<float> outputs = this->predict(inputs[i]);
        int max_index = 0;

        for (int j = 0; j < outputs.size(); j++) {
            if (outputs[j] > outputs[max_index]) {
                max_index = j;
            }
        }

        if (max_index == targets[i]) {
            correct++;
        }
    }

    return (float)correct / inputs.size();
}

float mse(float output, float target) { return pow(output - target, 2); }

float mse(std::vector<float> output, std::vector<float> target) {
    float error = 0.0;

    for (int i = 0; i < output.size(); i++) {
        error += mse(output[i], target[i]);
    }

    return error / output.size();
}
