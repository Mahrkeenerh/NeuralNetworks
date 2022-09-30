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

std::vector<double> DenseNetwork::predict(std::vector<double> input) {
    std::vector<double> output = input;

    for (int i = 0; i < this->layers.size(); i++) {
        output = this->layers[i].predict(output);
    }

    return output;
}

void DenseNetwork::fit(Dataset1D dataset, int epochs, double learning_rate, int epoch_stats) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        clock_t start, end;
        start = clock();
        std::cout << "..." << std::endl;

        for (int i = 0; i < dataset.train_data.size(); i++) {
            if (i % 100 == 0) {
                std::cout << "\033[F"
                          << "data: " << i + 1 << "/" << dataset.train_data.size() << "\033[K"
                          << std::endl;
            }

            // Backpropagation
            std::vector<double> outputs = this->predict(dataset.train_data[i]);
            std::vector<double> target_vector(this->layer_sizes[this->layer_sizes.size() - 1], 0);
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

            // // Print outputs, targets, and errors
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
                std::vector<double> l_inputs =
                    (l_i == 0) ? dataset.train_data[i] : this->layers[l_i - 1].outputs;

                for (int n_i = 0; n_i < this->layers[l_i].output_size; n_i++) {
                    for (int w_i = 1; w_i < this->layers[l_i].input_size + 1; w_i++) {
                        // double input = (w_i == 0) ? 1 : l_inputs[w_i - 1];
                        this->layers[l_i].weights[n_i][w_i] -=
                            this->layers[l_i].errors[n_i] * learning_rate * l_inputs[w_i - 1];
                    }
                    this->layers[l_i].weights[n_i][0] -= this->layers[l_i].errors[n_i] * learning_rate;
                }
            }
        }

        // double error = this->error(dataset.test_data, dataset.test_labels);

        // Stats
        if (epoch % epoch_stats == epoch_stats - 1 || epoch == epochs - 1 || epoch == 0) {
            double train_accuracy = this->accuracy(dataset.train_data, dataset.train_labels);
            double test_accuracy = this->accuracy(dataset.test_data, dataset.test_labels);

            end = clock();
            double epoch_time = (double)(end - start) / CLOCKS_PER_SEC;

            std::cout << "\033[FEpoch " << epoch + 1 << "/" << epochs
                      << " | Train Accuracy: " << train_accuracy << " | Test Accuracy: " << test_accuracy
                      << " | Epoch time: " << epoch_time
                      << "s | ETA: " << epoch_time * (epochs - epoch - 1) << "s\033[K" << std::endl;
        } else {
            end = clock();
            double epoch_time = (double)(end - start) / CLOCKS_PER_SEC;

            std::cout << "\033[FEpoch " << epoch + 1 << "/" << epochs << " | Epoch time: " << epoch_time
                      << "s | ETA: " << epoch_time * (epochs - epoch - 1) << "s\033[K" << std::endl;
        }
    }
}

double DenseNetwork::error(std::vector<std::vector<double>> inputs, std::vector<int> targets) {
    double error = 0;

    for (int i = 0; i < inputs.size(); i++) {
        std::vector<double> target_vector(this->layer_sizes[this->layer_sizes.size() - 1], 0);
        target_vector[targets[i]] = 1;

        error += mse(this->predict(inputs[i]), target_vector);
    }

    return error / inputs.size() * 2;
}

double DenseNetwork::accuracy(std::vector<std::vector<double>> inputs, std::vector<int> targets) {
    int correct = 0;

    for (int i = 0; i < inputs.size(); i++) {
        std::vector<double> outputs = this->predict(inputs[i]);
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

    return (double)correct / inputs.size();
}

double mse(double output, double target) { return pow(output - target, 2); }

double mse(std::vector<double> output, std::vector<double> target) {
    double error = 0.0;

    for (int i = 0; i < output.size(); i++) {
        error += mse(output[i], target[i]);
    }

    return error / output.size();
}
