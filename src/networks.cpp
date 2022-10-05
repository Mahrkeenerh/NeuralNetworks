#include "networks.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

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

void DenseNetwork::fit(Dataset1D dataset, int epochs, double learning_rate, bool verbose) {
    clock_t train_start = clock();

    for (int epoch = 0; epoch < epochs; epoch++) {
        clock_t epoch_start;
        epoch_start = clock();

        // Visual loading bar
        int progress, data_padding;
        int correct = 0;
        if (verbose) {
            progress = 0;
            data_padding = std::to_string(dataset.train_size).length() - std::to_string(0).length() + 1;

            std::cout << std::string(data_padding, ' ') << "0/" << dataset.train_size << " ["
                      << std::string(50, '-') << "]" << std::endl;
        }

        // Random idxs
        std::vector<int> idxs(dataset.train_size);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::random_shuffle(idxs.begin(), idxs.end());

        for (int i = 0; i < dataset.train_size; i++) {
            if (verbose && (int)((double)(i + 1) / dataset.train_size * 50) > progress) {
                progress = (int)((double)(i + 1) / dataset.train_size * 50);
                double acc = (double)correct / (i + 1);
                data_padding =
                    std::to_string(dataset.train_size).length() - std::to_string(i + 1).length() + 1;
                double epoch_eta = (double)(clock() - epoch_start) / CLOCKS_PER_SEC / (i + 1) *
                                   (dataset.train_size - i - 1);

                std::cout << "\033[F" << std::string(data_padding, ' ') << i + 1 << "/"
                          << dataset.train_size << " [" << std::string(progress - 1, '=') << ">"
                          << std::string(50 - progress, '-') << "] Train accuracy: " << acc
                          << " | Epoch ETA: " << epoch_eta << "s\033[K" << std::endl;
            }

            std::vector<double> outputs = this->predict(dataset.train_data[idxs[i]]);
            std::vector<double> target_vector(this->layer_sizes[this->layer_sizes.size() - 1], 0);
            target_vector[dataset.train_labels[idxs[i]]] = 1;

            // Add if correct
            if (std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end())) ==
                dataset.train_labels[idxs[i]]) {
                correct++;
            }

            // Calculate output layer errors
            for (int n_i = 0; n_i < this->layers[this->layers.size() - 1].output_size; n_i++) {
                this->layers[this->layers.size() - 1].errors[n_i] =
                    (outputs[n_i] - target_vector[n_i]) *
                    this->layers[this->layers.size() - 1].derivative(outputs[n_i]);
            }

            // Backpropagate errors
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

            // Update weights
            for (int l_i = 0; l_i < this->layers.size(); l_i++) {
                std::vector<double> l_inputs =
                    (l_i == 0) ? dataset.train_data[idxs[i]] : this->layers[l_i - 1].outputs;

                for (int n_i = 0; n_i < this->layers[l_i].output_size; n_i++) {
                    for (int w_i = 1; w_i < this->layers[l_i].input_size + 1; w_i++) {
                        this->layers[l_i].weights[n_i][w_i] -=
                            this->layers[l_i].errors[n_i] * learning_rate * l_inputs[w_i - 1];
                    }
                    this->layers[l_i].weights[n_i][0] -= this->layers[l_i].errors[n_i] * learning_rate;
                }
            }
        }

        // Stats
        if (verbose) {
            double train_accuracy = (double)correct / dataset.train_size;
            double test_accuracy = this->accuracy(dataset.test_data, dataset.test_labels);
            clock_t epoch_end = clock();

            int epoch_padding = std::to_string(epochs).length() - std::to_string(epoch + 1).length();
            double epoch_time = (double)(epoch_end - epoch_start) / CLOCKS_PER_SEC;
            double elapsed_time = (double)(epoch_end - train_start) / CLOCKS_PER_SEC;
            double eta_s = (elapsed_time / (epoch + 1)) * (epochs - epoch - 1);

            std::cout << "\033[FEpoch " << std::string(epoch_padding, ' ') << epoch + 1 << "/" << epochs
                      << " | Train Accuracy: " << train_accuracy << " | Test Accuracy: " << test_accuracy
                      << " | Epoch time: " << epoch_time << "s | ETA: " << eta_s << "s\033[K"
                      << std::endl;
        }
    }
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
