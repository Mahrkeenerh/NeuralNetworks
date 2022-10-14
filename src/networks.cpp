#include "networks.h"

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <numeric>

DenseNetwork::DenseNetwork(std::vector<int> layer_sizes) {
    this->layer_sizes = layer_sizes;

    // Create layers
    // for (int i = 0; i < layer_sizes.size() - 2; i++) {
    //     this->layers.push_back(new DenseLayer(layer_sizes[i], layer_sizes[i + 1], relu));
    // }
    // this->layers.push_back(new DropoutLayer(layer_sizes[0], 0.5));
    this->layers.push_back(new NoiseLayer(layer_sizes[0], 0.5, 0.1));
    this->layers.push_back(new DenseLayer(layer_sizes[0], layer_sizes[1], relu));
    // this->layers.push_back(new DenseLayer(layer_sizes[1], layer_sizes[1], relu));

    // Create output layer
    this->layers.push_back(new DenseLayer(layer_sizes[layer_sizes.size() - 2],
                                          layer_sizes[layer_sizes.size() - 1], sigmoid));
    // this->layers.push_back(
    //     new SoftmaxLayer(layer_sizes[layer_sizes.size() - 2], layer_sizes[layer_sizes.size() - 1]));
}

std::vector<double> DenseNetwork::predict(std::vector<double> input) {
    std::vector<double> output = input;

    for (int i = 0; i < this->layers.size(); i++) {
        output = this->layers[i]->predict(output);
    }

    return output;
}

std::vector<double> DenseNetwork::forwardpropagate(std::vector<double> input) {
    std::vector<double> output = input;

    for (int i = 0; i < this->layers.size(); i++) {
        output = this->layers[i]->forwardpropagate(output);
    }

    return output;
}

void DenseNetwork::backpropagate(std::vector<double> target_vector) {
    this->layers[this->layers.size() - 1]->out_errors(target_vector);

    for (int l_i = this->layers.size() - 2; l_i >= 0; l_i--) {
        this->layers[l_i]->backpropagate(this->layers[l_i + 1], target_vector);
    }
}

void DenseNetwork::update_weights(std::vector<double> input_data, double learning_rate) {
    for (int l_i = 0; l_i < this->layers.size(); l_i++) {
        std::vector<double> l_inputs = (l_i == 0) ? input_data : this->layers[l_i - 1]->outputs;

        this->layers[l_i]->update_weights(l_inputs, learning_rate);

        // for (int n_i = 0; n_i < this->layers[l_i].output_size; n_i++) {
        //     for (int w_i = 1; w_i < this->layers[l_i].input_size + 1; w_i++) {
        //         this->layers[l_i].weights[n_i][w_i] -=
        //             this->layers[l_i].errors[n_i] * learning_rate * l_inputs[w_i - 1];
        //     }
        //     this->layers[l_i].weights[n_i][0] -= this->layers[l_i].errors[n_i] * learning_rate;
        // }
    }
}

void DenseNetwork::fit(Dataset1D dataset, int epochs, double learning_rate, bool verbose) {
    double train_start = omp_get_wtime();

    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_start = omp_get_wtime();

        // Visual loading bar
        int progress, data_padding;
        int correct = 0;
        if (verbose) {
            progress = 0;
            data_padding = std::to_string(dataset.train_size).length() - std::to_string(0).length() + 1;

            std::cout << std::string(data_padding, ' ') << "0/" << dataset.train_size << " ["
                      << std::string(50, '.') << "]" << std::endl;
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
                double epoch_eta =
                    (omp_get_wtime() - epoch_start) / (i + 1) * (dataset.train_size - i - 1);

                std::cout << "\033[F" << std::string(data_padding, ' ') << i + 1 << "/"
                          << dataset.train_size << " [" << std::string(progress - 1, '=') << ">"
                          << std::string(50 - progress, '.') << "] Train accuracy: " << acc
                          << " | Epoch ETA: " << epoch_eta << "s\033[K" << std::endl;
            }

            std::vector<double> outputs = this->forwardpropagate(dataset.train_data[idxs[i]]);
            std::vector<double> target_vector(this->layer_sizes[this->layer_sizes.size() - 1], 0);
            target_vector[dataset.train_labels[idxs[i]]] = 1;

            // Add if correct
            if (std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end())) ==
                dataset.train_labels[idxs[i]]) {
                correct++;
            }

            this->backpropagate(target_vector);
            this->update_weights(dataset.train_data[idxs[i]], learning_rate);
        }

        // Stats
        if (verbose) {
            double train_accuracy = (double)correct / dataset.train_size;
            double test_accuracy = this->accuracy(dataset.test_data, dataset.test_labels);
            double epoch_end = omp_get_wtime();

            int epoch_padding = std::to_string(epochs).length() - std::to_string(epoch + 1).length();
            double epoch_time = epoch_end - epoch_start;
            double elapsed_time = epoch_end - train_start;
            double eta_s = (elapsed_time / (epoch + 1)) * (epochs - epoch - 1);

            std::cout << "\033[FEpoch " << std::string(epoch_padding, ' ') << epoch + 1 << "/" << epochs
                      << " | Train Accuracy: " << train_accuracy << " | Test Accuracy: " << test_accuracy
                      << " | Epoch time: " << epoch_time << "s | ETA: " << eta_s << "s\033[K"
                      << std::endl;
        }
    }
}

/*void DenseNetwork::fit(Dataset1D dataset, int epochs, double learning_rate, int batch_size,
                       bool verbose) {
    clock_t train_start = clock();

    for (int epoch = 0; epoch < epochs; epoch++) {
        clock_t epoch_start;
        epoch_start = clock();

        // Visual loading bar
        int progress, data_padding;
        int correct = 0;
        if (verbose) {
            progress = 0;
            data_padding = std::to_string(dataset.train_size / batch_size).length() -
                           std::to_string(0).length() + 1;

            std::cout << std::string(data_padding, ' ') << "0/" << dataset.train_size / batch_size
                      << " [" << std::string(50, '-') << "]" << std::endl;
        }

        // Random idxs
        std::vector<int> idxs(dataset.train_size);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::random_shuffle(idxs.begin(), idxs.end());

        for (int batch = 0; batch < (dataset.train_size / batch_size); batch++) {
            // Clear batch errors
            for (int l_i = 0; l_i < this->layers.size(); l_i++) {
                for (int n_i = 0; n_i < this->layers[l_i].output_size; n_i++) {
                    this->layers[l_i].batch_errors[n_i] = 0;
                }
            }

            for (int i = batch * batch_size; i < (batch + 1) * batch_size; i++) {
                if (verbose &&
                    (int)((double)(batch + 1) / (dataset.train_size / batch_size) * 50) > progress) {
                    progress = (int)((double)(batch + 1) / (dataset.train_size / batch_size) * 50);
                    double acc = (double)correct / (i + 1);
                    data_padding = std::to_string(dataset.train_size / batch_size).length() -
                                   std::to_string(batch + 1).length() + 1;
                    double epoch_eta = (double)(clock() - epoch_start) / CLOCKS_PER_SEC / (i + 1) *
                                       (dataset.train_size - i - 1);

                    std::cout << "\033[F" << std::string(data_padding, ' ') << batch + 1 << "/"
                              << dataset.train_size / batch_size << " ["
                              << std::string(progress - 1, '=') << ">" << std::string(50 - progress, '-')
                              << "] Train accuracy: " << acc << " | Epoch ETA: " << epoch_eta
                              << "s\033[K" << std::endl;
                }

                std::vector<double> outputs = this->predict(dataset.train_data[idxs[i]]);
                std::vector<double> target_vector(this->layer_sizes[this->layer_sizes.size() - 1], 0);
                target_vector[dataset.train_labels[idxs[i]]] = 1;

                // Add if correct
                if (std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end())) ==
                    dataset.train_labels[idxs[i]]) {
                    correct++;
                }

                this->backpropagate(outputs, target_vector);

                // Update batch errors
                for (int l_i = 0; l_i < this->layers.size(); l_i++) {
                    for (int n_i = 0; n_i < this->layers[l_i].output_size; n_i++) {
                        this->layers[l_i].batch_errors[n_i] += this->layers[l_i].errors[n_i];
                    }
                }
            }

            for (int l_i = 0; l_i < this->layers.size(); l_i++) {
                for (int n_i = 0; n_i < this->layers[l_i].output_size; n_i++) {
                    this->layers[l_i].errors[n_i] = this->layers[l_i].batch_errors[n_i] / batch_size;
                }
            }

            // this->update_weights(learning_rate);
            for (int i = batch * batch_size; i < (batch + 1) * batch_size; i++) {
                this->update_weights(dataset.train_data[idxs[i]], learning_rate);
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
}*/

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
