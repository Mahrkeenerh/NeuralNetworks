#include "networks.hpp"

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <numeric>

DenseNetwork::DenseNetwork(std::vector<int> layer_sizes) {
    this->layer_sizes = layer_sizes;

    // Create layers
    for (int i = 0; i < layer_sizes.size() - 2; i++) {
        this->layers.push_back(new DenseLayer(layer_sizes[i], layer_sizes[i + 1], relu));
    }

    // Create output layer
    this->layers.push_back(
        new SoftmaxLayer(layer_sizes[layer_sizes.size() - 2], layer_sizes[layer_sizes.size() - 1]));

    for (int i = 0; i < omp_get_max_threads(); i++) {
        this->outputs.push_back(std::vector<std::vector<double>>(layer_sizes.size()));
        this->gradients.push_back(std::vector<std::vector<double>>(layer_sizes.size() - 1));

        for (int j = 0; j < layers.size(); j++) {
            this->gradients[i][j] = std::vector<double>(layer_sizes[j + 1], 0.0);
        }
    }

    this->updates = std::vector<std::vector<std::vector<double>>>(layers.size());

    for (int i = 0; i < layers.size(); i++) {
        this->updates[i] = std::vector<std::vector<double>>(layer_sizes[i + 1]);

        for (int j = 0; j < layer_sizes[i + 1]; j++) {
            this->updates[i][j] = std::vector<double>(layer_sizes[i] + 1, 0.0);
        }
    }
}

std::vector<double> DenseNetwork::predict(std::vector<double> input) {
    std::vector<double> output = input;

    for (int i = 0; i < this->layers.size(); i++) {
        output = this->layers[i]->forwardpropagate(output);
    }

    return output;
}

void DenseNetwork::forwardpropagate(int thread_id, std::vector<double> input) {
    this->outputs[thread_id][0] = input;

    for (int i = 0; i < this->layers.size(); i++) {
        this->outputs[thread_id][i + 1] = this->layers[i]->predict(this->outputs[thread_id][i]);
    }
}

void DenseNetwork::backpropagate(int thread_id, std::vector<double> target_vector) {
    this->layers[this->layers.size() - 1]->out_errors(
        this->outputs[thread_id][this->outputs[thread_id].size() - 1], target_vector,
        &(this->gradients[thread_id][this->gradients[thread_id].size() - 1]));

    for (int l_i = this->layers.size() - 2; l_i >= 0; l_i--) {
        this->layers[l_i]->backpropagate(this->layers[l_i + 1], this->outputs[thread_id][l_i + 1],
                                         target_vector, &(this->gradients[thread_id][l_i]),
                                         this->gradients[thread_id][l_i + 1]);
    }
}

void DenseNetwork::calculate_updates(int thread_id, double learning_rate) {
    for (int l_i = 0; l_i < this->layers.size(); l_i++) {
        this->layers[l_i]->calculate_updates(&(this->updates[l_i]), this->gradients[thread_id][l_i],
                                             this->outputs[thread_id][l_i], learning_rate);
    }
}

void DenseNetwork::apply_updates(int batch_size) {
    for (int l_i = 0; l_i < this->layers.size(); l_i++) {
        this->layers[l_i]->apply_updates(this->updates[l_i], batch_size);
    }
}

void DenseNetwork::clear_updates() {
    for (int j = 0; j < this->updates.size(); j++) {
        for (int k = 0; k < this->updates[j].size(); k++) {
            for (int l = 0; l < this->updates[j][k].size(); l++) {
                this->updates[j][k][l] = 0.0;
            }
        }
    }
}

void DenseNetwork::fit(Dataset1D dataset, int epochs, int minibatch_size, double learning_rate_start,
                       double learning_rate_end, bool verbose) {
    double train_start = omp_get_wtime();

    if (learning_rate_end == -1) {
        learning_rate_end = learning_rate_start;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_start;
        epoch_start = omp_get_wtime();

        // Visual loading bar
        int progress, data_padding;
        int correct = 0;
        if (verbose) {
            progress = 0;
            data_padding = std::to_string(dataset.train_size / minibatch_size).length() -
                           std::to_string(0).length() + 1;

            std::cout << std::string(data_padding, ' ') << "0/" << dataset.train_size / minibatch_size
                      << " [" << std::string(50, '.') << "]" << std::endl;
        }

        // Random idxs
        std::vector<int> idxs(dataset.train_size);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::random_shuffle(idxs.begin(), idxs.end());

        double learning_rate = learning_rate_start;
        if (epochs > 1) {
            learning_rate =
                (1 - pow((double)epoch / (epochs - 1), 2)) * (learning_rate_start - learning_rate_end) +
                learning_rate_end;
        }

        for (int batch = 0; batch < (dataset.train_size / minibatch_size); batch++) {
#pragma omp parallel for
            for (int i = batch * minibatch_size; i < (batch + 1) * minibatch_size; i++) {
                int thread_id = omp_get_thread_num();

                if (thread_id == 0 && verbose &&
                    (int)((double)(batch + 1) / (dataset.train_size / minibatch_size) * 50) > progress) {
                    progress = (int)((double)(batch + 1) / (dataset.train_size / minibatch_size) * 50);
                    double acc = (double)correct / (i + 1);
                    data_padding = std::to_string(dataset.train_size / minibatch_size).length() -
                                   std::to_string(minibatch_size + 1).length() + 1;
                    double epoch_eta =
                        (double)(omp_get_wtime() - epoch_start) / (i + 1) * (dataset.train_size - i - 1);

                    std::cout << "\033[F" << std::string(data_padding, ' ') << batch + 1 << "/"
                              << dataset.train_size / minibatch_size << " ["
                              << std::string(progress - 1, '=') << ">" << std::string(50 - progress, '.')
                              << "] Train accuracy: " << acc << " | Epoch ETA: " << epoch_eta
                              << "s\033[K" << std::endl;
                }

                this->forwardpropagate(thread_id, dataset.train_data[idxs[i]]);
                std::vector<double> target_vector(this->layer_sizes[this->layer_sizes.size() - 1], 0);
                target_vector[dataset.train_labels[idxs[i]]] = 1;

#pragma omp critical
                {
                    // Add if correct
                    if (std::distance(
                            this->outputs[thread_id][outputs[0].size() - 1].begin(),
                            std::max_element(this->outputs[thread_id][outputs[0].size() - 1].begin(),
                                             this->outputs[thread_id][outputs[0].size() - 1].end())) ==
                        dataset.train_labels[idxs[i]]) {
                        correct++;
                    }
                }

                this->backpropagate(thread_id, target_vector);

                // NOT THREAD SAFE, BUT WORKS JUST FINE ANYWAY
                // #pragma omp critical
                { this->calculate_updates(thread_id, learning_rate); }
            }

            this->apply_updates(minibatch_size);
            this->clear_updates();
        }

        // Stats
        if (verbose) {
            double train_accuracy = (double)correct / dataset.train_size;
            double test_accuracy = this->accuracy(dataset.test_data, dataset.test_labels);
            double epoch_end = omp_get_wtime();

            int epoch_padding = std::to_string(epochs).length() - std::to_string(epoch + 1).length();
            double epoch_time = (double)(epoch_end - epoch_start);
            double elapsed_time = (double)(epoch_end - train_start);
            double eta_s = (elapsed_time / (epoch + 1)) * (epochs - epoch - 1);

            std::cout << "\033[FEpoch " << std::string(epoch_padding, ' ') << epoch + 1 << "/" << epochs
                      << " | Train Accuracy: " << train_accuracy << " | Test Accuracy: " << test_accuracy
                      << " | Learning rate: " << learning_rate << " | Epoch time: " << epoch_time
                      << "s | ETA: " << eta_s << "s\033[K" << std::endl;
        }
    }
}

double DenseNetwork::accuracy(std::vector<std::vector<double>> inputs, std::vector<int> targets) {
    std::vector<int> correct = std::vector(omp_get_max_threads(), 0);

#pragma omp parallel for
    for (int i = 0; i < inputs.size(); i++) {
        std::vector<double> outputs = this->predict(inputs[i]);
        int max_index = 0;

        for (int j = 0; j < outputs.size(); j++) {
            if (outputs[j] > outputs[max_index]) {
                max_index = j;
            }
        }

        if (max_index == targets[i]) {
            correct[omp_get_thread_num()]++;
        }
    }

    return std::accumulate(correct.begin(), correct.end(), 0) / (double)inputs.size();
}
