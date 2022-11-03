#include "networks.hpp"

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <numeric>

DenseNetwork::DenseNetwork(std::vector<Layer*> layers) {
    this->layers = layers;
    this->size = layers.size();

    int num_threads = omp_get_max_threads();

    this->layers[0]->setup(nullptr, this->layers[1], num_threads);

    for (int i = 1; i < this->size - 1; i++) {
        this->layers[i]->setup(this->layers[i - 1], this->layers[i + 1], num_threads);
    }

    this->layers[this->size - 1]->setup(this->layers[this->size - 2], nullptr, num_threads);
}

std::vector<double> DenseNetwork::predict(std::vector<double> input, int thread_id) {
    this->layers[0]->predict(input, thread_id);

    for (int i = 1; i < this->size; i++) {
        this->layers[i]->predict(thread_id);
    }

    return this->layers[this->size - 1]->get_outputs({thread_id});
}

void DenseNetwork::forwardpropagate(std::vector<double> input, int thread_id) {
    this->layers[0]->forwardpropagate(input, thread_id);

    for (int i = 1; i < this->size; i++) {
        this->layers[i]->forwardpropagate(thread_id);
    }
}

void DenseNetwork::backpropagate(int thread_id, std::vector<double> target_vector) {
    this->layers[this->size - 1]->out_errors(thread_id, target_vector);

    for (int l_i = this->size - 2; l_i >= 1; l_i--) {
        this->layers[l_i]->backpropagate(thread_id);
    }
}

void DenseNetwork::calculate_updates(int thread_id, double learning_rate) {
    for (int l_i = 1; l_i < this->size; l_i++) {
        this->layers[l_i]->calculate_updates(thread_id, learning_rate);
    }
}

void DenseNetwork::apply_updates(int batch_size) {
    for (int l_i = 1; l_i < this->size; l_i++) {
        this->layers[l_i]->apply_updates(batch_size);
    }
}

void DenseNetwork::clear_updates() {
    for (int l_i = 1; l_i < this->size; l_i++) {
        this->layers[l_i]->clear_updates();
    }
}

void DenseNetwork::before_batch() {
    for (int thread_id = 0; thread_id < omp_get_max_threads(); thread_id++) {
        for (int l_i = 1; l_i < this->size; l_i++) {
            this->layers[l_i]->before_batch(thread_id);
        }
    }
}

int get_train_id(int split_step, int train_size, int valid_size, int i) {
    int valid_start = valid_size * (split_step - 1);
    int valid_end = valid_start + valid_size;

    if (i < valid_start) {
        return i;
    } else {
        return (i + valid_size) % (train_size + valid_size);
    }
}

void DenseNetwork::fit(Dataset1D dataset, double split, int epochs, int minibatch_size,
                       double learning_rate_start, double learning_rate_end, bool verbose) {
    double train_start = omp_get_wtime();
    double epoch_start, epoch_end, epoch_eta, epoch_time, elapsed_time, eta_s;

    int padding, progress, correct;

    double learning_rate;
    double train_acc, valid_acc;

    if (learning_rate_end == -1) {
        learning_rate_end = learning_rate_start;
    }

    int split_step = 0;
    int train_size = dataset.train_size * (1 - split);
    int valid_size = dataset.train_size - train_size;

    std::vector<std::vector<double>> data = dataset.train_data;
    std::vector<int> labels = dataset.train_labels;

    std::vector<int> idxs(train_size);

    for (int epoch = 0; epoch < epochs; epoch++) {
        epoch_start = omp_get_wtime();
        correct = 0;

        // Visual loading bar
        if (verbose) {
            progress = 0;
            padding =
                std::to_string(train_size / minibatch_size).length() - std::to_string(0).length() + 1;

            std::cout << std::string(padding, ' ') << "0/" << train_size / minibatch_size << " ["
                      << std::string(50, '.') << "]" << std::endl;
        }

        // Cross-validation split
        if (split_step * epochs * split <= epoch) {
            split_step++;
        }

        // Shuffle data
        std::iota(idxs.begin(), idxs.end(), 0);
        std::random_shuffle(idxs.begin(), idxs.end());

        learning_rate = learning_rate_start;
        if (epochs > 1) {
            learning_rate =
                (1 - pow((double)epoch / (epochs - 1), 2)) * (learning_rate_start - learning_rate_end) +
                learning_rate_end;
        }

        for (int batch = 0; batch < (train_size / minibatch_size); batch++) {
            before_batch();

#pragma omp parallel for
            for (int i = batch * minibatch_size; i < (batch + 1) * minibatch_size; i++) {
                int thread_id = omp_get_thread_num();

                if (thread_id == 0 && verbose &&
                    (int)((double)(batch + 1) / (train_size / minibatch_size) * 50) > progress) {
                    progress = (int)((double)(batch + 1) / (train_size / minibatch_size) * 50);
                    train_acc = (double)correct / (i + 1);
                    padding = std::to_string(train_size / minibatch_size).length() -
                              std::to_string(minibatch_size + 1).length() + 1;
                    epoch_eta = (double)(omp_get_wtime() - epoch_start) / (i + 1) * (train_size - i - 1);

                    std::cout << "\033[F" << std::string(padding, ' ') << batch + 1 << "/"
                              << train_size / minibatch_size << " [" << std::string(progress - 1, '=')
                              << ">" << std::string(50 - progress, '.') << "] Train acc: " << train_acc
                              << " | Epoch ETA: " << epoch_eta << "s\033[K" << std::endl;
                }

                this->forwardpropagate(data[get_train_id(split_step, train_size, valid_size, idxs[i])],
                                       thread_id);
                std::vector<double> target_vector(this->layers[this->size - 1]->output_size, 0);
                target_vector[labels[get_train_id(split_step, train_size, valid_size, idxs[i])]] = 1;

#pragma omp critical
                {
                    std::vector<double> output = this->layers[this->size - 1]->get_outputs({thread_id});
                    // Add if correct
                    if (std::distance(output.begin(), std::max_element(output.begin(), output.end())) ==
                        labels[get_train_id(split_step, train_size, valid_size, idxs[i])]) {
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
            train_acc = (double)correct / train_size;
            valid_acc = this->valid_accuracy(data, labels, split_step, valid_size);
            epoch_end = omp_get_wtime();

            padding = std::to_string(epochs).length() - std::to_string(epoch + 1).length();
            epoch_time = (double)(epoch_end - epoch_start);
            elapsed_time = (double)(epoch_end - train_start);
            eta_s = (elapsed_time / (epoch + 1)) * (epochs - epoch - 1);

            std::cout << "\033[FEpoch " << std::string(padding, ' ') << epoch + 1 << "/" << epochs
                      << " | Train Acc: " << train_acc << " | Valid Acc: " << valid_acc
                      << " | Learning rate: " << learning_rate << " | Time: " << epoch_time
                      << "s | ETA: " << eta_s << "s\033[K" << std::endl;
        }
    }
}

double DenseNetwork::accuracy(std::vector<std::vector<double>> inputs, std::vector<int> targets) {
    std::vector<int> correct = std::vector(omp_get_max_threads(), 0);

#pragma omp parallel for
    for (int i = 0; i < inputs.size(); i++) {
        int thread_id = omp_get_thread_num();
        std::vector<double> outputs = this->predict(inputs[i], thread_id);
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

double DenseNetwork::valid_accuracy(std::vector<std::vector<double>> inputs, std::vector<int> targets,
                                    int split_step, int valid_size) {
    std::vector<int> correct = std::vector(omp_get_max_threads(), 0);

    int valid_start = valid_size * (split_step - 1);
    int valid_end = valid_start + valid_size;

#pragma omp parallel for
    for (int i = valid_start; i < valid_end; i++) {
        int thread_id = omp_get_thread_num();
        std::vector<double> outputs = this->predict(inputs[i], thread_id);
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

    return std::accumulate(correct.begin(), correct.end(), 0) / (double)valid_size;
}
