#include "networks.hpp"

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <numeric>

DenseNetwork::DenseNetwork(std::vector<layers::Layer*> layers) {
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
    this->layers[0]->predict(thread_id, input);

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

// First 3(4) layers are ignored
void DenseNetwork::backpropagate(int thread_id, std::vector<double> target_vector) {
    this->layers[this->size - 1]->out_errors(thread_id, target_vector);

    for (int l_i = this->size - 1; l_i >= 5; l_i--) {
        this->layers[l_i]->backpropagate(thread_id);
    }
}

// First 3(4) layers are ignored
void DenseNetwork::calculate_updates(int thread_id, double learning_rate) {
    for (int l_i = 4; l_i < this->size; l_i++) {
        this->layers[l_i]->calculate_updates(thread_id, learning_rate);
    }
}

// First 3(4) layers are ignored
void DenseNetwork::apply_updates(int batch_size) {
    for (int l_i = 4; l_i < this->size; l_i++) {
        this->layers[l_i]->apply_updates(batch_size);
    }
}

// First 3(4) layers are ignored
void DenseNetwork::clear_updates() {
    for (int l_i = 4; l_i < this->size; l_i++) {
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

void DenseNetwork::fit(Dataset1D dataset, int epochs, int minibatch_size,
                       LearningRateScheduler* learn_scheduler, bool verbose) {
    double train_start = omp_get_wtime();
    double epoch_start, epoch_end, epoch_eta, epoch_time, elapsed_time, eta_s;

    int padding, progress, correct;

    double learning_rate;
    double train_acc, valid_acc;

    std::vector<int> train_indices(dataset.train_size);

    learn_scheduler->network_setup(epochs);

    for (int epoch = 0; epoch < epochs; epoch++) {
        epoch_start = omp_get_wtime();
        correct = 0;

        // Visual loading bar
        if (verbose) {
            progress = 0;
            padding = std::to_string(dataset.train_size / minibatch_size).length() -
                      std::to_string(0).length() + 1;

            std::cout << std::string(padding, ' ') << "0/" << dataset.train_size / minibatch_size << " ["
                      << std::string(50, '.') << "]" << std::endl;
        }

        // Shuffle training data
        std::iota(train_indices.begin(), train_indices.end(), 0);
        std::random_shuffle(train_indices.begin(), train_indices.end());

        learning_rate = learn_scheduler->get_learning_rate(epoch);

        for (int batch = 0; batch < (dataset.train_size / minibatch_size); batch++) {
            before_batch();

#pragma omp parallel for
            for (int i = batch * minibatch_size; i < (batch + 1) * minibatch_size; i++) {
                int thread_id = omp_get_thread_num();

                if (thread_id == 0 && verbose &&
                    (int)((double)(batch + 1) / (dataset.train_size / minibatch_size) * 50) > progress) {
                    progress = (int)((double)(batch + 1) / (dataset.train_size / minibatch_size) * 50);
                    train_acc = (double)correct / (i + 1);
                    padding = std::to_string(dataset.train_size / minibatch_size).length() -
                              std::to_string(batch + 1).length() + 1;
                    epoch_eta =
                        (double)(omp_get_wtime() - epoch_start) / (i + 1) * (dataset.train_size - i - 1);

                    std::cout << "\033[F" << std::string(padding, ' ') << batch + 1 << "/"
                              << dataset.train_size / minibatch_size << " ["
                              << std::string(progress - 1, '=') << ">" << std::string(50 - progress, '.')
                              << "] Train acc: " << train_acc << " | Epoch ETA: " << epoch_eta
                              << "s\033[K" << std::endl;
                }

                this->forwardpropagate(dataset.train_data[train_indices[i]], thread_id);
                std::vector<double> target_vector(this->layers[this->size - 1]->output_shape[0], 0);
                target_vector[dataset.train_labels[train_indices[i]]] = 1;

#pragma omp critical
                {
                    std::vector<double> output = this->layers[this->size - 1]->get_outputs(thread_id);
                    // Add if correct
                    if (std::distance(output.begin(), std::max_element(output.begin(), output.end())) ==
                        dataset.train_labels[train_indices[i]]) {
                        correct++;
                    }
                }

                this->backpropagate(thread_id, target_vector);

                // // NOT THREAD SAFE, BUT WORKS JUST FINE ANYWAY
                // // #pragma omp critical
                { this->calculate_updates(thread_id, learning_rate); }
            }

            this->apply_updates(minibatch_size);
            this->clear_updates();
        }

        // Stats
        if (verbose) {
            train_acc = (double)correct / dataset.train_size;
            valid_acc = this->accuracy(dataset.valid_data, dataset.valid_labels);
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

double DenseNetwork::accuracy(std::vector<std::vector<double>> inputs, std::vector<int> labels) {
    std::vector<int> correct = std::vector(omp_get_max_threads(), 0);

#pragma omp parallel for
    for (int i = 0; i < (int)inputs.size(); i++) {
        int thread_id = omp_get_thread_num();
        std::vector<double> outputs = this->predict(inputs[i], thread_id);
        int max_index = 0;

        for (int j = 0; j < (int)outputs.size(); j++) {
            if (outputs[j] > outputs[max_index]) {
                max_index = j;
            }
        }

        if (max_index == labels[i]) {
            correct[omp_get_thread_num()]++;
        }
    }

    return std::accumulate(correct.begin(), correct.end(), 0) / (double)inputs.size();
}

void DenseNetwork::save_predictions(Dataset1D dataset) {
    std::vector<int> predictions(dataset.train_size + dataset.valid_size);

#pragma omp parallel for
    for (int i = 0; i < dataset.train_size; i++) {
        std::vector<double> outputs = this->predict(dataset.train_data[i], omp_get_thread_num());
        int max_index = 0;

        for (int j = 0; j < (int)outputs.size(); j++) {
            if (outputs[j] > outputs[max_index]) {
                max_index = j;
            }
        }

        predictions[i] = max_index;
    }

#pragma omp parallel for
    for (int i = 0; i < dataset.valid_size; i++) {
        std::vector<double> outputs = this->predict(dataset.valid_data[i], omp_get_thread_num());
        int max_index = 0;

        for (int j = 0; j < (int)outputs.size(); j++) {
            if (outputs[j] > outputs[max_index]) {
                max_index = j;
            }
        }

        predictions[i + dataset.train_size] = max_index;
    }

    std::ofstream predictions_file;
    predictions_file.open("train_predictions.csv");

    for (int i = 0; i < (int)predictions.size(); i++) {
        predictions_file << predictions[i] << std::endl;
    }

    predictions_file.close();

    predictions.resize(dataset.test_size);

#pragma omp parallel for
    for (int i = 0; i < dataset.test_size; i++) {
        std::vector<double> outputs = this->predict(dataset.test_data[i], omp_get_thread_num());
        int max_index = 0;

        for (int j = 0; j < (int)outputs.size(); j++) {
            if (outputs[j] > outputs[max_index]) {
                max_index = j;
            }
        }

        predictions[i] = max_index;
    }

    predictions_file.open("test_predictions.csv");

    for (int i = 0; i < (int)predictions.size(); i++) {
        predictions_file << predictions[i] << std::endl;
    }

    predictions_file.close();
}

ConstantLearningRate::ConstantLearningRate(double learning_rate) { this->learning_rate = learning_rate; }

double ConstantLearningRate::get_learning_rate(int epoch) { return this->learning_rate; }

LinearLearningRate::LinearLearningRate(double learning_rate_start, double learning_rate_end) {
    this->learning_rate_start = learning_rate_start;
    this->learning_rate_end = learning_rate_end;
}

void LinearLearningRate::network_setup(int epochs) {
    this->learning_rate_slope = (learning_rate_end - learning_rate_start) / (epochs - 1);
}

double LinearLearningRate::get_learning_rate(int epoch) {
    return learning_rate_start + learning_rate_slope * epoch;
}

HalvingLearningRate::HalvingLearningRate(double learning_rate) {
    this->learning_rate = 2 * learning_rate;
}

double HalvingLearningRate::get_learning_rate(int epoch) {
    this->learning_rate /= 2;
    return this->learning_rate;
}

CustomSquareLearningRate::CustomSquareLearningRate(double learning_rate_start, double learning_rate_end,
                                                   double slope) {
    this->learning_rate_start = learning_rate_start;
    this->learning_rate_end = learning_rate_end;
    this->slope = slope;
}

void CustomSquareLearningRate::network_setup(int epochs) { this->epochs = epochs; }

double CustomSquareLearningRate::get_learning_rate(int epoch) {
    if (epochs == 1) {
        return this->learning_rate_start;
    }

    return (1 - pow((double)epoch / (this->epochs - 1), this->slope)) *
               (this->learning_rate_start - this->learning_rate_end) +
           this->learning_rate_end;
}
