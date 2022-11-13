#include "datasets.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

Dataset1D::Dataset1D(double val_split, bool normalize, double noise) {
    this->train_size = 60000 * (1 - val_split);
    this->valid_size = 60000 * val_split;
    this->test_size = 10000;

    this->train_data = std::vector<std::vector<double>>(this->train_size);
    this->train_labels = std::vector<int>(this->train_size);

    this->valid_data = std::vector<std::vector<double>>(this->valid_size);
    this->valid_labels = std::vector<int>(this->valid_size);

    this->test_data = std::vector<std::vector<double>>(this->test_size);
    this->test_labels = std::vector<int>(this->test_size);

    this->load_data();

    if (normalize) {
        this->normalize_data();
    }

    if (noise != -1) {
        this->noise_data(noise);
    }
}

void Dataset1D::load_data() {
    std::string line;
    std::vector<double> row(784);
    std::string cell;
    int j;

    std::ifstream train_data_file("data/fashion_mnist_train_vectors.csv");

    // Load training data
    for (int i = 0; i < this->train_size; i++) {
        std::getline(train_data_file, line);
        std::stringstream line_stream(line);

        j = 0;

        while (std::getline(line_stream, cell, ',')) {
            row[j++] = std::stof(cell) / 255;
        }

        this->train_data[i] = row;
    }

    for (int i = 0; i < this->valid_size; i++) {
        std::getline(train_data_file, line);
        std::stringstream line_stream(line);

        j = 0;

        while (std::getline(line_stream, cell, ',')) {
            row[j++] = std::stof(cell) / 255;
        }

        this->valid_data[i] = row;
    }

    std::ifstream train_labels_file("data/fashion_mnist_train_labels.csv");

    // Load training labels
    for (int i = 0; i < this->train_size; i++) {
        std::getline(train_labels_file, line);
        this->train_labels[i] = std::stoi(line);
    }

    for (int i = 0; i < this->valid_size; i++) {
        std::getline(train_labels_file, line);
        this->valid_labels[i] = std::stoi(line);
    }

    std::ifstream test_data_file("data/fashion_mnist_test_vectors.csv");

    // Load test data
    for (int i = 0; i < this->test_size; i++) {
        std::getline(test_data_file, line);
        std::stringstream line_stream(line);

        j = 0;

        while (std::getline(line_stream, cell, ',')) {
            row[j++] = std::stof(cell) / 255;
        }

        this->test_data[i] = row;
    }

    std::ifstream test_labels_file("data/fashion_mnist_test_labels.csv");

    // Load validation labels
    for (int i = 0; i < this->test_size; i++) {
        std::getline(test_labels_file, line);
        this->test_labels[i] = std::stoi(line);
    }
}

// Ugly and duplicate, but it works a little
void Dataset1D::noise_data(double noise_strength) {
    // Add gaussian noise to training data
    for (int i = 0; i < this->train_size; i++) {
        for (int j = 0; j < 784; j++) {
            double u1 = rand() / (double)RAND_MAX;
            double u2 = rand() / (double)RAND_MAX;
            double noise = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

            // Avoid infinite values
            while (noise == INFINITY || noise == -INFINITY) {
                u1 = rand() / (double)RAND_MAX;
                u2 = rand() / (double)RAND_MAX;
                noise = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            }

            this->train_data[i][j] += noise * noise_strength;
        }
    }
}

void Dataset1D::normalize_data() {
    int sample_size = this->train_data[0].size();
    // Calculate mean of each feature
    std::vector<double> mean_vector(sample_size, 0);

#pragma omp parallel for
    for (int i = 0; i < sample_size; i++) {
        double sum = 0;
        for (int j = 0; j < this->train_size; j++) {
            sum += this->train_data[j][i];
        }

        mean_vector[i] = sum / this->train_size;
    }

    // Calculate standard deviation of each feature
    std::vector<double> std_vector(sample_size, 0);

#pragma omp parallel for
    for (int i = 0; i < sample_size; i++) {
        double sum = 0;
        for (int j = 0; j < this->train_size; j++) {
            sum += (this->train_data[j][i] - mean_vector[i]) * (this->train_data[j][i] - mean_vector[i]);
        }

        std_vector[i] = sqrt(sum / this->train_size);
    }

    // Normalize training data
#pragma omp parallel for
    for (int i = 0; i < this->train_size; i++) {
        for (int j = 0; j < sample_size; j++) {
            this->train_data[i][j] = (this->train_data[i][j] - mean_vector[j]) / std_vector[j];
        }
    }

    // Normalize validation data
#pragma omp parallel for
    for (int i = 0; i < this->valid_size; i++) {
        for (int j = 0; j < sample_size; j++) {
            this->valid_data[i][j] = (this->valid_data[i][j] - mean_vector[j]) / std_vector[j];
        }
    }

    // Normalize test data
#pragma omp parallel for
    for (int i = 0; i < (int)this->test_data.size(); i++) {
        for (int j = 0; j < (int)this->test_data[i].size(); j++) {
            this->test_data[i][j] = (this->test_data[i][j] - mean_vector[j]) / std_vector[j];
        }
    }
}
