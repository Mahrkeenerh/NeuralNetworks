#include "datasets.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

Dataset1D::Dataset1D(int train_size, int test_size, bool normalize, double noise) {
    this->train_size = train_size;
    this->test_size = test_size;

    // Set sizes
    if (train_size == -1) {
        this->train_size = 60000;
    }
    if (test_size == -1) {
        this->test_size = 10000;
    }

    // Validate sizes
    if (train_size > 60000 || train_size < 0) {
        this->train_size = 60000;
    }
    if (test_size > 10000 || test_size < 0) {
        this->test_size = 10000;
    }

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

    this->train_data.resize(this->train_size);
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

    this->train_labels.resize(this->train_size);
    std::ifstream train_labels_file("data/fashion_mnist_train_labels.csv");

    // Load training labels
    for (int i = 0; i < this->train_size; i++) {
        std::getline(train_labels_file, line);
        this->train_labels[i] = std::stoi(line);
    }

    this->test_data.resize(this->test_size);
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

    this->test_labels.resize(this->test_size);
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
    // Calculate mean of each feature
    std::vector<double> mean_vector(this->train_data[0].size(), 0);

#pragma omp parallel for
    for (int i = 0; i < (int)this->train_data[0].size(); i++) {
        double sum = 0;
        for (int j = 0; j < (int)this->train_data.size(); j++) {
            sum += this->train_data[j][i];
        }

        mean_vector[i] = sum / this->train_data.size();
    }

    // Calculate standard deviation of each feature
    std::vector<double> std_vector(this->train_data[0].size(), 0);

#pragma omp parallel for
    for (int i = 0; i < (int)this->train_data[0].size(); i++) {
        double sum = 0;
        for (int j = 0; j < (int)this->train_data.size(); j++) {
            sum += (this->train_data[j][i] - mean_vector[i]) * (this->train_data[j][i] - mean_vector[i]);
        }

        std_vector[i] = sqrt(sum / this->train_data.size());
    }

    // Normalize training data
#pragma omp parallel for
    for (int i = 0; i < (int)this->train_data.size(); i++) {
        for (int j = 0; j < (int)this->train_data[i].size(); j++) {
            this->train_data[i][j] = (this->train_data[i][j] - mean_vector[j]) / std_vector[j];
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
