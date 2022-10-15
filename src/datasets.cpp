#include "datasets.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

Dataset1D::Dataset1D(std::vector<std::vector<double>> data, std::vector<int> labels) {
    this->train_data = data;
    this->train_labels = labels;

    this->test_data = data;
    this->test_labels = labels;

    this->train_size = data.size();
    this->test_size = data.size();
}

Dataset1D::Dataset1D(int train_size, int test_size, bool normalize) {
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

    load_data();
    if (normalize) {
        normalize_data();
    }
}

void Dataset1D::load_data() {
    std::string line;

    std::ifstream train_data_file("data/fashion_mnist_train_vectors.csv");

    // Load training data
    for (int i = 0; i < this->train_size; i++) {
        std::vector<double> row;
        std::string cell;
        std::getline(train_data_file, line);
        std::stringstream line_stream(line);

        while (std::getline(line_stream, cell, ',')) {
            row.push_back(std::stof(cell) / 255);
        }

        this->train_data.push_back(row);
    }

    std::ifstream train_labels_file("data/fashion_mnist_train_labels.csv");

    // Load training labels
    for (int i = 0; i < this->train_size; i++) {
        std::getline(train_labels_file, line);
        this->train_labels.push_back(std::stoi(line));
    }

    std::ifstream test_data_file("data/fashion_mnist_test_vectors.csv");

    // Load test data
    for (int i = 0; i < this->test_size; i++) {
        std::vector<double> row;
        std::string cell;
        std::getline(test_data_file, line);
        std::stringstream line_stream(line);

        while (std::getline(line_stream, cell, ',')) {
            row.push_back(std::stof(cell) / 255);
        }

        this->test_data.push_back(row);
    }

    std::ifstream test_labels_file("data/fashion_mnist_test_labels.csv");

    // Load validation labels
    for (int i = 0; i < this->test_size; i++) {
        std::getline(test_labels_file, line);
        this->test_labels.push_back(std::stoi(line));
    }
}

void Dataset1D::normalize_data() {
    // Calculate mean of each feature
    std::vector<double> mean_vector;
    for (int i = 0; i < this->train_data[0].size(); i++) {
        double sum = 0;
        for (int j = 0; j < this->train_data.size(); j++) {
            sum += this->train_data[j][i];
        }

        mean_vector.push_back(sum / this->train_data.size());
    }

    // Calculate standard deviation of each feature
    std::vector<double> std_vector;
    for (int i = 0; i < this->train_data[0].size(); i++) {
        double sum = 0;
        for (int j = 0; j < this->train_data.size(); j++) {
            sum += pow(this->train_data[j][i] - mean_vector[i], 2);
        }

        std_vector.push_back(sqrt(sum / this->train_data.size()));
    }

    // Normalize training data
    for (int i = 0; i < this->train_data.size(); i++) {
        for (int j = 0; j < this->train_data[i].size(); j++) {
            this->train_data[i][j] = (this->train_data[i][j] - mean_vector[j]) / std_vector[j];
        }
    }

    // Normalize test data
    for (int i = 0; i < this->test_data.size(); i++) {
        for (int j = 0; j < this->test_data[i].size(); j++) {
            this->test_data[i][j] = (this->test_data[i][j] - mean_vector[j]) / std_vector[j];
        }
    }
}
