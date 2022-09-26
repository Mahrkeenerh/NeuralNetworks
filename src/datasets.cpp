#include "datasets.h"

#include <fstream>
#include <sstream>

Datasets1D::Datasets1D(int train_size, int test_size) {
    this->train_size = train_size;
    this->test_size = test_size;

    // Set sizes
    if (train_size == -1) {
        this->train_size = 50000;
    }
    if (test_size == -1) {
        this->test_size = 10000;
    }

    // Validate sizes
    if (train_size > 50000 || train_size < 0) {
        this->train_size = 50000;
    }
    if (test_size > 10000 || test_size < 0) {
        this->test_size = 10000;
    }

    load_data();
}

void Datasets1D::load_data() {
    std::string line;

    std::ifstream train_data_file("old_data/fashion_mnist_train_vectors.csv");

    // Load training data
    for (int i = 0; i < this->train_size; i++) {
        std::vector<float> row;
        std::string cell;
        std::getline(train_data_file, line);
        std::stringstream line_stream(line);

        while (std::getline(line_stream, cell, ',')) {
            row.push_back(std::stof(cell) / 255);
        }

        this->train_data.push_back(row);
    }

    // Load test data
    for (int i = 0; i < this->test_size; i++) {
        std::vector<float> row;
        std::string cell;
        std::getline(train_data_file, line);
        std::stringstream line_stream(line);

        while (std::getline(line_stream, cell, ',')) {
            row.push_back(std::stof(cell) / 255);
        }

        this->test_data.push_back(row);
    }

    std::ifstream train_labels_file("old_data/fashion_mnist_train_labels.csv");

    // Load training labels
    for (int i = 0; i < this->train_size; i++) {
        std::getline(train_labels_file, line);
        this->train_labels.push_back(std::stoi(line));
    }

    // Load test labels
    for (int i = 0; i < this->test_size; i++) {
        std::getline(train_labels_file, line);
        this->test_labels.push_back(std::stoi(line));
    }

    std::ifstream validation_data_file("old_data/fashion_mnist_test_vectors.csv");

    // Load validation data
    for (int i = 0; i < this->test_size; i++) {
        std::vector<float> row;
        std::string cell;
        std::getline(validation_data_file, line);
        std::stringstream line_stream(line);

        while (std::getline(line_stream, cell, ',')) {
            row.push_back(std::stof(cell) / 255);
        }

        this->valid_data.push_back(row);
    }

    std::ifstream validation_labels_file("old_data/fashion_mnist_test_labels.csv");

    // Load validation labels
    for (int i = 0; i < this->test_size; i++) {
        std::getline(validation_labels_file, line);
        this->valid_labels.push_back(std::stoi(line));
    }
}
