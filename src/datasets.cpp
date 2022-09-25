#include "datasets.h"

#include <iostream>

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

    // Allocate memory
    this->train_data = std::vector<std::vector<float>>(this->train_size, std::vector<float>(28 * 28));
    this->train_labels = std::vector<int>(this->train_size);
    this->test_data = std::vector<std::vector<float>>(this->test_size, std::vector<float>(28 * 28));
    this->test_labels = std::vector<int>(this->test_size);
    this->valid_data = std::vector<std::vector<float>>(test_size, std::vector<float>(28 * 28));
    this->valid_labels = std::vector<int>(test_size);

    load_data();
}

void Datasets1D::load_data() {
    // Load training data
    FILE *train_data_file = fopen("old_data/fashion_mnist_train_vectors.csv", "r");
    for (int i = 0; i < this->train_size; i++) {
        for (int j = 0; j < 28 * 28; j++) {
            fscanf(train_data_file, "%f,", &train_data[i][j]);
            train_data[i][j] /= 255;
        }
    }

    // Load test data
    for (int i = 0; i < this->test_size; i++) {
        for (int j = 0; j < 28 * 28; j++) {
            fscanf(train_data_file, "%f,", &test_data[i][j]);
            test_data[i][j] /= 255;
        }
    }
    fclose(train_data_file);

    // Load training labels
    FILE *train_labels_file = fopen("old_data/fashion_mnist_train_labels.csv", "r");
    for (int i = 0; i < this->train_size; i++) {
        fscanf(train_labels_file, "%d,", &train_labels[i]);
    }

    // Load test labels
    for (int i = 0; i < this->test_size; i++) {
        fscanf(train_labels_file, "%d,", &test_labels[i]);
    }
    fclose(train_labels_file);

    // Load validation data
    FILE *validation_data_file = fopen("old_data/fashion_mnist_test_vectors.csv", "r");
    for (int i = 0; i < this->test_size; i++) {
        for (int j = 0; j < 28 * 28; j++) {
            fscanf(validation_data_file, "%f,", &valid_data[i][j]);
            valid_data[i][j] /= 255;
        }
    }
    fclose(validation_data_file);

    // Load validation labels
    FILE *validation_labels_file = fopen("old_data/fashion_mnist_test_labels.csv", "r");
    for (int i = 0; i < this->test_size; i++) {
        fscanf(validation_labels_file, "%d,", &valid_labels[i]);
    }
    fclose(validation_labels_file);
}
