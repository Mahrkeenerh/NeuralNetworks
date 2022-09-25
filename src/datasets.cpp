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

    // Allocate memory for training data
    for (int i = 0; i < this->train_size; i++) {
        train_data[i] = new float[28 * 28];
    }

    // Allocate memory for test and validation data
    for (int i = 0; i < this->test_size; i++) {
        test_data[i] = new float[28 * 28];
        validation_data[i] = new float[28 * 28];
    }

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
        fscanf(train_labels_file, "%f,", &train_labels[i]);
    }

    // Load test labels
    for (int i = 0; i < this->test_size; i++) {
        fscanf(train_labels_file, "%f,", &test_labels[i]);
    }
    fclose(train_labels_file);

    // Load validation data
    FILE *validation_data_file = fopen("old_data/fashion_mnist_test_vectors.csv", "r");
    for (int i = 0; i < this->test_size; i++) {
        for (int j = 0; j < 28 * 28; j++) {
            fscanf(validation_data_file, "%f,", &validation_data[i][j]);
            validation_data[i][j] /= 255;
        }
    }
    fclose(validation_data_file);

    // Load validation labels
    FILE *validation_labels_file = fopen("old_data/fashion_mnist_test_labels.csv", "r");
    for (int i = 0; i < this->test_size; i++) {
        fscanf(validation_labels_file, "%f,", &validation_labels[i]);
    }
    fclose(validation_labels_file);
}
