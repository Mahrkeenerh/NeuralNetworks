#include "datasets.h"

#include <iostream>

Datasets1D::Datasets1D() {
    // Allocate memory for training data
    for (int i = 0; i < 50000; i++) {
        train_data[i] = new float[28 * 28];
    }

    // Allocate memory for test and validation data
    for (int i = 0; i < 10000; i++) {
        test_data[i] = new float[28 * 28];
        validation_data[i] = new float[28 * 28];
    }

    load_data();
}

void Datasets1D::load_data() {
    // Load training data
    FILE *train_data_file = fopen("old_data/fashion_mnist_train_vectors.csv", "r");
    for (int i = 0; i < 50000; i++) {
        for (int j = 0; j < 28 * 28; j++) {
            fscanf(train_data_file, "%f,", &train_data[i][j]);
            train_data[i][j] /= 255;
        }
    }

    // Load test data
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 28 * 28; j++) {
            fscanf(train_data_file, "%f,", &test_data[i][j]);
            test_data[i][j] /= 255;
        }
    }
    fclose(train_data_file);

    // Load training labels
    FILE *train_labels_file = fopen("old_data/fashion_mnist_train_labels.csv", "r");
    for (int i = 0; i < 50000; i++) {
        fscanf(train_labels_file, "%f,", &train_labels[i]);
    }

    // Load test labels
    for (int i = 0; i < 10000; i++) {
        fscanf(train_labels_file, "%f,", &test_labels[i]);
    }
    fclose(train_labels_file);

    // Load validation data
    FILE *validation_data_file = fopen("old_data/fashion_mnist_test_vectors.csv", "r");
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 28 * 28; j++) {
            fscanf(validation_data_file, "%f,", &validation_data[i][j]);
            validation_data[i][j] /= 255;
        }
    }
    fclose(validation_data_file);

    // Load validation labels
    FILE *validation_labels_file = fopen("old_data/fashion_mnist_test_labels.csv", "r");
    for (int i = 0; i < 10000; i++) {
        fscanf(validation_labels_file, "%f,", &validation_labels[i]);
    }
    fclose(validation_labels_file);
}
