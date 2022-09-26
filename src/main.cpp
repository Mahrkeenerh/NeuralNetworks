#include <cmath>
#include <iostream>
#include <vector>

#include "datasets.h"
#include "layers.h"
#include "networks.h"

void xor_net() {
    DenseNetwork network({2, 3, 2});

    // Evaluate network
    std::vector<std::vector<float>> input_data = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}};
    std::vector<int> target_data = {0, 1, 1, 0};

    // Train network
    network.train(input_data, target_data, 10000, 0.1);

    // Evaluate network
    for (int i = 0; i < 4; i++) {
        std::vector<float> output = network.predict(input_data[i]);

        int result = output[0] > output[1] ? 0 : 1;

        std::cout << "Input: " << input_data[i][0] << ", " << input_data[i][1];
        std::cout << " | Output: " << result;
        std::cout << " | Confidence: " << output[result];
        std::cout << " | Target: " << target_data[i] << std::endl;
    }
}

void mnist_net() {
    Datasets1D datasets(800, 0);

    DenseNetwork network({784, 16, 10});

    network.train(datasets.train_data, datasets.train_labels, 10, 0.1);

    // Evaluate network
    for (int i = 0; i < 10; i++) {
        std::vector<float> output = network.predict(datasets.test_data[i]);

        int result = 0;
        for (int j = 1; j < 10; j++) {
            if (output[j] > output[result]) {
                result = j;
            }
        }

        std::cout << "i: " << i;
        std::cout << " | Output: " << result;
        std::cout << " | Confidence: " << output[result];
        std::cout << " | Target: " << datasets.test_labels[i] << std::endl;
    }
}

int main() {
    // measure time
    clock_t start, end;

    start = clock();
    Datasets1D datasets(-1, -1, true);
    end = clock();

    std::cout << "Time: " << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << "ms" << std::endl;

    start = clock();
    Datasets1D datasets2(-1, -1, false);
    end = clock();

    std::cout << "Time: " << (end - start) / (double)(CLOCKS_PER_SEC / 1000) << "ms" << std::endl;

    // validate training data
    for (int i = 0; i < 50000; i++) {
        for (int j = 0; j < 784; j++) {
            if (datasets.train_data[i][j] != datasets2.train_data[i][j]) {
                std::cout << "Error: " << i << ", " << j << std::endl;
            }
        }
    }

    // validate training labels
    for (int i = 0; i < 50000; i++) {
        if (datasets.train_labels[i] != datasets2.train_labels[i]) {
            std::cout << "Error: " << i << std::endl;
        }
    }

    // validate test data
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 784; j++) {
            if (datasets.test_data[i][j] != datasets2.test_data[i][j]) {
                std::cout << "Error: " << i << ", " << j << std::endl;
            }
        }
    }

    // validate test labels
    for (int i = 0; i < 10000; i++) {
        if (datasets.test_labels[i] != datasets2.test_labels[i]) {
            std::cout << "Error: " << i << std::endl;
        }
    }

    // validate valid data
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 784; j++) {
            if (datasets.valid_data[i][j] != datasets2.valid_data[i][j]) {
                std::cout << "Error: " << i << ", " << j << std::endl;
            }
        }
    }

    // validate valid labels
    for (int i = 0; i < 10000; i++) {
        if (datasets.valid_labels[i] != datasets2.valid_labels[i]) {
            std::cout << "Error: " << i << std::endl;
        }
    }

    // Validation complete
    std::cout << "Validation complete" << std::endl;

    // xor_net();
    // mnist_net();

    return 0;
}
