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
    clock_t start = clock();

    xor_net();
    // mnist_net();

    // measure time
    clock_t end = clock();
    float elapsed_secs = float(end - start) / CLOCKS_PER_SEC;
    std::cout << "Time elapsed: " << elapsed_secs << std::endl;

    return 0;
}
