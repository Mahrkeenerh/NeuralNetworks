#include <cmath>
#include <iostream>
#include <vector>

#include "datasets.h"
#include "layers.h"
#include "networks.h"

void xor_net() {
    std::vector<std::vector<float>> input_data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<int> target_data = {0, 1, 1, 0};
    Dataset1D dataset = Dataset1D(input_data, target_data);

    DenseNetwork network({2, 3, 2});

    // Train network
    network.fit(dataset, 100, 1);

    // Evaluate network
    for (int i = 0; i < 4; i++) {
        std::vector<float> output = network.predict(input_data[i]);

        int result = output[0] > output[1] ? 0 : 1;

        std::cout << "Input: " << input_data[i][0] << ", " << input_data[i][1];
        std::cout << " | Target: " << target_data[i];
        std::cout << " | Output: " << result;
        std::cout << " | Confidence: " << output[result] << std::endl;
    }
}

void mnist_net() {
    Dataset1D dataset(8000, 2000);
    DenseNetwork network({784, 128, 10});

    network.fit(dataset, 50, 0.0001);

    // Evaluate network
    for (int i = 0; i < 10; i++) {
        std::vector<float> output = network.predict(dataset.test_data[i]);

        int result = 0;
        for (int j = 1; j < 10; j++) {
            if (output[j] > output[result]) {
                result = j;
            }
        }

        std::cout << "i: " << i;
        std::cout << " | Target: " << dataset.test_labels[i];
        std::cout << " | Output: " << result;
        std::cout << " | Confidence: " << output[result] << std::endl;
    }
}

int main() {
    // measure time
    clock_t start, end;

    start = clock();

    // xor_net();
    mnist_net();

    end = clock();
    std::cout << "Time: " << (end - start) / CLOCKS_PER_SEC << "ms" << std::endl;

    return 0;
}
