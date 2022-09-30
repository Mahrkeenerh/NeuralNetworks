#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "datasets.h"
#include "layers.h"
#include "networks.h"

void xor_net() {
    std::vector<std::vector<double>> input_data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<int> target_data = {0, 1, 1, 0};
    Dataset1D dataset = Dataset1D(input_data, target_data);

    DenseNetwork network({2, 3, 2});

    // Train network
    network.fit(dataset, 500, 1);

    // Evaluate network
    for (int i = 0; i < 4; i++) {
        std::vector<double> output = network.predict(input_data[i]);

        int result = output[0] > output[1] ? 0 : 1;

        std::cout << "Input: " << input_data[i][0] << ", " << input_data[i][1];
        std::cout << " | Target: " << target_data[i];
        std::cout << " | Output: " << result;
        std::cout << " | Confidence: " << output[result] << std::endl;
    }
}

void mnist_net(int epochs, double learning_rate) {
    Dataset1D dataset;
    DenseNetwork network({784, 128, 64, 10});

    network.fit(dataset, epochs, learning_rate, 1);

    // Evaluate network
    for (int i = 0; i < 10; i++) {
        std::vector<double> output = network.predict(dataset.test_data[i]);

        int result = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        double confidence = output[result] / std::accumulate(output.begin(), output.end(), 0.0);

        std::cout << "i: " << i;
        std::cout << " | Target: " << dataset.test_labels[i];
        std::cout << " | Output: " << result;
        std::cout << " | Confidence: " << confidence << std::endl;
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    // Get epochs and learning rate from command line arguments
    int epochs = 10;
    double learning_rate = 0.01;
    if (argc > 1) {
        epochs = std::stoi(argv[1]);
    }
    if (argc > 2) {
        learning_rate = std::stod(argv[2]);
    }

    // measure time
    clock_t start, end;

    start = clock();

    // xor_net();
    mnist_net(epochs, learning_rate);

    end = clock();
    std::cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    return 0;
}
