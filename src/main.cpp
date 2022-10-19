#include <omp.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "datasets.hpp"
#include "networks.hpp"
#include "optimizations.hpp"

void mnist_net(int epochs, int minibatch_size, double learning_rate_start, double learning_rate_end) {
    Dataset1D dataset;
    DenseNetwork network =
        DenseNetwork(784).add_layer(new DenseLayer(256, relu)).add_layer(new SoftmaxLayer(10));

    network.fit(dataset, epochs, minibatch_size, learning_rate_start, learning_rate_end);

    // Evaluate network
    // for (int i = 0; i < 10; i++) {
    //     std::vector<double> output = network.predict(dataset.test_data[i]);

    //     int result = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    //     double confidence = output[result] / std::accumulate(output.begin(), output.end(), 0.0);

    //     std::cout << "i: " << i;
    //     std::cout << " | Target: " << dataset.test_labels[i];
    //     std::cout << " | Output: " << result;
    //     std::cout << " | Confidence: " << confidence << std::endl;
    // }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    // Get epochs and learning rate from command line arguments
    int epochs = 10;
    int minibatch_size = 64;
    double learning_rate_start = 0.001;
    double learning_rate_end = learning_rate_start / 10;
    if (argc > 1) {
        epochs = std::stoi(argv[1]);
    }
    if (argc > 2) {
        minibatch_size = std::stoi(argv[2]);
    }
    if (argc > 3) {
        learning_rate_start = std::stod(argv[3]);
    }
    if (argc > 4) {
        learning_rate_end = std::stod(argv[4]);
    }

    // Redirect cout to file
    // std::ofstream out("output.txt");
    // std::streambuf *coutbuf = std::cout.rdbuf();
    // std::cout.rdbuf(out.rdbuf());

    // Set double precision
    std::cout << std::fixed;
    std::cout << std::setprecision(5);

    // measure time
    double start, end;

    for (int i = 0; i < 10; i++) {
        start = omp_get_wtime();
        mnist_net(epochs, minibatch_size, learning_rate_start, learning_rate_end);
        end = omp_get_wtime();
        std::cout << "Time: " << (double)(end - start) << "s" << std::endl;
    }

    return 0;
}
