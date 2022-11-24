#include <omp.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "neural_fr/datasets.hpp"
#include "neural_fr/networks.hpp"

void conv_net() {
    Dataset1D dataset(0.0);
    DenseNetwork network({new layers::Input(28, 28), new layers::Conv2D(64, 3, 1, layers::leaky_relu),
                          new layers::MaxPool2D(2, 2), new layers::Flatten2D(), new layers::Dropout(0.2),
                          new layers::Dense(128, layers::leaky_relu),
                          new layers::Dense(10, layers::softmax)});

    LearningRateScheduler* custom_learn = new CustomSquareLearningRate(0.0001, 0.000001, 1.65);

    network.fit(dataset, 25, 128, custom_learn, false);

    network.save_predictions(dataset);
}

void mnist_net(int epochs, int minibatch_size, LearningRateScheduler* learn_scheduler) {
    Dataset1D dataset(0.05, true, 0.1);
    DenseNetwork network({new layers::Input(28, 28), new layers::Conv2D(64, 3, 1, layers::leaky_relu),
                          new layers::MaxPool2D(2, 2), new layers::Flatten2D(), new layers::Dropout(0.2),
                          new layers::Dense(128, layers::leaky_relu),
                          new layers::Dense(10, layers::softmax)});

    network.fit(dataset, epochs, minibatch_size, learn_scheduler);

    std::cout << "Test Accuracy: " << network.accuracy(dataset.test_data, dataset.test_labels)
              << std::endl;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    // Get epochs and learning rate from command line arguments
    // int epochs = 10;
    // int minibatch_size = 64;
    // double learning_rate_start = 0.001;
    // double learning_rate_end = learning_rate_start / 10;
    // if (argc > 1) {
    //     epochs = std::stoi(argv[1]);
    // }
    // if (argc > 2) {
    //     minibatch_size = std::stoi(argv[2]);
    // }
    // if (argc > 3) {
    //     learning_rate_start = std::stod(argv[3]);
    //     learning_rate_end = learning_rate_start / 10;
    // }
    // if (argc > 4) {
    //     learning_rate_end = std::stod(argv[4]);
    // }

    // Redirect cout to file
    // std::ofstream out("output.txt");
    // std::streambuf *coutbuf = std::cout.rdbuf();
    // std::cout.rdbuf(out.rdbuf());

    // Set double precision
    std::cout << std::fixed;
    std::cout << std::setprecision(4);

    // measure time
    double start, end;

    start = omp_get_wtime();
    conv_net();
    end = omp_get_wtime();
    std::cout << "Time: " << (double)(end - start) << "s" << std::endl;

    return 0;
}
