#include <iostream>
#include <vector>

#include "datasets.h"
#include "layers.h"

class DenseNetwork {
   public:
    DenseNetwork(std::vector<int> layer_sizes) {
        this->layer_sizes = layer_sizes;

        for (int i = 0; i < layer_sizes.size() - 1; i++) {
            this->layers.push_back(DenseLayer(layer_sizes[i], layer_sizes[i + 1]));
        }
    }

    std::vector<double> predict(std::vector<double> input) {
        std::vector<double> output = input;

        for (int i = 0; i < this->layers.size(); i++) {
            output = this->layers[i].predict(output);
        }

        return output;
    }

   private:
    std::vector<int> layer_sizes;
    std::vector<DenseLayer> layers;
};

int main() {
    // measure time
    clock_t start = clock();

    Datasets1D datasets;

    std::cout << "Training data: ";
    for (int i = 0; i < 28 * 28; i++) {
        std::cout << datasets.train_data[0][i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Training label: " << datasets.train_labels[0] << std::endl;

    std::cout << "Test data: ";
    for (int i = 0; i < 28 * 28; i++) {
        std::cout << datasets.test_data[0][i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Test label: " << datasets.test_labels[0] << std::endl;

    std::cout << "Validation data: ";
    for (int i = 0; i < 28 * 28; i++) {
        std::cout << datasets.validation_data[0][i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Validation label: " << datasets.validation_labels[0] << std::endl;

    DenseNetwork network({2, 3, 2});

    std::vector<double> input = {1.0, 2.0};
    std::vector<double> output = network.predict(input);

    for (int i = 0; i < output.size(); i++) {
        std::cout << output[i] << std::endl;
    }

    // measure time
    clock_t end = clock();
    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Time elapsed: " << elapsed_secs << std::endl;

    return 0;
}
