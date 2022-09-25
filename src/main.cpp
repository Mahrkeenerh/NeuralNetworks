#include <iostream>
#include <vector>

#include "activations.h"
#include "datasets.h"
#include "layers.h"

class DenseNetwork {
   public:
    DenseNetwork(std::vector<int> layer_sizes) {
        this->layer_sizes = layer_sizes;

        // Create layers
        for (int i = 0; i < layer_sizes.size() - 2; i++) {
            this->layers.push_back(
                DenseLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    relu));
        }

        // Create output layer
        this->layers.push_back(
            DenseLayer(
                layer_sizes[layer_sizes.size() - 2],
                layer_sizes[layer_sizes.size() - 1],
                sigmoid));
    }

    std::vector<float> predict(std::vector<float> input) {
        std::vector<float> output = input;

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

    DenseNetwork network({2, 4, 1});

    std::vector<float> output = network.predict({1.0, 2.0});

    for (int i = 0; i < output.size(); i++) {
        std::cout << output[i] << std::endl;
    }

    // measure time
    clock_t end = clock();
    float elapsed_secs = float(end - start) / CLOCKS_PER_SEC;
    std::cout << "Time elapsed: " << elapsed_secs << std::endl;

    return 0;
}
