#include <cmath>
#include <iostream>
#include <vector>

#include "datasets.h"
#include "layers.h"
#include "network_functions.h"

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

    // Datasets1D datasets(500, 100);
    DenseNetwork network({2, 4, 2});

    // Evaluate network
    std::vector<float> input_data[4] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}};
    std::vector<float> target_data[4] = {
        {1, 0},
        {0, 1},
        {0, 1},
        {1, 0}};

    for (int i = 0; i < 4; i++) {
        std::vector<float> output = network.predict(input_data[i]);
        float error = mse(output, target_data[i]);

        int result = output[0] > output[1] ? 0 : 1;

        std::cout << "Input: " << input_data[i][0] << ", " << input_data[i][1];
        std::cout << " | Output: " << result;
        std::cout << " | Confidence: " << output[0] << ", " << output[1];
        std::cout << " | Target: " << target_data[i][0] << ", " << target_data[i][1];
        std::cout << " | Error: " << error << std::endl;
    }

    // measure time
    clock_t end = clock();
    float elapsed_secs = float(end - start) / CLOCKS_PER_SEC;
    std::cout << "Time elapsed: " << elapsed_secs << std::endl;

    return 0;
}
