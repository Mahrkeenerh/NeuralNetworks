#include <cmath>
#include <iostream>
#include <vector>

#include "datasets.h"
#include "layers.h"
#include "networks.h"

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
