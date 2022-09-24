#include <iostream>
#include <vector>

class DenseLayer {
   public:
    int input_size;
    int output_size;

    DenseLayer(int input_size, int output_size) {
        this->input_size = input_size;
        this->output_size = output_size;

        this->weights = std::vector<std::vector<double>>(output_size, std::vector<double>(input_size, 0.0));
        this->biases = std::vector<double>(output_size, 0.0);

        // Initialize weights and biases with random values between -1 and 1
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                this->weights[i][j] = (double)rand() / RAND_MAX * 2 - 1;
            }
            this->biases[i] = (double)rand() / RAND_MAX * 2 - 1;
        }
    }

    std::vector<double> get_output(std::vector<double> input) {
        std::vector<double> output(this->output_size, 0.0);

        for (int i = 0; i < this->output_size; i++) {
            for (int j = 0; j < this->input_size; j++) {
                output[i] += this->weights[i][j] * input[j];
            }
            output[i] += this->biases[i];
        }

        // TODO apply activation function
        return output;
    }

   private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

int main() {
    DenseLayer layer(2, 3);
    std::vector<double> input = {1.0, 2.0};
    std::vector<double> output = layer.get_output(input);

    for (int i = 0; i < output.size(); i++) {
        std::cout << output[i] << std::endl;
    }
}
