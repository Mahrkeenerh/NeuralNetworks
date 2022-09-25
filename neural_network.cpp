#include <iostream>
#include <vector>

class Datasets1D {
   public:
    float *train_data[50000];
    float train_labels[50000];
    float *test_data[10000];
    float test_labels[10000];
    float *validation_data[10000];
    float validation_labels[10000];

    Datasets1D() {
        for (int i = 0; i < 50000; i++) {
            train_data[i] = new float[28 * 28];
        }

        for (int i = 0; i < 10000; i++) {
            test_data[i] = new float[28 * 28];
            validation_data[i] = new float[28 * 28];
        }

        load_data();
    }

   private:
    void load_data() {
        // Load training data
        FILE *train_data_file = fopen("old_data/fashion_mnist_train_vectors.csv", "r");
        for (int i = 0; i < 50000; i++) {
            for (int j = 0; j < 28 * 28; j++) {
                fscanf(train_data_file, "%f,", &train_data[i][j]);
                train_data[i][j] /= 255;
            }
        }

        // Load test data
        for (int i = 0; i < 10000; i++) {
            for (int j = 0; j < 28 * 28; j++) {
                fscanf(train_data_file, "%f,", &test_data[i][j]);
                test_data[i][j] /= 255;
            }
        }
        fclose(train_data_file);

        // Load training labels
        FILE *train_labels_file = fopen("old_data/fashion_mnist_train_labels.csv", "r");
        for (int i = 0; i < 50000; i++) {
            fscanf(train_labels_file, "%f,", &train_labels[i]);
        }

        // Load test labels
        for (int i = 0; i < 10000; i++) {
            fscanf(train_labels_file, "%f,", &test_labels[i]);
        }
        fclose(train_labels_file);

        // Load validation data
        FILE *validation_data_file = fopen("old_data/fashion_mnist_test_vectors.csv", "r");
        for (int i = 0; i < 10000; i++) {
            for (int j = 0; j < 28 * 28; j++) {
                fscanf(validation_data_file, "%f,", &validation_data[i][j]);
                validation_data[i][j] /= 255;
            }
        }
        fclose(validation_data_file);

        // Load validation labels
        FILE *validation_labels_file = fopen("old_data/fashion_mnist_test_labels.csv", "r");
        for (int i = 0; i < 10000; i++) {
            fscanf(validation_labels_file, "%f,", &validation_labels[i]);
        }
        fclose(validation_labels_file);
    }
};

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

    std::vector<double> predict(std::vector<double> input) {
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

    return 0;
}
