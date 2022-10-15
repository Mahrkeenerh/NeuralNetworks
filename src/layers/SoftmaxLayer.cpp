#include "SoftmaxLayer.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

SoftmaxLayer::SoftmaxLayer(int input_size, int output_size) {
	this->input_size = input_size;
	this->output_size = output_size;

	this->weights =
	    std::vector<std::vector<float>>(output_size, std::vector<float>(input_size + 1, 1.0));
	this->errors = std::vector<float>(output_size, 0.0);
	this->outputs = std::vector<float>(output_size, 0.0);

	// Momentum value
	this->beta1 = 0.2;
	this->weight_delta =
	    std::vector<std::vector<float>>(output_size, std::vector<float>(input_size + 1, 0.0));

	// Initialize weights
	for (int i = 0; i < output_size; i++) {
		for (int j = 0; j < input_size + 1; j++) {
			this->weights[i][j] = randn() * sqrt(2.0 / input_size);
		}
	}

	// Adam settings
	this->momentum =
	    std::vector<std::vector<float>>(output_size, std::vector<float>(input_size + 1, 0.0));
	this->variance =
	    std::vector<std::vector<float>>(output_size, std::vector<float>(input_size + 1, 0.0));
	this->beta1 = 0.1;
	this->beta2 = 0.999;
	this->eta = 0.01;
	this->epsilon = 1e-8;
}

std::vector<float> SoftmaxLayer::predict(std::vector<float> input) {
	// #pragma omp parallel for
	// Calculate output for each neuron
	float sum = 0;
	for (int n_i = 0; n_i < this->output_size; n_i++) {
		this->outputs[n_i] = this->weights[n_i][0];

		for (int i = 0; i < this->input_size; i++) {
			this->outputs[n_i] += this->weights[n_i][i + 1] * input[i];
		}
		sum += exp(this->outputs[n_i]);
	}

	for (int n_i = 0; n_i < this->output_size; n_i++) {
		this->outputs[n_i] = exp(this->outputs[n_i] - 0.0001) / sum;
	}

	return this->outputs;
}

void SoftmaxLayer::out_errors(std::vector<float> target_vector) {
	// Calculate errors - MSE and apply activation function
	// for (int n_i = 0; n_i < this->output_size; n_i++) {
	//    this->errors[n_i] = (this->outputs[n_i] - target_vector[n_i]) *
	//    this->derivative(this->outputs[n_i]);
	// }

	// Derivative of cross entropy loss
	for (int n_i = 0; n_i < this->output_size; n_i++) {
		this->errors[n_i] = this->outputs[n_i] - target_vector[n_i];
	}
}

void SoftmaxLayer::backpropagate(Layer* connected_layer, std::vector<float> target_vector) {
	// #pragma omp parallel for
	return;
}

void SoftmaxLayer::update_weights(std::vector<float> input, float learning_rate, int t) {
	// #pragma omp parallel for
	float update;
	for (int n_i = 0; n_i < this->output_size; n_i++) {
		update = this->errors[0] * learning_rate + this->beta1 * this->weight_delta[n_i][0];
		this->weights[n_i][0] -= update;
		this->weight_delta[n_i][0] = update;

		for (int w_i = 1; w_i < this->input_size + 1; w_i++) {
			update = this->errors[n_i] * learning_rate * input[w_i - 1] +
			         this->beta1 * this->weight_delta[n_i][w_i];
			this->weights[n_i][w_i] -= update;
			this->weight_delta[n_i][w_i] = update;
		}
	}
}
