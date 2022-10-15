#include "DenseLayer.h"

#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size, float (*activation)(float)) {
	this->input_size = input_size;
	this->output_size = output_size;
	this->activation = activation;

	if (activation == sigmoid) {
		this->derivative = sigmoid_derivative;
	} else if (activation == relu) {
		this->derivative = relu_derivative;
	} else if (activation == leaky_relu) {
		this->derivative = leaky_relu_derivative;
	} else if (activation == swish) {
		this->derivative = swish_derivative;
	}

	this->weights =
	    std::vector<std::vector<float>>(output_size, std::vector<float>(input_size + 1, 0.0));
	this->errors = std::vector<float>(output_size, 0.0);
	// this->batch_errors = std::vector<float>(output_size, 0.0);
	this->outputs = std::vector<float>(output_size, 0.0);

	// Momentum value
	this->beta1 = 0.3;
	this->weight_delta =
	    std::vector<std::vector<float>>(output_size, std::vector<float>(input_size + 1, 0.0));

	// Adam settings
	/* this->momentum =
	    std::vector<std::vector<float>>(output_size, std::vector<float>(input_size + 1, 0.0));
	this->variance =
	    std::vector<std::vector<float>>(output_size, std::vector<float>(input_size + 1, 0.0));
	this->beta1 = 0.9;
	this->beta2 = 0.999;
	this->eta = 0.01;
	this->epsilon = 1e-8; */

	// Initialize weights
	for (int i = 0; i < output_size; i++) {
		for (int j = 0; j < input_size + 1; j++) {
			if (activation == sigmoid) {
				// Initialize weights with random values with uniform distribution
				// [-(1 / sqrt(input_size)), 1 / sqrt(input_size)]
				this->weights[i][j] =
				    (rand() / (float)RAND_MAX) * 2.0 / sqrt(input_size) - 1.0 / sqrt(input_size);
			} else {
				// Initialize with normal distribution
				this->weights[i][j] = randn() * sqrt(2.0 / input_size);
			}
		}
	}
}

std::vector<float> DenseLayer::predict(std::vector<float> input) {
	// #pragma omp parallel for
	// Calculate output for each neuron
	for (int n_i = 0; n_i < this->output_size; n_i++) {
		this->outputs[n_i] = this->weights[n_i][0];

		for (int i = 0; i < this->input_size; i++) {
			this->outputs[n_i] += this->weights[n_i][i + 1] * input[i];
		}
	}

	// #pragma omp parallel for
	// Apply activation function
	for (int i = 0; i < this->output_size; i++) {
		this->outputs[i] = this->activation(this->outputs[i]);
	}

	return this->outputs;
}

void DenseLayer::out_errors(std::vector<float> target_vector) {
	// Calculate errors - MSE
	for (int n_i = 0; n_i < this->output_size; n_i++) {
		this->errors[n_i] = (this->outputs[n_i] - target_vector[n_i]);
	}

	// Apply activation function
	for (int n_i = 0; n_i < this->output_size; n_i++) {
		this->errors[n_i] *= this->derivative(this->outputs[n_i]);
	}
}

void DenseLayer::backpropagate(Layer* connected_layer, std::vector<float> target_vector) {
	// #pragma omp parallel for
	for (int n_i = 0; n_i < this->output_size; n_i++) {
		this->errors[n_i] = 0;

		for (int o_i = 0; o_i < connected_layer->output_size; o_i++) {
			this->errors[n_i] += connected_layer->errors[o_i] * connected_layer->weights[o_i][n_i + 1];
		}
	}

	// Apply activation function
	for (int n_i = 0; n_i < this->output_size; n_i++) {
		this->errors[n_i] *= this->derivative(this->outputs[n_i]);
	}
}

void DenseLayer::update_weights(std::vector<float> input, float learning_rate, int t) {
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

	// Adam
	/* float grad, alpha;
	#pragma omp parallel for
	for (int n_i = 0; n_i < this->output_size; n_i++) {
	    //#pragma omp parallel for
	    for (int w_i = 0; w_i < this->input_size + 1; w_i++) {
	        grad = this->errors[n_i];

	        this->momentum[n_i][w_i] = this->beta1 * this->momentum[n_i][w_i] + (1 - this->beta1) * grad;
	        this->variance[n_i][w_i] =
	            this->beta2 * this->variance[n_i][w_i] + (1 - this->beta2) * pow(grad, 2);

	        alpha = this->eta * sqrt((1 - pow(this->beta2, t + 1)) / (1 - pow(this->beta1, t + 1)));

	        // Bias
	        if (w_i == 0) {
	            this->weights[n_i][w_i] -= learning_rate * alpha * this->momentum[n_i][w_i] /
	                                       (sqrt(this->variance[n_i][w_i]) + this->epsilon);
	        }
	        // Weight
	        else {
	            this->weights[n_i][w_i] -= learning_rate * input[w_i - 1] * alpha *
	                                       this->momentum[n_i][w_i] /
	                                       (sqrt(this->variance[n_i][w_i]) + this->epsilon);
	        }
	    }
	} */
}
