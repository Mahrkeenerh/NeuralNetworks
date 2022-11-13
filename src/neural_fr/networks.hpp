#ifndef networks
#define networks

#include <vector>

#include "datasets.hpp"
#include "layers.hpp"

class LearningRateScheduler {
   public:
    virtual double get_learning_rate(int epoch) { return 0; }
    virtual void network_setup(int epochs) {}
};

class DenseNetwork {
   public:
    DenseNetwork(std::vector<layers::Layer*> layers);

    DenseNetwork add_layer(layers::Layer* layer);

    std::vector<double> predict(std::vector<double> input, int thread_id = 0);

    void fit(Dataset1D dataset, int epochs, int minibatch_size, LearningRateScheduler* learn_scheduler,
             bool verbose = true);

    double accuracy(std::vector<std::vector<double>> inputs, std::vector<int> targets);

   private:
    int size = 0;
    std::vector<layers::Layer*> layers;

    void forwardpropagate(std::vector<double> input, int thread_id);
    void backpropagate(int thread_id, std::vector<double> target_vector);

    void calculate_updates(int thread_id, double learning_rate);
    void apply_updates(int minibatch_size);
    void clear_updates();

    void before_batch();
};

class ConstantLearningRate : public LearningRateScheduler {
   public:
    ConstantLearningRate(double learning_rate);

    double get_learning_rate(int epoch) override;

   private:
    double learning_rate;
};

class LinearLearningRate : public LearningRateScheduler {
   public:
    LinearLearningRate(double learning_rate_start, double learning_rate_end);

    void network_setup(int epochs) override;
    double get_learning_rate(int epoch) override;

   private:
    double learning_rate_start, learning_rate_end;
    double learning_rate_slope;
};

class HalvingLearningRate : public LearningRateScheduler {
   public:
    HalvingLearningRate(double learning_rate);

    double get_learning_rate(int epoch) override;

   private:
    double learning_rate;
};

class CustomSquareLearningRate : public LearningRateScheduler {
   public:
    CustomSquareLearningRate(double learning_rate_start, double learning_rate_end, double slope);

    void network_setup(int epochs) override;
    double get_learning_rate(int epoch) override;

   private:
    double learning_rate_start, learning_rate_end;
    double slope;
    int epochs;
};

#endif
