#ifndef datasets_hpp
#define datasets_hpp

#include <vector>

class Dataset1D {
   public:
    std::vector<std::vector<double>> train_data, test_data;
    std::vector<int> train_labels, test_labels;

    int train_size, test_size;

    Dataset1D(int train_size = -1, int test_size = -1, bool normalize = true, double noise = -1);

   private:
    void load_data();
    void normalize_data();
    void noise_data(double noise_strength);
};

#endif
