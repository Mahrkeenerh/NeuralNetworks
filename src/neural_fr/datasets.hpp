#ifndef datasets_hpp
#define datasets_hpp

#include <vector>

class Dataset1D {
   public:
    std::vector<std::vector<double>> train_data, valid_data, test_data;
    std::vector<int> train_labels, valid_labels, test_labels;

    int train_size, valid_size, test_size;

    Dataset1D(double val_split, bool normalize = true, double noise = -1);

   private:
    void load_data();
    void normalize_data();
    void noise_data(double noise_strength);
};

#endif
