#ifndef datasets_h
#define datasets_h

#include <vector>

class Datasets1D {
   public:
    std::vector<std::vector<float>> train_data, test_data, valid_data;
    std::vector<int> train_labels, test_labels, valid_labels;

    int train_size, test_size;

    Datasets1D(int train_size = -1, int test_size = -1, bool old = false);

   private:
    void load_data();
    void load_old();
};

#endif
