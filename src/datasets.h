#ifndef datasets_h
#define datasets_h

class Datasets1D {
   public:
    float *train_data[50000], train_labels[50000];
    float *test_data[10000], test_labels[10000];
    float *validation_data[10000], validation_labels[10000];

    int train_size, test_size;

    Datasets1D(int train_size = -1, int test_size = -1);

   private:
    void load_data();
};

#endif
