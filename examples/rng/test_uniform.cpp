#include <iostream>
#include "caffe/util/math_functions.hpp"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cout << "Usage: test_bernouli "
        << "lower_bound upper_bound num" << std::endl;
    return 1;
  }
  float lower = atof(argv[1]);
  float upper = atof(argv[2]);
  int num = atoi(argv[3]);
  float* outputs = new float[num];
  std::cout << "Generate " << num << " random numbers "
      << "within (" << lower << ", " << upper << ") uniformly" << std::endl;
  caffe::caffe_rng_uniform(num, lower, upper, outputs);
  for (int i = 0; i < num; ++i)
    std::cout << outputs[i] << std::endl;
  delete [] outputs;

  return 0;
}
