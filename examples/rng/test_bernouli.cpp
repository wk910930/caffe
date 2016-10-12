#include "caffe/util/math_functions.hpp"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: test_bernouli ratio num" << std::endl;
    return 1;
  }
  float ratio = atof(argv[1]);
  int num = atoi(argv[2]);
  int* outputs = new int[num];
  caffe::caffe_rng_bernoulli(num, ratio, outputs);
  std::cout << "Generate " << num << " random numbers "
      << "from {0, 1} with ratio " << ratio << std::endl;
  for (int i = 0; i < num; ++i)
    std::cout << i << ": " << outputs[i] << std::endl;
  delete [] outputs;

  return 0;
}
