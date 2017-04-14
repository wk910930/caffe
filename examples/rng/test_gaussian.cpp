#include "caffe/util/math_functions.hpp"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cout << "Usage: test_gaussian "
        << "mu(mean) sigma(std) num" << std::endl;
    return 1;
  }
  float mu = atof(argv[1]);
  float std = atof(argv[2]);
  int num = atoi(argv[3]);
  float* outputs = new float[num];
  std::cout << "Generate " << num << " random numbers "
      << "with mu=" << mu << " and sigma=" << std << "." << std::endl;
  caffe::caffe_rng_gaussian(num, mu, std, outputs);
  for (int i = 0; i < num; ++i)
    std::cout << outputs[i] << std::endl;
  delete [] outputs;

  return 0;
}
