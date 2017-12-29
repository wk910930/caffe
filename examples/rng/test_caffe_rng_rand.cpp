#include "caffe/util/math_functions.hpp"

int main(int argc, char* argv[]) {
  std::cout << caffe::caffe_rng_rand() << std::endl;
  return 0;
}
