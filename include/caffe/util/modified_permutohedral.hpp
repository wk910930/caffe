#ifndef CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
#define CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/hash_table.hpp"

/************************************************/
/***          ModifiedPermutohedral Lattice   ***/
/************************************************/

namespace caffe {

typedef struct MatrixEntry {
  int index;
  float weight;
} MatrixEntry;

class ModifiedPermutohedral {
 public:
  ModifiedPermutohedral()
    : is_gpu_init_(false),
      N_(0),
      M_(0),
      d_(0) {}
  ~ModifiedPermutohedral() {
    #ifndef CPU_ONLY
    if (is_gpu_init_) {
      CUDA_CHECK(cudaFree(matrix));
    }
    #endif
  }

  void init_cpu(const float* features, int num_dimensions, int num_points);
  void init_gpu(const float* features, int num_dimensions, int w, int h);

  void compute_cpu(float* out, const float* in, int value_size,
      bool reverse = false, bool add = false) const;
  void compute_cpu(double* out, const double* in, int value_size,
      bool reverse = false, bool add = false) const;

  void compute_gpu(float* out, const float* in, int value_size,
      bool reverse = false, bool add = false) const;
  void compute_gpu(double* out, const double* in, int value_size,
      bool reverse = false, bool add = false) const;

 protected:
  struct Neighbors{
    int n1, n2;
    Neighbors(int n1 = 0, int n2 = 0):n1(n1), n2(n2) {}
  };
  // Check if GPU hash table if initialize
  bool is_gpu_init_ = false;

  std::vector<int> offset_, rank_;
  std::vector<float> barycentric_;
  std::vector<Neighbors> blur_neighbors_;

  // GPU specific
  MatrixEntry *matrix;
  HashTable table;

  // Number of elements, size of sparse discretized space,
  // dimension of features width and height
  int N_, M_, d_, w_, h_;

  void sseCompute(float* out, const float* in, int value_size,
      bool reverse = false, bool add = false) const;
  void sseCompute(double* out, const double* in, int value_size,
      bool reverse = false, bool add = false) const;

  void seqCompute(float* out, const float* in, int value_size,
      bool reverse = false, bool add = false) const;
  void seqCompute(double* out, const double* in, int value_size,
      bool reverse = false, bool add = false) const;
};

}  // namespace caffe

#endif  // CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
