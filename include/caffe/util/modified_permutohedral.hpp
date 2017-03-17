#ifndef CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
#define CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_

#include "caffe/common.hpp"

namespace caffe {

/************************************************/
/***          ModifiedPermutohedral Lattice   ***/
/************************************************/

template <typename Dtype>
class ModifiedPermutohedral {
 public:
  ModifiedPermutohedral() {}

  void init(const Dtype* features, int num_dimensions, int num_points);
  void compute(Dtype* out, const Dtype* in, int value_size,
      bool reverse = false, bool add = false) const;

 private:
  struct Neighbors{
    int n1;
    int n2;
    Neighbors(int n1 = 0, int n2 = 0):n1(n1), n2(n2) {}
  };

  std::vector<int> offset_;
  std::vector<int> rank_;
  std::vector<float> barycentric_;
  std::vector<Neighbors> blur_neighbors_;

  // Number of elements, size of sparse discretized space,
  // dimension of features width and height
  int N_ = 0;
  int M_ = 0;
  int d_ = 0;
  int w_ = 0;
  int h_ = 0;
};

}  // namespace caffe

#endif  // CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
