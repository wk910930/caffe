#ifndef CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
#define CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_

#include <vector>

#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
class ModifiedPermutohedral {
 public:
  void init(const Dtype* features, int num_dimensions, int num_points);
  void compute(Dtype* out, const Dtype* in, int value_size,
      bool reverse = false, bool add = false) const;

 private:
  struct Neighbors{
    int n1;
    int n2;
    Neighbors(int n1 = 0, int n2 = 0) : n1(n1), n2(n2) {}
  };

  std::vector<int> offset_;
  std::vector<int> rank_;
  std::vector<Dtype> barycentric_;
  std::vector<Neighbors> blur_neighbors_;

  int N_ = 0;  // number of elements
  int M_ = 0;  // size of sparse discretized space
  int d_ = 0;  // dimension of features
  int w_ = 0;  // width
  int h_ = 0;  // height
};

}  // namespace caffe

#endif  // CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
