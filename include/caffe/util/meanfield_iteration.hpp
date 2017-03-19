#ifndef CAFFE_MEANFIELD_LAYER_HPP_
#define CAFFE_MEANFIELD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/modified_permutohedral.hpp"

namespace caffe {

template <typename Dtype>
class MeanfieldIteration {
 public:
  /**
   * Must be invoked only once after the construction of the layer.
   */
  void OneTimeSetUp(
      Blob<Dtype>* unary_terms,
      Blob<Dtype>* softmax_input,
      Blob<Dtype>* output_blob,
      const shared_ptr<ModifiedPermutohedral<Dtype> >& spatial_lattice,
      const Blob<Dtype>& spatial_norm);

  /**
   * Must be invoked before invoking {@link Forward_cpu()}
   */
  virtual void PrePass(
      const vector<shared_ptr<Blob<Dtype> > >& parameters_to_copy_from,
      const vector<shared_ptr<
          ModifiedPermutohedral<Dtype> > >& bilateral_lattices,
      const Blob<Dtype>& bilateral_norms);

  /**
   * CPU Forward/Backward pass
   */
  virtual void Forward_cpu();
  virtual void Backward_cpu();

  /**
   * GPU Forward/Backward pass
   */
#ifndef CPU_ONLY
  virtual void Forward_gpu();
  virtual void Backward_gpu();
#endif

  // A quick hack. This should be properly encapsulated.
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

 protected:
  /** The vector that stores the learnable parameters as a set of blobs. */
  vector<shared_ptr<Blob<Dtype> > > blobs_;

  int num_;
  int channels_;
  int num_pixels_;

  Blob<Dtype> pairwise_terms_;
  Blob<Dtype> spatial_out_blob_;
  Blob<Dtype> bilateral_out_blob_;
  Blob<Dtype> softmax_input_;
  Blob<Dtype> softmax_output_;
  Blob<Dtype> message_passing_;

  // Addition
  shared_ptr<EltwiseLayer<Dtype> > sum_layer_;
  vector<Blob<Dtype>*> sum_top_vec_;
  vector<Blob<Dtype>*> sum_bottom_vec_;

  // Normalization
  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
  vector<Blob<Dtype>*> softmax_top_vec_;
  vector<Blob<Dtype>*> softmax_bottom_vec_;

  Blob<Dtype> spatial_norm_;
  Blob<Dtype> bilateral_norms_;

  shared_ptr<ModifiedPermutohedral<Dtype> > spatial_lattice_;
  vector<shared_ptr<ModifiedPermutohedral<Dtype> > > bilateral_lattices_;
};

}  // namespace caffe

#endif  // CAFFE_MEANFIELD_LAYER_HPP_
