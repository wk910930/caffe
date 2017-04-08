#ifndef CAFFE_MULTISTAGE_MEANFIELD_LAYER_HPP_
#define CAFFE_MULTISTAGE_MEANFIELD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/meanfield_iteration.hpp"

namespace caffe {

template <typename Dtype>
class MultiStageMeanfieldLayer : public Layer<Dtype> {
 public:
  explicit MultiStageMeanfieldLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiStageMeanfield"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  void init_param_blobs(const MultiStageMeanfieldParameter& meanfield_param);
  void init_spatial_lattice();
  void init_bilateral_buffers();
  void compute_spatial_kernel();
  void compute_bilateral_kernel(const Dtype* image_features);

  int num_;
  int channels_;
  int height_;
  int width_;
  int num_pixels_;

  int spatial_dim_;
  int bilateral_dim_;

  float eps_;

  Dtype theta_alpha_;
  Dtype theta_beta_;
  Dtype theta_gamma_;
  int num_iterations_;

  Blob<Dtype> norm_feed_;
  Blob<Dtype> spatial_norm_;
  Blob<Dtype> bilateral_norms_;

  vector<Blob<Dtype>*> split_layer_bottom_vec_;
  vector<Blob<Dtype>*> split_layer_top_vec_;
  vector<shared_ptr<Blob<Dtype> > > split_layer_out_blobs_;
  vector<shared_ptr<Blob<Dtype> > > iteration_output_blobs_;
  vector<shared_ptr<MeanfieldIteration<Dtype> > > meanfield_iterations_;

  shared_ptr<SplitLayer<Dtype> > split_layer_;

  Blob<Dtype> spatial_kernel_buffer_;
  Blob<Dtype> bilateral_kernel_buffer_;
  shared_ptr<ModifiedPermutohedral<Dtype> > spatial_lattice_;
  vector<shared_ptr<ModifiedPermutohedral<Dtype> > > bilateral_lattices_;
};

}  // namespace caffe

#endif  // CAFFE_MULTISTAGE_MEANFIELD_LAYER_HPP_
