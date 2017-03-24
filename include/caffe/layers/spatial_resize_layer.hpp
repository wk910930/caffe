#ifndef CAFFE_SPATIAL_RESIZE_LAYER_HPP_
#define CAFFE_SPATIAL_RESIZE_LAYER_HPP_

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
class SpatialResizeLayer : public Layer<Dtype> {
 public:
  explicit SpatialResizeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SpatialResize"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int new_height_;
  int new_width_;
};

}  // namespace caffe

#endif  // CAFFE_SPATIAL_RESIZE_LAYER_HPP_
