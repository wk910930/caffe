#ifndef CAFFE_PSROI_POOLING_LAYER_HPP_
#define CAFFE_PSROI_POOLING_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * PSROIPoolingLayer:
 *   Position-Sensitive Region of Interest Pooling Layer
 */
template <typename Dtype>
class PSROIPoolingLayer : public Layer<Dtype> {
 public:
  explicit PSROIPoolingLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PSROIPooling"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype spatial_scale_;
  int output_dim_;
  int group_size_;

  int channels_;
  int height_;
  int width_;

  int pooled_height_;
  int pooled_width_;
  Blob<int> mapping_channel_;
};

}  // namespace caffe

#endif  // CAFFE_PSROI_POOLING_LAYER_HPP_
