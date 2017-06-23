#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/memory_layer.hpp"

namespace caffe {

template <typename Dtype>
void MemoryLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_class_ = this->layer_param_.memory_param().num_class();
  moving_average_fraction_ =
      this->layer_param_.memory_param().moving_average_fraction();
  has_ignore_label_ =
    this->layer_param_.memory_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.memory_param().ignore_label();
  }
  // Initialize parameters
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // Allocate memory: [num_class, channels, height, width]
    this->blobs_.resize(1);
    vector<int> shape;
    shape.push_back(num_class_);
    shape.push_back(bottom[0]->channels());
    shape.push_back(bottom[0]->height());
    shape.push_back(bottom[0]->width());
    // moving average mean
    this->blobs_[0].reset(new Blob<Dtype>(shape));
    caffe_set(this->blobs_[0]->count(), Dtype(0),
        this->blobs_[0]->mutable_cpu_data());
  }
  // Mask statistics from optimization by setting local learning rates to zero.
  for (int i = 0; i < this->blobs_.size(); ++i) {
    if (this->layer_param_.param_size() == i) {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      fixed_param_spec->set_lr_mult(0.f);
    } else {
      CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
          << "Cannot configure memory statistics as layer parameters.";
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void MemoryLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->ReshapeLike(*(bottom[0]));
}

template <typename Dtype>
void MemoryLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // sample_length = CxHxW
  int sample_length = bottom[0]->count(1);
  // Accumulate to the memory
  for (int n = 0; n < num_; ++n) {
    const int label_value = static_cast<int>(label[n]);
    CHECK_GT(label_value, -1) << "label should be within [0, "
        << num_class_ - 1 << "]";
    CHECK_LT(label_value, num_class_) << "label should be within [0, "
        << num_class_ - 1 << "]";
    Dtype* memory_start_ptr = this->blobs_[0]->mutable_cpu_data() +
        this->blobs_[0]->offset(label_value);
    caffe_cpu_axpby(sample_length,
        (Dtype(1) - moving_average_fraction_), bottom_data,
        moving_average_fraction_, memory_start_ptr);
    // Move to next instance
    bottom_data += bottom[0]->offset(1);
  }
  // Jump back to the first instance
  bottom_data = bottom[0]->cpu_data();
  for (int n = 0; n < num_; ++n) {
    const int label_value = static_cast<int>(label[n]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      caffe_copy(sample_length, bottom_data, top_data);
    } else {
      Dtype* memory_start_ptr = this->blobs_[0]->mutable_cpu_data() +
          this->blobs_[0]->offset(label_value);
      caffe_copy(sample_length, memory_start_ptr, top_data);
    }
    // Move to next instance
    top_data += top[0]->offset(1);
    bottom_data += bottom[0]->offset(1);
  }
}

template <typename Dtype>
void MemoryLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MemoryLayer);
#endif

INSTANTIATE_CLASS(MemoryLayer);
REGISTER_LAYER_CLASS(Memory);

}  // namespace caffe
