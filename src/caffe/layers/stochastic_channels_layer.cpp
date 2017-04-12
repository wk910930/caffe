#include <vector>

#include "caffe/layers/stochastic_channels_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StochasticChannelsLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.stochastic_channels_param().channel_id_size() > 0) {
    random_ = false;
  } else {
    random_ = true;
  }
  if (random_) {
    num_output_ = this->layer_param_.stochastic_channels_param().num_output();
    if (num_output_ == 0) {
      num_output_ = bottom[0]->channels();
    }
  } else {
    num_output_ =
        this->layer_param_.stochastic_channels_param().channel_id_size();
  }
  CHECK_LE(num_output_, bottom[0]->channels())
      << "num_output should be less than input channels.";
  remain_channel_idx_.resize(num_output_);
}

template <typename Dtype>
void StochasticChannelsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  top_shape.push_back(bottom[0]->num());
  top_shape.push_back(num_output_);
  top_shape.push_back(bottom[0]->height());
  top_shape.push_back(bottom[0]->width());
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void StochasticChannelsLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num_pixels = bottom[0]->count(2);
  if (num_output_ < bottom[0]->channels()) {
    if (random_) {
      // Create random numbers for selecting channels
      for (int i = 0; i < num_output_; ++i) {
        remain_channel_idx_[i] = caffe_rng_rand() % bottom[0]->channels();
      }
    } else {
      for (int i = 0; i < num_output_; ++i) {
        remain_channel_idx_[i] =
            this->layer_param_.stochastic_channels_param().channel_id(i);
      }
    }
    // Copy selected channels to top
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int i = 0; i < num_output_; ++i) {
        int c_id = remain_channel_idx_[i];
        caffe_copy(num_pixels,
            bottom_data + bottom[0]->offset(0, c_id),
            top_data + top[0]->offset(0, i));
      }
      bottom_data += bottom[0]->offset(1);
      top_data += top[0]->offset(1);
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void StochasticChannelsLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0.), bottom_diff);
    if (num_output_ < bottom[0]->channels()) {
      int num_pixels = bottom[0]->count(2);
      for (int n = 0; n < bottom[0]->num(); ++n) {
        for (int i = 0; i < num_output_; ++i) {
          int c_id = remain_channel_idx_[i];
          caffe_copy(num_pixels,
              top_diff + top[0]->offset(0, i),
              bottom_diff + bottom[0]->offset(0, c_id));
        }
        bottom_diff += bottom[0]->offset(1);
        top_diff += top[0]->offset(1);
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(StochasticChannelsLayer);
#endif

INSTANTIATE_CLASS(StochasticChannelsLayer);
REGISTER_LAYER_CLASS(StochasticChannels);

}  // namespace caffe
