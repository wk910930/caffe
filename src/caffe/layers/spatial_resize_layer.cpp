#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <vector>

#include "caffe/layers/spatial_resize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SpatialResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void SpatialResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int tmp_h = this->layer_param_.resize_param().height();
  int tmp_w = this->layer_param_.resize_param().width();

  new_height_ = tmp_h > 0 ? tmp_h : bottom[0]->height();
  new_width_ = tmp_w > 0 ? tmp_w : bottom[0]->width();

  vector<int> top_shape;
  top_shape.push_back(bottom[0]->num());
  top_shape.push_back(bottom[0]->channels());
  top_shape.push_back(new_height_);
  top_shape.push_back(new_width_);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void SpatialResizeLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  for (int n = 0; n < num; ++n) {
    // we resize the bottom data channel-by-channel
    for (int c = 0; c < channels; ++c) {
      Dtype* cv_data = new Dtype[height * width];
      caffe_copy(height * width, bottom_data, cv_data);
      // Convert from blob to cvMat
      cv::Mat cv_img = cv::Mat(height, width, CV_64F, cv_data);
      // Resize
      cv::Mat resized_cv_img;
      resize(cv_img, resized_cv_img, cv::Size(new_width_, new_height_));
      // Convert from cvMat to blob
      caffe_copy(new_height_ * new_width_, reinterpret_cast<Dtype*>(resized_cv_img.data), top_data);
      delete [] cv_data;
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

template <typename Dtype>
void SpatialResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // Do nothing
  return;
}

#ifdef CPU_ONLY
STUB_GPU(SpatialResizeLayer);
#endif

INSTANTIATE_CLASS(SpatialResizeLayer);
REGISTER_LAYER_CLASS(SpatialResize);

}  // namespace caffe
