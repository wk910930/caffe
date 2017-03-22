#include <vector>

#include "caffe/util/meanfield_iteration.hpp"

namespace caffe {

/**
 * Forward pass during the inference.
 */
template <typename Dtype>
void MeanfieldIteration<Dtype>::Forward_gpu() {
  /*-------------------- Normalization --------------------*/
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  /*-------------------- Message Passing --------------------*/
  for (int n = 0; n < num_; ++n) {
    const Dtype* prob_input_data = softmax_output_.cpu_data() +
        softmax_output_.offset(n);
    // Gaussian filters: spatial
    Dtype* spatial_out_cpu_data = spatial_out_blob_.mutable_cpu_data() +
        spatial_out_blob_.offset(n);
    spatial_lattice_->compute(spatial_out_cpu_data, prob_input_data, channels_);
    Dtype* spatial_out_data = spatial_out_blob_.mutable_gpu_data() +
        spatial_out_blob_.offset(n);
    // Gaussian filters: bilateral
    Dtype* bilateral_out_cpu_data = bilateral_out_blob_.mutable_cpu_data() +
        bilateral_out_blob_.offset(n);
    bilateral_lattices_[n]->compute(bilateral_out_cpu_data, prob_input_data,
        channels_);
    Dtype* bilateral_out_data = bilateral_out_blob_.mutable_gpu_data() +
        bilateral_out_blob_.offset(n);
    // Pixel-wise normalization.
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_gpu_mul(num_pixels_,
          spatial_norm_.gpu_data(),
          spatial_out_data + channel_id * num_pixels_,
          spatial_out_data + channel_id * num_pixels_);
      caffe_gpu_mul(num_pixels_,
          bilateral_norms_.gpu_data() + bilateral_norms_.offset(n),
          bilateral_out_data + channel_id * num_pixels_,
          bilateral_out_data + channel_id * num_pixels_);
    }
  }

  /*-------------------- Weighting Filter Outputs --------------------*/
  caffe_gpu_set(message_passing_.count(), Dtype(0.),
      message_passing_.mutable_gpu_data());
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        channels_, num_pixels_, channels_,
        Dtype(1.),
        this->blobs_[0]->gpu_data(),
        spatial_out_blob_.gpu_data() + spatial_out_blob_.offset(n),
        Dtype(0.),
        message_passing_.mutable_gpu_data() + message_passing_.offset(n));
  }
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        channels_, num_pixels_, channels_,
        Dtype(1.),
        this->blobs_[1]->gpu_data(),
        bilateral_out_blob_.gpu_data() + bilateral_out_blob_.offset(n),
        Dtype(1.),
        message_passing_.mutable_gpu_data() + message_passing_.offset(n));
  }

  /*-------------------- Compatibility Transform --------------------*/
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        channels_, num_pixels_, channels_,
        Dtype(1.),
        this->blobs_[2]->gpu_data(),
        message_passing_.gpu_data() + message_passing_.offset(n),
        Dtype(0.),
        pairwise_terms_.mutable_gpu_data() + pairwise_terms_.offset(n));
  }

  /*-------------------- Adding Unary Potentials --------------------*/
  sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);
}

template <typename Dtype>
void MeanfieldIteration<Dtype>::Backward_gpu() {
  /*-------------------- Add unary gradient --------------------*/
  vector<bool> eltwise_propagate_down(sum_bottom_vec_.size(), true);
  sum_layer_->Backward(sum_top_vec_, eltwise_propagate_down, sum_bottom_vec_);

  /*-------------------- Update compatibility diffs --------------------*/
  caffe_gpu_set(this->blobs_[2]->count(), Dtype(0.),
      this->blobs_[2]->mutable_gpu_diff());
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        channels_, channels_, num_pixels_,
        Dtype(1.),
        pairwise_terms_.gpu_diff() + pairwise_terms_.offset(n),
        message_passing_.gpu_data() + message_passing_.offset(n),
        Dtype(1.),
        this->blobs_[2]->mutable_gpu_diff());
  }

  /*----------------- Gradient after compatibility transform -----------------*/
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
        channels_, num_pixels_, channels_,
        Dtype(1.),
        this->blobs_[2]->gpu_data(),
        pairwise_terms_.gpu_diff() + pairwise_terms_.offset(n),
        Dtype(0.),
        message_passing_.mutable_gpu_diff() + message_passing_.offset(n));
  }

  /*-------------------- Gradient w.r.t. kernels weights --------------------*/
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0.),
      this->blobs_[0]->mutable_gpu_diff());
  caffe_gpu_set(this->blobs_[1]->count(), Dtype(0.),
      this->blobs_[1]->mutable_gpu_diff());

  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        channels_, channels_, num_pixels_,
        Dtype(1.),
        message_passing_.gpu_diff() + message_passing_.offset(n),
        spatial_out_blob_.gpu_data() + spatial_out_blob_.offset(n),
        Dtype(1.),
        this->blobs_[0]->mutable_gpu_diff());
  }

  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        channels_, channels_, num_pixels_,
        Dtype(1.),
        message_passing_.gpu_diff() + message_passing_.offset(n),
        bilateral_out_blob_.gpu_data() + bilateral_out_blob_.offset(n),
        Dtype(1.),
        this->blobs_[1]->mutable_gpu_diff());
  }

  // Check whether there's a way to improve the accuracy of this calculation.
  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
        channels_, num_pixels_, channels_,
        Dtype(1.),
        this->blobs_[0]->gpu_data(),
        message_passing_.gpu_diff() + message_passing_.offset(n),
        Dtype(0.),
        spatial_out_blob_.mutable_gpu_diff() + spatial_out_blob_.offset(n));
  }

  for (int n = 0; n < num_; ++n) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
        channels_, num_pixels_, channels_,
        Dtype(1.),
        this->blobs_[1]->gpu_data(),
        message_passing_.gpu_diff() + message_passing_.offset(n),
        Dtype(0.),
        bilateral_out_blob_.mutable_gpu_diff() + bilateral_out_blob_.offset(n));
  }

  /*-------------------- BP thru normalization --------------------*/
  for (int n = 0; n < num_; ++n) {
    Dtype* spatial_out_diff = spatial_out_blob_.mutable_gpu_diff() +
        spatial_out_blob_.offset(n);
    Dtype* bilateral_out_diff = bilateral_out_blob_.mutable_gpu_diff() +
        bilateral_out_blob_.offset(n);
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_gpu_mul(num_pixels_,
          spatial_norm_.gpu_data(),
          spatial_out_diff + channel_id * num_pixels_,
          spatial_out_diff + channel_id * num_pixels_);
      caffe_gpu_mul(num_pixels_,
          bilateral_norms_.gpu_data() + bilateral_norms_.offset(n),
          bilateral_out_diff + channel_id * num_pixels_,
          bilateral_out_diff + channel_id * num_pixels_);
    }
  }

  /*-------------------- Gradient for message passing --------------------*/
  for (int n = 0; n < num_; ++n) {
    spatial_lattice_->compute(
        softmax_output_.mutable_cpu_diff() + softmax_output_.offset(n),
        spatial_out_blob_.cpu_diff() + spatial_out_blob_.offset(n),
        channels_, true, false);
    bilateral_lattices_[n]->compute(
        softmax_output_.mutable_cpu_diff() + softmax_output_.offset(n),
        bilateral_out_blob_.cpu_diff() + bilateral_out_blob_.offset(n),
        channels_, true, true);
  }

  vector<bool> propagate_down(softmax_bottom_vec_.size(), true);
  softmax_layer_->Backward(softmax_top_vec_, propagate_down,
      softmax_bottom_vec_);
}

// Instantiate class
template void MeanfieldIteration<float>::Forward_gpu();
template void MeanfieldIteration<double>::Forward_gpu();
template void MeanfieldIteration<float>::Backward_gpu();
template void MeanfieldIteration<double>::Backward_gpu();

}  // namespace caffe
