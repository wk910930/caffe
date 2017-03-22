#include <vector>

#include "caffe/util/meanfield_iteration.hpp"

namespace caffe {

/**
 * To be invoked once only immediately after construction.
 */
template <typename Dtype>
void MeanfieldIteration<Dtype>::OneTimeSetUp(
    Blob<Dtype>* unary_terms,
    Blob<Dtype>* softmax_input,
    Blob<Dtype>* output_blob,
    const shared_ptr<ModifiedPermutohedral<Dtype> >& spatial_lattice,
    const Blob<Dtype>& spatial_norm) {
  num_ = unary_terms->num();
  channels_ = unary_terms->channels();
  num_pixels_ = unary_terms->height() * unary_terms->width();

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Meanfield iteration skipping parameter initialization.";
  } else {
    blobs_.resize(3);
    // spatial kernel weight
    blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, channels_));
    // bilateral kernel weight
    blobs_[1].reset(new Blob<Dtype>(1, 1, channels_, channels_));
    // compatibility transform matrix
    blobs_[2].reset(new Blob<Dtype>(1, 1, channels_, channels_));
  }

  pairwise_terms_.ReshapeLike(*unary_terms);
  spatial_out_blob_.ReshapeLike(*unary_terms);
  bilateral_out_blob_.ReshapeLike(*unary_terms);
  message_passing_.ReshapeLike(*unary_terms);

  // Addition configuration
  sum_bottom_vec_.clear();
  sum_bottom_vec_.push_back(unary_terms);  // unary terms
  sum_bottom_vec_.push_back(&pairwise_terms_);  // pairwise terms
  sum_top_vec_.clear();
  sum_top_vec_.push_back(output_blob);
  LayerParameter sum_param;
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(1.));
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(-1.));
  sum_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);
  sum_layer_.reset(new EltwiseLayer<Dtype>(sum_param));
  sum_layer_->SetUp(sum_bottom_vec_, sum_top_vec_);

  // Normalization configuration
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(softmax_input);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&softmax_output_);
  LayerParameter softmax_param;
  softmax_layer_.reset(new SoftmaxLayer<Dtype>(softmax_param));
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  spatial_lattice_ = spatial_lattice;
  spatial_norm_.CopyFrom(spatial_norm, false, true);
}

/**
 * To be invoked before every call to the Forward_cpu() method.
 */
template <typename Dtype>
void MeanfieldIteration<Dtype>::PrePass(
    const vector<shared_ptr<Blob<Dtype> > >& parameters_to_copy_from,
    const vector<shared_ptr<ModifiedPermutohedral<Dtype> > >& bilateral_lattices,
    const Blob<Dtype>& bilateral_norms) {
  bilateral_lattices_ = bilateral_lattices;
  bilateral_norms_.CopyFrom(bilateral_norms, false, true);
  // Get copies of the up-to-date parameters.
  for (int i = 0; i < parameters_to_copy_from.size(); ++i) {
    blobs_[i]->CopyFrom(*(parameters_to_copy_from[i].get()));
  }
}

/**
 * Forward pass during the inference.
 */
template <typename Dtype>
void MeanfieldIteration<Dtype>::Forward_cpu() {
  /*-------------------- Normalization --------------------*/
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  /*-------------------- Message Passing --------------------*/
  for (int n = 0; n < num_; ++n) {
    const Dtype* prob_input_data = softmax_output_.cpu_data() +
        softmax_output_.offset(n);
    // Gaussian filters: spatial
    Dtype* spatial_out_data = spatial_out_blob_.mutable_cpu_data() +
        spatial_out_blob_.offset(n);
    spatial_lattice_->compute(spatial_out_data, prob_input_data, channels_);
    // Gaussian filters: bilateral
    Dtype* bilateral_out_data = bilateral_out_blob_.mutable_cpu_data() +
        bilateral_out_blob_.offset(n);
    bilateral_lattices_[n]->compute(bilateral_out_data, prob_input_data,
        channels_);
    // Pixel-wise normalization.
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_,
          spatial_norm_.cpu_data(),
          spatial_out_data + channel_id * num_pixels_,
          spatial_out_data + channel_id * num_pixels_);
      caffe_mul(num_pixels_,
          bilateral_norms_.cpu_data() + bilateral_norms_.offset(n),
          bilateral_out_data + channel_id * num_pixels_,
          bilateral_out_data + channel_id * num_pixels_);
    }
  }

  /*-------------------- Weighting Filter Outputs --------------------*/
  caffe_set(message_passing_.count(), Dtype(0.),
      message_passing_.mutable_cpu_data());
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        channels_, num_pixels_, channels_,
        Dtype(1.),
        this->blobs_[0]->cpu_data(),
        spatial_out_blob_.cpu_data() + spatial_out_blob_.offset(n),
        Dtype(0.),
        message_passing_.mutable_cpu_data() + message_passing_.offset(n));
  }
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        channels_, num_pixels_, channels_,
        Dtype(1.),
        this->blobs_[1]->cpu_data(),
        bilateral_out_blob_.cpu_data() + bilateral_out_blob_.offset(n),
        Dtype(1.),
        message_passing_.mutable_cpu_data() + message_passing_.offset(n));
  }

  /*-------------------- Compatibility Transform --------------------*/
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
        channels_, num_pixels_, channels_,
        Dtype(1.),
        this->blobs_[2]->cpu_data(),
        message_passing_.cpu_data() + message_passing_.offset(n),
        Dtype(0.),
        pairwise_terms_.mutable_cpu_data() + pairwise_terms_.offset(n));
  }

  /*-------------------- Adding Unary Potentials --------------------*/
  sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);
}

template <typename Dtype>
void MeanfieldIteration<Dtype>::Backward_cpu() {
  /*-------------------- Add unary gradient --------------------*/
  vector<bool> eltwise_propagate_down(sum_bottom_vec_.size(), true);
  sum_layer_->Backward(sum_top_vec_, eltwise_propagate_down, sum_bottom_vec_);

  /*-------------------- Update compatibility diffs --------------------*/
  caffe_set(this->blobs_[2]->count(), Dtype(0.),
      this->blobs_[2]->mutable_cpu_diff());
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        channels_, channels_, num_pixels_,
        Dtype(1.),
        pairwise_terms_.cpu_diff() + pairwise_terms_.offset(n),
        message_passing_.cpu_data() + message_passing_.offset(n),
        Dtype(1.),
        this->blobs_[2]->mutable_cpu_diff());
  }

  /*----------------- Gradient after compatibility transform -----------------*/
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
        channels_, num_pixels_, channels_,
        Dtype(1.),
        this->blobs_[2]->cpu_data(),
        pairwise_terms_.cpu_diff() + pairwise_terms_.offset(n),
        Dtype(0.),
        message_passing_.mutable_cpu_diff() + message_passing_.offset(n));
  }

  /*-------------------- Gradient w.r.t. kernels weights --------------------*/
  caffe_set(this->blobs_[0]->count(), Dtype(0.),
      this->blobs_[0]->mutable_cpu_diff());
  caffe_set(this->blobs_[1]->count(), Dtype(0.),
      this->blobs_[1]->mutable_cpu_diff());

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        channels_, channels_, num_pixels_,
        Dtype(1.),
        message_passing_.cpu_diff() + message_passing_.offset(n),
        spatial_out_blob_.cpu_data() + spatial_out_blob_.offset(n),
        Dtype(1.),
        this->blobs_[0]->mutable_cpu_diff());
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
        channels_, channels_, num_pixels_,
        Dtype(1.),
        message_passing_.cpu_diff() + message_passing_.offset(n),
        bilateral_out_blob_.cpu_data() + bilateral_out_blob_.offset(n),
        Dtype(1.),
        this->blobs_[1]->mutable_cpu_diff());
  }

  // Check whether there's a way to improve the accuracy of this calculation.
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
        channels_, num_pixels_, channels_,
        Dtype(1.),
        this->blobs_[0]->cpu_data(),
        message_passing_.cpu_diff() + message_passing_.offset(n),
        Dtype(0.),
        spatial_out_blob_.mutable_cpu_diff() + spatial_out_blob_.offset(n));
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
        channels_, num_pixels_, channels_,
        Dtype(1.),
        this->blobs_[1]->cpu_data(),
        message_passing_.cpu_diff() + message_passing_.offset(n),
        Dtype(0.),
        bilateral_out_blob_.mutable_cpu_diff() + bilateral_out_blob_.offset(n));
  }

  /*-------------------- BP thru normalization --------------------*/
  for (int n = 0; n < num_; ++n) {
    Dtype* spatial_out_diff = spatial_out_blob_.mutable_cpu_diff() +
        spatial_out_blob_.offset(n);
    Dtype* bilateral_out_diff = bilateral_out_blob_.mutable_cpu_diff() +
        bilateral_out_blob_.offset(n);
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_,
          spatial_norm_.cpu_data(),
          spatial_out_diff + channel_id * num_pixels_,
          spatial_out_diff + channel_id * num_pixels_);
      caffe_mul(num_pixels_,
          bilateral_norms_.cpu_data() + bilateral_norms_.offset(n),
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

INSTANTIATE_CLASS(MeanfieldIteration);

}  // namespace caffe
