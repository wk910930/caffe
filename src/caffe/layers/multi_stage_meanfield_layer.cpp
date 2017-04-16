#include <vector>

#include "caffe/layers/multi_stage_meanfield_layer.hpp"
#include "caffe/util/tvg_common_utils.hpp"

namespace caffe {

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const MultiStageMeanfieldParameter& meanfield_param =
      this->layer_param_.multi_stage_meanfield_param();
  num_iterations_ = meanfield_param.num_iterations();
  CHECK_GT(num_iterations_, 1) << "num_iterations must be greater than 1.";
  // bandwidth values of Gaussian kernels
  theta_alpha_ = meanfield_param.theta_alpha();
  theta_beta_ = meanfield_param.theta_beta();
  theta_gamma_ = meanfield_param.theta_gamma();

  eps_ = meanfield_param.eps();

  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_pixels_ = height_ * width_;

  spatial_dim_ = 2;
  bilateral_dim_ = 2 + bottom[2]->channels();

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // blobs_[0] - spatial kernel weights
    // blobs_[1] - bilateral kernel weights
    // blobs_[2] - compatibility matrix
    this->blobs_.resize(3);
    // Allocate space for kernel weights.
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, channels_));
    this->blobs_[1].reset(new Blob<Dtype>(1, 1, channels_, channels_));
    this->blobs_[2].reset(new Blob<Dtype>(1, 1, channels_, channels_));
    // Initialize the kernels weights.
    read_into_the_diagonal(meanfield_param.spatial_filter_weights_str(),
        this->blobs_[0].get());
    read_into_the_diagonal(meanfield_param.bilateral_filter_weights_str(),
        this->blobs_[1].get());
    // Initialize the compatibility matrix.
    switch (meanfield_param.compatibility_mode()) {
    case MultiStageMeanfieldParameter_Mode_POTTS:
      read_into_the_diagonal(Dtype(-1.), this->blobs_[2].get());
      break;
    case MultiStageMeanfieldParameter_Mode_POTTS_ZERO:
      read_into_the_diagonal(Dtype(-1.), this->blobs_[2].get());
      caffe_add_scalar(this->blobs_[2]->count(), Dtype(1.),
          this->blobs_[2]->mutable_cpu_data());
      break;
    default:
      LOG(FATAL) << "Unknown compatibility mode.";
    }
  }  // parameter initialization

  norm_feed_.Reshape(1, 1, height_, width_);
  caffe_set(norm_feed_.count(), Dtype(1.), norm_feed_.mutable_cpu_data());

  // Initialize the spatial lattice.
  // This does not need to be computed for each image for the fixed size.
  init_spatial_lattice();

  // Configure the split layer that is used to make copies of the unary term.
  // One copy for each iteration.
  // It may be possible to optimize this calculation later.
  LayerParameter split_param(this->layer_param_);
  split_param.set_type("Split");
  split_layer_ = LayerRegistry<Dtype>::CreateLayer(split_param);
  split_bottom_vec_.clear();
  split_bottom_vec_.push_back(bottom[0]);
  split_top_vec_.clear();
  split_top_blobs_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; ++i) {
    split_top_blobs_[i].reset(new Blob<Dtype>());
    split_top_vec_.push_back(split_top_blobs_[i].get());
  }
  split_layer_->SetUp(split_bottom_vec_, split_top_vec_);

  // Make blobs to store outputs of each meanfield iteration.
  // Output of the last iteration is stored in top[0].
  // So we need only (num_iterations_ - 1) blobs.
  iteration_output_blobs_.resize(num_iterations_ - 1);
  for (int i = 0; i < num_iterations_ - 1; ++i) {
    iteration_output_blobs_[i].reset(new Blob<Dtype>(bottom[0]->shape()));
  }
  // Make instances of MeanfieldIteration and initialize them.
  meanfield_iterations_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; ++i) {
    meanfield_iterations_[i].reset(new MeanfieldIteration<Dtype>());
    meanfield_iterations_[i]->OneTimeSetUp(
        split_top_blobs_[i].get(),
        (i == 0) ? bottom[1] : iteration_output_blobs_[i - 1].get(),
        (i == num_iterations_ - 1) ? top[0] : iteration_output_blobs_[i].get(),
        spatial_lattice_, spatial_norm_);
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]->shape() == bottom[1]->shape());
  CHECK_EQ(num_, bottom[0]->num()) << "Currently the num should be fixed.";
  CHECK_EQ(num_, bottom[2]->num())
      << "num does not match between unary terms and image.";
  CHECK_EQ(height_, bottom[2]->height())
      << "height does not match between unary terms and image.";
  CHECK_EQ(width_, bottom[2]->width())
      << "width does not match between unary terms and image.";
  top[0]->ReshapeLike(*bottom[0]);
}

/**
 * Performs filter-based mean field inference given the image and unary.
 *
 * bottom[0] - Unary terms
 * bottom[1] - Softmax input/Output from the previous iteration (a copy of the unary terms if this is the first stage).
 * bottom[2] - Image features (e.g. RGB image)
 *
 * top[0] - Output of the mean field inference (not normalized).
 */
template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  split_layer_->Forward(split_bottom_vec_, split_top_vec_);
  // Initialize the bilateral lattices.
  init_bilateral_lattice(bottom[2]);
  for (int i = 0; i < num_iterations_; ++i) {
    meanfield_iterations_[i]->PrePass(
        this->blobs_, bilateral_lattices_, bilateral_norms_);
    meanfield_iterations_[i]->Forward_cpu();
  }
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  for (int i = num_iterations_ - 1; i >= 0; --i) {
    meanfield_iterations_[i]->Backward_cpu();
  }
  vector<bool> split_propagate_down(1, true);
  split_layer_->Backward(split_top_vec_, split_propagate_down,
      split_bottom_vec_);
  // Accumulate diffs from mean field iterations.
  for (int blob_id = 0; blob_id < this->blobs_.size(); ++blob_id) {
    if (this->param_propagate_down_[blob_id]) {
      Blob<Dtype>* cur_blob = this->blobs_[blob_id].get();
      caffe_set(cur_blob->count(), Dtype(0.), cur_blob->mutable_cpu_diff());
      for (int i = 0; i < num_iterations_; ++i) {
        const Dtype* diffs_to_add =
            meanfield_iterations_[i]->blobs()[blob_id]->cpu_diff();
        caffe_axpy(cur_blob->count(), Dtype(1.),
            diffs_to_add, cur_blob->mutable_cpu_diff());
      }
    }
  }
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::init_spatial_lattice() {
  spatial_norm_.Reshape(1, 1, height_, width_);
  caffe_set(spatial_norm_.count(), Dtype(0.), spatial_norm_.mutable_cpu_data());
  compute_spatial_kernel();
  spatial_lattice_.init(spatial_kernel_buffer_.cpu_data(),
      spatial_dim_, num_pixels_);
  // Calculate spatial filter normalization factors.
  Dtype* norm_output_data = spatial_norm_.mutable_cpu_data();
  spatial_lattice_.compute(norm_output_data, norm_feed_.cpu_data(), 1);
  for (int i = 0; i < num_pixels_; ++i) {
    norm_output_data[i] = Dtype(1.) / (norm_output_data[i] + eps_);
  }
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::init_bilateral_lattice(
    const Blob<Dtype>* image_features) {
  bilateral_lattices_.clear();
  bilateral_lattices_.resize(num_);
  bilateral_norms_.Reshape(num_, 1, height_, width_);
  caffe_set(bilateral_norms_.count(), Dtype(0.),
      bilateral_norms_.mutable_cpu_data());
  for (int n = 0; n < num_; ++n) {
    compute_bilateral_kernel(image_features->cpu_data() +
        image_features->offset(n));
    bilateral_lattices_[n].init(bilateral_kernel_buffer_.cpu_data(),
        bilateral_dim_, num_pixels_);
    // Calculate bilateral filter normalization factors.
    Dtype* norm_output_data = bilateral_norms_.mutable_cpu_data() +
        bilateral_norms_.offset(n);
    bilateral_lattices_[n].compute(norm_output_data, norm_feed_.cpu_data(), 1);
    for (int i = 0; i < num_pixels_; ++i) {
      norm_output_data[i] = Dtype(1.) / (norm_output_data[i] + eps_);
    }
  }
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::compute_spatial_kernel() {
  spatial_kernel_buffer_.Reshape(1, spatial_dim_, height_, width_);
  Dtype* spatial_data = spatial_kernel_buffer_.mutable_cpu_data();
  caffe_set(spatial_kernel_buffer_.count(), Dtype(0.), spatial_data);
  for (int p = 0; p < num_pixels_; ++p) {
    spatial_data[spatial_dim_ * p + 0] =
        static_cast<Dtype>(p % width_) / theta_gamma_;
    spatial_data[spatial_dim_ * p + 1] =
        static_cast<Dtype>(p / width_) / theta_gamma_;
  }
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::compute_bilateral_kernel(
      const Dtype* image_features) {
  bilateral_kernel_buffer_.Reshape(1, bilateral_dim_, height_, width_);
  Dtype* bilateral_data = bilateral_kernel_buffer_.mutable_cpu_data();
  caffe_set(bilateral_dim_ * num_pixels_, Dtype(0.), bilateral_data);
  for (int p = 0; p < num_pixels_; ++p) {
    bilateral_data[bilateral_dim_ * p + 0] =
        static_cast<Dtype>(p % width_) / theta_alpha_;
    bilateral_data[bilateral_dim_ * p + 1] =
        static_cast<Dtype>(p / width_) / theta_alpha_;
    // Appearance
    for (int c = 0; c < bilateral_dim_ - 2; ++c) {
      bilateral_data[bilateral_dim_ * p + 2 + c] =
          static_cast<Dtype>((image_features + num_pixels_ * c)[p]) /
          theta_beta_;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiStageMeanfieldLayer);
#endif

INSTANTIATE_CLASS(MultiStageMeanfieldLayer);
REGISTER_LAYER_CLASS(MultiStageMeanfield);

}  // namespace caffe
