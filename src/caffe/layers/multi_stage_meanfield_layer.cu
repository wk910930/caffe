#include <vector>

#include "caffe/layers/multi_stage_meanfield_layer.hpp"

namespace caffe {

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  split_layer_->Forward(split_layer_bottom_vec_, split_layer_top_vec_);
  // Initialize the bilateral lattices.
  bilateral_lattices_.clear();
  bilateral_lattices_.resize(num_);
  bilateral_norms_.Reshape(num_, 1, height_, width_);
  caffe_set(bilateral_norms_.count(), Dtype(0.),
      bilateral_norms_.mutable_cpu_data());
  for (int n = 0; n < num_; ++n) {
    const Dtype* image_features = bottom[2]->cpu_data() + bottom[2]->offset(n);
    compute_bilateral_kernel(image_features);
    bilateral_lattices_[n].reset(new ModifiedPermutohedral<Dtype>());
    bilateral_lattices_[n]->init(bilateral_kernel_buffer_.cpu_data(),
        bilateral_dim_, num_pixels_);
    // Calculate bilateral filter normalization factors.
    Dtype* norm_output_data = bilateral_norms_.mutable_cpu_data() +
        bilateral_norms_.offset(n);
    bilateral_lattices_[n]->compute(norm_output_data, norm_feed_.cpu_data(), 1);
    for (int i = 0; i < num_pixels_; ++i) {
      norm_output_data[i] = Dtype(1.) / (norm_output_data[i] + eps_);
    }
  }
  for (int i = 0; i < num_iterations_; ++i) {
    meanfield_iterations_[i]->PrePass(
        this->blobs_, bilateral_lattices_, bilateral_norms_);
    meanfield_iterations_[i]->Forward_gpu();
  }
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  for (int i = num_iterations_ - 1; i >= 0; --i) {
    meanfield_iterations_[i]->Backward_gpu();
  }
  vector<bool> split_layer_propagate_down(1, true);
  split_layer_->Backward(split_layer_top_vec_, split_layer_propagate_down,
      split_layer_bottom_vec_);
  // Accumulate diffs from mean field iterations.
  for (int blob_id = 0; blob_id < this->blobs_.size(); ++blob_id) {
    if (this->param_propagate_down_[blob_id]) {
      Blob<Dtype>* cur_blob = this->blobs_[blob_id].get();
      caffe_gpu_set(cur_blob->count(), Dtype(0.), cur_blob->mutable_gpu_diff());
      for (int i = 0; i < num_iterations_; ++i) {
        const Dtype* diffs_to_add =
            meanfield_iterations_[i]->blobs()[blob_id]->gpu_diff();
        caffe_gpu_axpy(cur_blob->count(), Dtype(1.),
            diffs_to_add, cur_blob->mutable_gpu_diff());
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiStageMeanfieldLayer);

}  // namespace caffe
