#include <vector>

#include "caffe/layers/multi_stage_meanfield_layer.hpp"

namespace caffe {

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  split_layer_->Forward(split_bottom_vec_, split_top_vec_);
  // Initialize the bilateral lattices.
  init_bilateral_lattice(bottom[2]);
  for (int i = 0; i < num_iterations_; ++i) {
    meanfield_iterations_[i]->PrePass(this->blobs_,
        spatial_lattice_, spatial_norm_,
        bilateral_lattices_, bilateral_norms_);
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
  vector<bool> split_propagate_down(1, true);
  split_layer_->Backward(split_top_vec_, split_propagate_down,
      split_bottom_vec_);
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
