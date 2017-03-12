#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/multi_stage_meanfield_layer.hpp"
#include "caffe/util/tvg_common_utils.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MultiStageMeanfieldLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MultiStageMeanfieldLayerTest()
      : unary_terms_blob_(new Blob<Dtype>(1, 5, 6, 6)),
        previous_output_blob_(new Blob<Dtype>(1, 5, 6, 6)),
        rgb_blob_(new Blob<Dtype>(1, 3, 6, 6)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    caffe::FillAsLogProb(unary_terms_blob_);
    caffe::FillAsLogProb(previous_output_blob_);
    caffe::FillAsRGB(rgb_blob_);
    blob_bottom_vec_.push_back(unary_terms_blob_);
    blob_bottom_vec_.push_back(previous_output_blob_);
    blob_bottom_vec_.push_back(rgb_blob_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MultiStageMeanfieldLayerTest() {
    delete unary_terms_blob_;
    delete previous_output_blob_;
    delete rgb_blob_;
    delete blob_top_;
  }

  Blob<Dtype>* const unary_terms_blob_;
  Blob<Dtype>* const previous_output_blob_;
  Blob<Dtype>* const rgb_blob_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MultiStageMeanfieldLayerTest, TestDtypesAndDevices);

// Top blob should have the same shape as the unary term
TYPED_TEST(MultiStageMeanfieldLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultiStageMeanfieldParameter* ms_mf_param =
      layer_param.mutable_multi_stage_meanfield_param();
  ms_mf_param->set_num_iterations(2);
  ms_mf_param->set_bilateral_filter_weights_str("5 5 5 5 5");
  ms_mf_param->set_spatial_filter_weights_str("3 3 3 3 3");
  ms_mf_param->set_compatibility_mode(MultiStageMeanfieldParameter_Mode_POTTS);
  ms_mf_param->set_theta_alpha(5);
  ms_mf_param->set_theta_beta(2);
  ms_mf_param->set_theta_gamma(3);

  MultiStageMeanfieldLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 6);
}

// Unary term should never change during the mean filed iteration
TYPED_TEST(MultiStageMeanfieldLayerTest, TestUnaryTerms) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultiStageMeanfieldParameter* ms_mf_param =
      layer_param.mutable_multi_stage_meanfield_param();
  ms_mf_param->set_num_iterations(5);
  ms_mf_param->set_bilateral_filter_weights_str("5 5 5 5 5");
  ms_mf_param->set_spatial_filter_weights_str("3 3 3 3 3");
  ms_mf_param->set_compatibility_mode(MultiStageMeanfieldParameter_Mode_POTTS);
  ms_mf_param->set_theta_alpha(5);
  ms_mf_param->set_theta_beta(2);
  ms_mf_param->set_theta_gamma(3);

  MultiStageMeanfieldLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  Blob<Dtype> ori_unary_terms_blob(1, 5, 6, 6);
  caffe::FillAsLogProb(&ori_unary_terms_blob);
  EXPECT_EQ(ori_unary_terms_blob.count(), this->unary_terms_blob_->count());
  const int count = this->blob_top_->count();
  EXPECT_EQ(count, this->unary_terms_blob_->count());

  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(this->unary_terms_blob_->cpu_data()[i],
        ori_unary_terms_blob.cpu_data()[i]);
  }
}

TYPED_TEST(MultiStageMeanfieldLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultiStageMeanfieldParameter* ms_mf_param =
      layer_param.mutable_multi_stage_meanfield_param();
  ms_mf_param->set_num_iterations(2);
  ms_mf_param->set_bilateral_filter_weights_str("5 5 5 5 5");
  ms_mf_param->set_spatial_filter_weights_str("3 3 3 3 3");
  ms_mf_param->set_compatibility_mode(MultiStageMeanfieldParameter_Mode_POTTS);
  ms_mf_param->set_theta_alpha(5);
  ms_mf_param->set_theta_beta(2);
  ms_mf_param->set_theta_gamma(3);

  MultiStageMeanfieldLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  // Check gradients w.r.t. unary terms
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  // Check gradients w.r.t. previous outputs
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}

}  // namespace caffe
