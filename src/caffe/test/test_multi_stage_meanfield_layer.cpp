#include <vector>

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
      : unary_terms_blob_(new Blob<Dtype>(2, 4, 5, 5)),
        previous_output_blob_(new Blob<Dtype>(2, 4, 5, 5)),
        rgb_blob_(new Blob<Dtype>(2, 3, 5, 5)),
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
  ms_mf_param->set_theta_alpha(5);
  ms_mf_param->set_theta_beta(2);
  ms_mf_param->set_theta_gamma(3);
  ms_mf_param->set_num_iterations(2);
  ms_mf_param->set_spatial_filter_weights_str("3 3 3 3");
  ms_mf_param->set_bilateral_filter_weights_str("5 5 5 5");

  MultiStageMeanfieldLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(MultiStageMeanfieldLayerTest, TestUnaryTermGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultiStageMeanfieldParameter* ms_mf_param =
      layer_param.mutable_multi_stage_meanfield_param();
  ms_mf_param->set_theta_alpha(5);
  ms_mf_param->set_theta_beta(2);
  ms_mf_param->set_theta_gamma(3);
  ms_mf_param->set_num_iterations(2);
  ms_mf_param->set_spatial_filter_weights_str("3 3 3 3");
  ms_mf_param->set_bilateral_filter_weights_str("5 5 5 5");

  MultiStageMeanfieldLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  // Check gradients w.r.t. unary terms
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(MultiStageMeanfieldLayerTest, TestPreviousGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultiStageMeanfieldParameter* ms_mf_param =
      layer_param.mutable_multi_stage_meanfield_param();
  ms_mf_param->set_theta_alpha(5);
  ms_mf_param->set_theta_beta(2);
  ms_mf_param->set_theta_gamma(3);
  ms_mf_param->set_num_iterations(2);
  ms_mf_param->set_spatial_filter_weights_str("3 3 3 3");
  ms_mf_param->set_bilateral_filter_weights_str("5 5 5 5");

  MultiStageMeanfieldLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  // Check gradients w.r.t. previous outputs
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}

}  // namespace caffe
