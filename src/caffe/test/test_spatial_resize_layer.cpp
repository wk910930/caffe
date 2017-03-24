#include <vector>

#include "caffe/layers/spatial_resize_layer.hpp"
#include "caffe/util/tvg_common_utils.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class SpatialResizeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SpatialResizeLayerTest()
      : blob_bottom_(new Blob<Dtype>(3, 5, 8, 8)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SpatialResizeLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SpatialResizeLayerTest, TestDtypesAndDevices);

TYPED_TEST(SpatialResizeLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ResizeParameter* resize_param = layer_param.mutable_resize_param();
  resize_param->set_height(4);
  resize_param->set_width(4);
  SpatialResizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 4);
}

TYPED_TEST(SpatialResizeLayerTest, TestForwardRGB) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ResizeParameter* resize_param = layer_param.mutable_resize_param();
  resize_param->set_height(8);
  resize_param->set_width(8);
  SpatialResizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Fill the values
  FillAsRGB(this->blob_bottom_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  int count = this->blob_bottom_->count();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

TYPED_TEST(SpatialResizeLayerTest, TestForwardProbs) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ResizeParameter* resize_param = layer_param.mutable_resize_param();
  resize_param->set_height(8);
  resize_param->set_width(8);
  SpatialResizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Fill the values
  FillAsProb(this->blob_bottom_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  int count = this->blob_bottom_->count();
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

}  // namespace caffe
