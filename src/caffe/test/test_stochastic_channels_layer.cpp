#include <vector>

#include "caffe/layers/stochastic_channels_layer.hpp"
#include "caffe/util/tvg_common_utils.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class StochasticChannelsLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  StochasticChannelsLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    num_ = 8;
    channels_ = 16;
    height_ = 3;
    width_ = 5;
    blob_bottom_->Reshape(num_, channels_, height_, width_);
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < blob_bottom_->count(); i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      blob_bottom_->mutable_cpu_data()[i +  5] = 9;
      blob_bottom_->mutable_cpu_data()[i +  6] = 4;
      blob_bottom_->mutable_cpu_data()[i +  7] = 1;
      blob_bottom_->mutable_cpu_data()[i +  8] = 4;
      blob_bottom_->mutable_cpu_data()[i +  9] = 8;
      blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 3;
    }
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~StochasticChannelsLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  int num_ = 0;
  int channels_ = 0;
  int height_ = 0;
  int width_ = 0;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForward_Specify_Channels() {
    int num_selected = caffe_rng_rand() % channels_ + 1;  // [1, channels]

    LayerParameter layer_param;
    StochasticChannelsParameter* stochastic_channels_param =
        layer_param.mutable_stochastic_channels_param();
    for (int i = 0; i < num_selected; ++i) {
      int rnd_channel_id = caffe_rng_rand() % channels_;  // [0, channels-1]
      stochastic_channels_param->add_channel_id(rnd_channel_id);
    }
    StochasticChannelsLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    EXPECT_EQ(blob_top_->num(), num_);
    EXPECT_EQ(blob_top_->channels(), num_selected);
    EXPECT_EQ(blob_top_->height(), height_);
    EXPECT_EQ(blob_top_->width(), width_);

    for (int i = 0; i < blob_top_->count(); i += 15) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 3);
    }
  }

  void TestForward_Random_Channels() {
    int num_output = caffe_rng_rand() % channels_ + 1;

    LayerParameter layer_param;
    StochasticChannelsParameter* stochastic_channels_param =
        layer_param.mutable_stochastic_channels_param();
    stochastic_channels_param->set_num_output(num_output);
    StochasticChannelsLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    EXPECT_EQ(blob_top_->num(), num_);
    EXPECT_EQ(blob_top_->channels(), num_output);
    EXPECT_EQ(blob_top_->height(), height_);
    EXPECT_EQ(blob_top_->width(), width_);

    for (int i = 0; i < blob_top_->count(); i += 15) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 3);
    }
  }

  void TestForward_All_Channels() {
    LayerParameter layer_param;
    StochasticChannelsLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    EXPECT_EQ(blob_top_->num(), num_);
    EXPECT_EQ(blob_top_->channels(), channels_);
    EXPECT_EQ(blob_top_->height(), height_);
    EXPECT_EQ(blob_top_->width(), width_);

    for (int i = 0; i < blob_bottom_->count(); ++i) {
      EXPECT_EQ(blob_bottom_->cpu_data()[i], blob_top_->cpu_data()[i]);
    }
  }

  void TestBackward() {
    LayerParameter layer_param;
    StochasticChannelsLayer<Dtype> layer(layer_param);

    GradientChecker<Dtype> checker(1e-3, 1e-4);
    checker.CheckGradientExhaustive(&layer, blob_bottom_vec_, blob_top_vec_);
  }
};

TYPED_TEST_CASE(StochasticChannelsLayerTest, TestDtypesAndDevices);

TYPED_TEST(StochasticChannelsLayerTest, TestForward_Specify_Channels) {
  this->TestForward_Specify_Channels();
}

TYPED_TEST(StochasticChannelsLayerTest, TestForward_Random_Channels) {
  this->TestForward_Random_Channels();
}

TYPED_TEST(StochasticChannelsLayerTest, TestForward_All_Channels) {
  this->TestForward_All_Channels();
}

TYPED_TEST(StochasticChannelsLayerTest, TestBackward) {
  this->TestBackward();
}

}  // namespace caffe
