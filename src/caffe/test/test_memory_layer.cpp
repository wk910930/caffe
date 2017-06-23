#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/memory_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class MemoryLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MemoryLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    num_ = 4;
    channels_ = 128;
    height_ = 7;
    width_ = 7;
    Caffe::set_random_seed(1701);
    // Setup data
    blob_bottom_data_->Reshape(num_, channels_, height_, width_);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    // Setup label
    blob_bottom_label_->Reshape(num_, 1, 1, 1);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    // Setup top
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~MemoryLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
  }

  void TestForward_moving_average_fraction_zero() {
    LayerParameter layer_param;
    MemoryParameter* memory_param = layer_param.mutable_memory_param();
    int num_class = num_;  // Make num_class equal to batch for easy test
    memory_param->set_num_class(num_class);
    for (int i = 0; i < num_; ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = i;
    }
    memory_param->set_moving_average_fraction(0.0);

    MemoryLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    EXPECT_EQ(blob_top_->num(), num_);
    EXPECT_EQ(blob_top_->channels(), channels_);
    EXPECT_EQ(blob_top_->height(), height_);
    EXPECT_EQ(blob_top_->width(), width_);

    for (int i = 0; i < blob_bottom_data_->count(); ++i) {
      EXPECT_EQ(blob_bottom_data_->cpu_data()[i], blob_top_->cpu_data()[i]);
    }
  }

  void TestForward_moving_average_fraction_half() {
    LayerParameter layer_param;
    MemoryParameter* memory_param = layer_param.mutable_memory_param();
    int num_class = num_;  // Make num_class equal to batch for easy test
    memory_param->set_num_class(num_class);
    for (int i = 0; i < num_; ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = i;
    }
    memory_param->set_moving_average_fraction(0.5);

    MemoryLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    EXPECT_EQ(blob_top_->num(), num_);
    EXPECT_EQ(blob_top_->channels(), channels_);
    EXPECT_EQ(blob_top_->height(), height_);
    EXPECT_EQ(blob_top_->width(), width_);

    for (int i = 0; i < blob_bottom_data_->count(); ++i) {
      EXPECT_EQ(0.5 * blob_bottom_data_->cpu_data()[i],
          blob_top_->cpu_data()[i]);
    }
  }

  void TestForward_ignore_label() {
    LayerParameter layer_param;
    MemoryParameter* memory_param = layer_param.mutable_memory_param();
    int num_class = num_;  // Make num_class equal to batch for easy test
    memory_param->set_num_class(num_class);
    for (int i = 0; i < num_; ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = i;
    }
    memory_param->set_moving_average_fraction(0.9);
    memory_param->set_ignore_label(0);

    MemoryLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    EXPECT_EQ(blob_top_->num(), num_);
    EXPECT_EQ(blob_top_->channels(), channels_);
    EXPECT_EQ(blob_top_->height(), height_);
    EXPECT_EQ(blob_top_->width(), width_);

    const Dtype* bottom_data = blob_bottom_data_->cpu_data();
    const Dtype* top_data = blob_top_->cpu_data();
    for (int n = 0; n < num_; ++n) {
      int label_value = blob_bottom_label_->cpu_data()[n];
      int sample_length = blob_bottom_data_->count(1);
      for (int i = 0; i < sample_length; ++i) {
        if (label_value == 0) {
          EXPECT_EQ(bottom_data[i], top_data[i]);
        } else {
          EXPECT_NEAR(0.1 * bottom_data[i], top_data[i], 1e-4);
        }
      }
      bottom_data += blob_bottom_data_->offset(1);
      top_data += blob_top_->offset(1);
    }
  }

 private:
  int num_;
  int channels_;
  int height_;
  int width_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MemoryLayerTest, TestDtypesAndDevices);

TYPED_TEST(MemoryLayerTest, TestForward_moving_average_fraction_zero) {
  this->TestForward_moving_average_fraction_zero();
}

TYPED_TEST(MemoryLayerTest, TestForward_moving_average_fraction_half) {
  this->TestForward_moving_average_fraction_half();
}

TYPED_TEST(MemoryLayerTest, TestForward_ignore_label) {
  this->TestForward_ignore_label();
}

}  // namespace caffe
