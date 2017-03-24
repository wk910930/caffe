#ifndef CAFFE_TVG_COMMON_UTILS_HPP_
#define CAFFE_TVG_COMMON_UTILS_HPP_

#include <string>

#include "caffe/blob.hpp"

namespace caffe {

template <typename Dtype>
void read_into_the_diagonal(const std::string& source, Blob<Dtype>* blob) {
  // Check blob dim
  CHECK_EQ(blob->num(), 1) << "num should be 1.";
  CHECK_EQ(blob->channels(), 1) << "channels should be 1.";
  CHECK_EQ(blob->height(), blob->width()) << "Only support square matrix.";
  // Initialize blob as a zero-matrix
  Dtype* data = blob->mutable_cpu_data();
  caffe_set(blob->count(), Dtype(0.), data);

  std::stringstream iss;
  iss.clear();
  iss << source;

  int height = blob->height();
  for (int i = 0; i < height; ++i) {
    std::string token;
    if (std::getline(iss, token, ' ')) {
      data[i * height + i] = std::stof(token);
    } else {
      throw std::runtime_error("A malformed string! >" + source
          + "<. Couldn't read " + std::to_string(height) + " values.");
    }
  }
}
template void read_into_the_diagonal(const std::string& source,
    Blob<float>* blob);
template void read_into_the_diagonal(const std::string& source,
    Blob<double>* blob);


template <typename Dtype>
void read_into_the_diagonal(Dtype value, Blob<Dtype>* blob) {
  // Check blob dim
  CHECK_EQ(blob->num(), 1) << "num should be 1.";
  CHECK_EQ(blob->channels(), 1) << "channels should be 1.";
  CHECK_EQ(blob->height(), blob->width()) << "Only support square matrix.";
  // Initialize blob as a zero-matrix
  Dtype* data = blob->mutable_cpu_data();
  caffe_set(blob->count(), Dtype(0.), data);
  int height = blob->height();
  for (int i = 0; i < height; ++i) {
    data[i * height + i] = value;
  }
}
template void read_into_the_diagonal(float value, Blob<float>* blob);
template void read_into_the_diagonal(double value, Blob<double>* blob);


template <typename Dtype>
void FillAsRGB(Blob<Dtype>* blob) {
  for (int i = 0; i < blob->count(); ++i) {
    blob->mutable_cpu_data()[i] = Dtype(caffe_rng_rand() % 256);
  }
}
template void FillAsRGB(Blob<float>* blob);
template void FillAsRGB(Blob<double>* blob);


template <typename Dtype>
void FillAsProb(Blob<Dtype>* blob) {
  for (int i = 0; i < blob->count(); ++i) {
    Dtype num = static_cast<Dtype>(caffe_rng_rand()) / RAND_MAX;
    blob->mutable_cpu_data()[i] =
        static_cast<Dtype>((num != Dtype(0.)) ? num : Dtype(0.0002));
  }
  for (int n = 0; n < blob->num(); ++n) {
    for (int h = 0; h < blob->height(); ++h) {
      for (int w = 0; w < blob->width(); ++w) {
        Dtype total = 0;
        for (int c = 0; c < blob->channels(); ++c) {
          total += blob->data_at(n, c, h, w);
        }
        for (int c = 0; c < blob->channels(); ++c) {
          blob->mutable_cpu_data()[blob->offset(n, c, h, w)] =
              blob->data_at(n, c, h, w) / total;
        }
      }
    }
  }
}
template void FillAsProb(Blob<float>* blob);
template void FillAsProb(Blob<double>* blob);


template <typename Dtype>
void FillAsLogProb(Blob<Dtype>* blob) {
  FillAsProb(blob);
  for (int i = 0; i < blob->count(); ++i) {
    blob->mutable_cpu_data()[i] = log(blob->cpu_data()[i]);
  }
}
template void FillAsLogProb(Blob<float>* blob);
template void FillAsLogProb(Blob<double>* blob);

}  // namespace caffe

#endif  // CAFFE_TVG_COMMON_UTILS_HPP_
