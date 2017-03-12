#ifndef CAFFE_TVG_COMMON_UTILS_HPP_
#define CAFFE_TVG_COMMON_UTILS_HPP_

#include <string>

#include "caffe/blob.hpp"

namespace caffe {

template <typename Dtype>
void read_into_the_diagonal(const std::string& source, Blob<Dtype>& blob) {
  const int height = blob.height();
  Dtype *data = blob.mutable_cpu_data();
  caffe_set(blob.count(), Dtype(0.), data);

  std::stringstream iss;
  iss.clear();
  iss << source;
  std::string token;

  for (int i = 0; i < height; ++i) {
    if (std::getline(iss, token, ' ')) {
      data[i * height + i] = std::stof(token);
    } else {
      throw std::runtime_error(
          "A malformed string! >" + source + "<. Couldn't read "
          + std::to_string(height) + " values.");
    }
  }
}
template void read_into_the_diagonal(const std::string& source,
    Blob<float>& blob);
template void read_into_the_diagonal(const std::string& source,
    Blob<double>& blob);


template <typename Dtype>
void PrintBlob(const Blob<Dtype>* blob, bool print_diff = false,
    const char* info = 0) {
  const Dtype* data = print_diff ? blob->cpu_diff() : blob->cpu_data();
  if (info != 0) {
    printf("%s: \n", info);
  }
  for (int n = 0; n < blob->num(); n++) {
    for (int c = 0; c < blob->channels(); c++) {
      for (int h = 0; h < blob->height(); h++) {
        for (int w = 0; w < blob->width(); w++) {
          int offset = ((n * blob->channels() + c) * blob->height() + h) * blob->width() + w;
          printf("%11.6f ", *(data + offset));
        }
        printf("\n");
      }
      printf("\n");
    }
  }
  printf("-- End of Blob --\n\n");
}
template void PrintBlob(const Blob<float>* blob, bool print_diff = false,
    const char* info = 0);
template void PrintBlob(const Blob<double>* blob, bool print_diff = false,
    const char* info = 0);


template <typename Dtype>
void FillAsRGB(Blob<Dtype>* blob, bool time_seed = false) {
  if (time_seed) {
    srand(time(NULL));
  } else {
    srand(999);
  }
  for (int i = 0; i < blob->count(); ++i) {
    blob->mutable_cpu_data()[i] = rand() % 256;
  }
}
template void FillAsRGB(Blob<float>* const blob, bool time_seed = false);
template void FillAsRGB(Blob<double>* const blob, bool time_seed = false);


template<typename Dtype>
void FillAsProb(Blob<Dtype>* blob, bool time_seed = false) {
  if (time_seed) {
    srand(time(NULL));
  } else {
    srand(999);
  }
  for (int i = 0; i < blob->count(); ++i) {
    double num = (double) rand() / (double) RAND_MAX;
    blob->mutable_cpu_data()[i] = static_cast<Dtype>((num != 0) ? num : 0.0002);
  }
  for (int n = 0; n < blob->num(); ++n) {
    for (int h = 0; h < blob->height(); ++h) {
      for (int w = 0; w < blob->width(); ++w) {
        Dtype total = 0;
        for (int c = 0; c < blob->channels(); ++c) {
          total += blob->data_at(n, c, h, w);
        }
        for (int c = 0; c < blob->channels(); ++c) {
          blob->mutable_cpu_data()[blob->offset(n, c, h, w)] = blob->data_at(n, c, h, w) / total;
        }
      }
    }
  }
}
template void FillAsProb(Blob<float>* const blobm, bool time_seed = false);
template void FillAsProb(Blob<double>* const blob, bool time_seed = false);


template<typename Dtype>
void FillAsLogProb(Blob<Dtype>* blob) {
  FillAsProb(blob);
  for (int i = 0; i < blob->count(); ++i) {
    blob->mutable_cpu_data()[i] = log(blob->cpu_data()[i]);
  }
}
template void FillAsLogProb(Blob<float>* const blob);
template void FillAsLogProb(Blob<double>* const blob);

}  // namespace caffe

#endif  // CAFFE_TVG_COMMON_UTILS_HPP_
