#include <algorithm>
#include <utility>

#include "caffe/util/hash_table_copy.hpp"
#include "caffe/util/modified_permutohedral.hpp"

namespace caffe {

/************************************************/
/***          ModifiedPermutohedral Lattice    ***/
/************************************************/

template <typename Dtype>
void ModifiedPermutohedral<Dtype>::init(const Dtype* features,
    int num_dimensions, int num_points) {
  // Compute the lattice coordinates for each feature.
  // There is going to be a lot of magic here
  N_ = num_points;
  d_ = num_dimensions;
  HashTableCopy hash_table(d_, N_*(d_+1));

  // Allocate the class memory
  offset_.resize((d_+1) * N_);
  rank_.resize((d_+1) * N_);
  barycentric_.resize((d_+1) * N_);

  // Allocate the local memory
  Dtype* scale_factor = new Dtype[d_];
  Dtype* elevated = new Dtype[d_+1];
  Dtype* rem0 = new Dtype[d_+1];
  Dtype* barycentric = new Dtype[d_+2];
  int* rank = new int[d_+1];
  int* canonical = new int[(d_+1)*(d_+1)];
  int* key = new int[d_+1];

  // Compute the canonical simplex
  for (int i = 0; i <= d_; ++i) {
    for (int j = 0; j <= d_ - i; ++j) {
      canonical[i * (d_+1) + j] = i;
    }
    for (int j = d_ - i + 1; j <= d_; ++j) {
      canonical[i * (d_+1) + j] = i - (d_+1);
    }
  }

  // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
  Dtype inv_std_dev = sqrt(2.0 / 3.0) * (d_+1);
  // Compute the diagonal part of E (p.5 in [Adams etal 2010])
  for (int i = 0; i < d_; ++i) {
    scale_factor[i] = 1.0 / sqrt(static_cast<double>((i+2)*(i+1)))
        * inv_std_dev;
  }
  // Compute the simplex each feature lies in
  for (int k = 0; k < N_; ++k) {
    // Elevate the features ( y = Ep, see p.5 in [Adams etal 2010])
    const Dtype* f = features + k * num_dimensions;
    // sm contains the sum of 1..n of our feature vector
    Dtype sm = 0;
    for (int j = d_; j > 0; j--) {
      Dtype cf = f[j-1] * scale_factor[j-1];
      elevated[j] = sm - j * cf;
      sm += cf;
    }
    elevated[0] = sm;

    // Find the closest 0-colored simplex through rounding
    Dtype down_factor = 1.0f / (d_ + 1);
    Dtype up_factor = d_ + 1;
    int sum = 0;
    for (int i = 0; i <= d_; ++i) {
      int rd2 = 0;
      Dtype v = down_factor * elevated[i];
      Dtype up = ceilf(v) * up_factor;
      Dtype down = floorf(v) * up_factor;
      if (up - elevated[i] < elevated[i] - down) {
        rd2 = static_cast<int>(up);
      } else {
        rd2 = static_cast<int>(down);
      }
      rem0[i] = rd2;
      sum += rd2 * down_factor;
    }
    // Find the simplex we are in and store it in rank
    // (where rank describes what position coordinate i
    // has in the sorted order of the features values)
    for (int i = 0; i <= d_; ++i) {
      rank[i] = 0;
    }
    for (int i = 0; i < d_; ++i) {
      double di = elevated[i] - rem0[i];
      for (int j = i+1; j <= d_; ++j) {
        if (di < elevated[j] - rem0[j]) {
          rank[i]++;
        } else {
          rank[j]++;
        }
      }
    }
    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= d_; ++i) {
      rank[i] += sum;
      if (rank[i] < 0) {
        rank[i] += d_ + 1;
        rem0[i] += d_ + 1;
      } else if (rank[i] > d_) {
        rank[i] -= d_ + 1;
        rem0[i] -= d_ + 1;
      }
    }
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for (int i = 0; i <= d_+1; ++i) {
      barycentric[i] = 0;
    }
    for (int i = 0; i <= d_; ++i) {
      Dtype v = (elevated[i] - rem0[i]) * down_factor;
      barycentric[d_ - rank[i]] += v;
      barycentric[d_ - rank[i] + 1] -= v;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[d_+1];

    // Compute all vertices and their offset
    for (int remainder = 0; remainder <= d_; ++remainder) {
      for (int i = 0; i < d_; ++i) {
        key[i] = rem0[i] + canonical[remainder*(d_+1) + rank[i]];
      }
      offset_[k*(d_+1) + remainder] = hash_table.find(key, true);
      rank_[k*(d_+1) + remainder] = rank[remainder];
      barycentric_[k*(d_+1) + remainder] = barycentric[remainder];
    }
  }
  delete [] scale_factor;
  delete [] elevated;
  delete [] rem0;
  delete [] barycentric;
  delete [] rank;
  delete [] canonical;
  delete [] key;

  // Find the Neighbors of each lattice point

  // Get the number of vertices in the lattice
  M_ = hash_table.size();

  // Create the neighborhood structure
  blur_neighbors_.resize((d_+1) * M_);

  int* n1 = new int[d_+1];
  int* n2 = new int[d_+1];

  // For each of d+1 axes,
  for (int j = 0; j <= d_; ++j) {
    for (int i = 0; i < M_; ++i) {
      const int* key = hash_table.getKey(i);
      for (int k = 0; k < d_; ++k) {
        n1[k] = key[k] - 1;
        n2[k] = key[k] + 1;
      }
      n1[j] = key[j] + d_;
      n2[j] = key[j] - d_;

      blur_neighbors_[j * M_ + i].n1 = hash_table.find(n1);
      blur_neighbors_[j * M_ + i].n2 = hash_table.find(n2);
    }
  }
  delete[] n1;
  delete[] n2;
}

template <typename Dtype>
void ModifiedPermutohedral<Dtype>::compute(Dtype* out, const Dtype* in,
    int value_size, bool reverse, bool add) const {
  // Shift all values by 1 such that -1 -> 0 (used for blurring)
  Dtype* values = new Dtype[(M_+2) * value_size];
  Dtype* new_values = new Dtype[(M_+2) * value_size];

  for (int i = 0; i < (M_+2) * value_size; ++i) {
    values[i] = new_values[i] = 0;
  }
  // Splatting
  for (int i = 0;  i < N_; ++i) {
    for (int j = 0; j <= d_; ++j) {
      int o = offset_[i*(d_+1) + j] + 1;
      Dtype w = barycentric_[i*(d_+1) + j];
      for (int k = 0; k < value_size; k++) {
        values[o * value_size + k] += w * in[k * N_ + i];
      }
    }
  }
  for (int j = reverse ? d_ : 0; j <= d_ && j >= 0; reverse ? j-- : j++) {
    for (int i = 0; i < M_; ++i) {
      Dtype* old_val = values + (i+1) * value_size;
      Dtype* new_val = new_values + (i+1) * value_size;

      int n1 = blur_neighbors_[j*M_+i].n1 + 1;
      int n2 = blur_neighbors_[j*M_+i].n2 + 1;
      Dtype* n1_val = values + n1 * value_size;
      Dtype* n2_val = values + n2 * value_size;
      for (int k = 0; k < value_size; ++k) {
        new_val[k] = old_val[k] + 0.5 * (n1_val[k] + n2_val[k]);
      }
    }
    std::swap(values, new_values);
  }
  // Alpha is a magic scaling constant
  // (write Andrew if you really wanna understand this)
  Dtype alpha = 1.0f / (1+powf(2, -d_));
  // Slicing
  for (int i = 0; i < N_; ++i) {
    if  (!add) {
      for (int k = 0; k < value_size; ++k) {
        out[i + k * N_] = 0;
      }
    }
    for (int j = 0; j <= d_; ++j) {
      int o = offset_[i * (d_+1) + j] + 1;
      Dtype w = barycentric_[i * (d_+1) + j];
      for (int k = 0; k < value_size; ++k) {
        out[i + k * N_] += w * values[o * value_size + k] * alpha;
      }
    }
  }
  delete[] values;
  delete[] new_values;
}

INSTANTIATE_CLASS(ModifiedPermutohedral);

}  // namespace caffe
