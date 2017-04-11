#ifndef CAFFE_HASH_TABLE_HPP_
#define CAFFE_HASH_TABLE_HPP_

#include <vector>

#include "caffe/common.hpp"

namespace caffe {

/*****************************************/
/***                Hash Table                ***/
/*****************************************/

/* Description
The idea of hashing is to distribute the entries (key/value pairs) across
an array of buckets.
Given a key, the algorithm computes an index that suggests where
the entry can be found:

--------------------------------------------
|  index = f(key, array_size)  |
--------------------------------------------

Often this is done in two steps:

---------------------------------------------
|  hash = hashfunc(key)         |
|  index = hash % array_size  |
---------------------------------------------

In the case that the array size is a power of two,
the remainder operation is reduced to masking, which improves speed,
but can increase problems with a poor hash function.
*/

class HashTable{
 public:
  HashTable(int key_size, int num_elements);

  inline int size() const { return filled_; }
  inline const int* getKey(int i) const { return &keys_[i * key_size_]; }
  /*
   * hash function
   *
   * A good hash function and implementation algorithm are essential for good hash table performance.
   * A basic requirement is that the function should provide a uniform distribution of hash values.
   * A non-uniform distribution increases the number of collisions and the cost of resolving them.
   * Uniformity is sometimes difficult to ensure by design,
   * but may be evaluated empirically using statistical tests.
  */
  inline size_t hashfunc(const int* key) const {
    size_t r = 0;
    for (size_t i = 0; i < key_size_; ++i) {
      r += key[i];
      r *= 1664525;
    }
    return r;
  }

  int find(const int* key, bool create = false);

 private:
  size_t filled_;
  size_t key_size_;
  size_t capacity_;
  std::vector<int> keys_;
  std::vector<int> table_;
};

}  // namespace caffe

#endif  // CAFFE_HASH_TABLE_HPP_
