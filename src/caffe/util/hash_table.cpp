#include <algorithm>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/hash_table.hpp"

namespace caffe {

HashTable::HashTable(int key_size, int num_elements) {
  filled_ = 0;
  key_size_ = key_size;
  capacity_ = 2 * num_elements;
  keys_.resize((capacity_ / 2 + 10) * key_size_);
  table_.resize(capacity_, -1);
}

int HashTable::find(const int* key, bool create) {
  if (2 * filled_ >= capacity_) {
    // Create the new memory and copy the values in
    int old_capacity = capacity_;
    capacity_ *= 2;
    std::vector<int> old_keys((old_capacity + 10) * key_size_);
    std::copy(keys_.begin(), keys_.end(), old_keys.begin());
    std::vector<int> old_table(capacity_, -1);
    // Swap the memory
    table_.swap(old_table);
    keys_.swap(old_keys);
    // Reinsert each element
    for (int i = 0; i < old_capacity; ++i) {
      if (old_table[i] >= 0) {
        int e = old_table[i];
        size_t h = hashfunc(getKey(e)) % capacity_;
        while (table_[h] >= 0) {
          h = h < (capacity_ - 1) ? h+1 : 0;
        }
        table_[h] = e;
      }
    }
  }
  if (std::find(table_.begin(), table_.end(), -1) == table_.end()) {
    LOG(FATAL) << "We are entering infinite loop.";
  }
  size_t index = hashfunc(key) % capacity_;
  // Find the element with the right key, using linear probing
  while (true) {
    int e = table_[index];
    if (e == -1) {
      if (create) {
        // Insert a new key
        for (size_t i = 0; i < key_size_; i++) {
          keys_[filled_*key_size_ + i] = key[i];
        }
        // Return the new id
        return table_[index] = filled_++;
      } else {
        return -1;
      }
    }
    // Check if the current key is The One
    bool found = true;
    for (size_t i = 0; i < key_size_; ++i) {
      if (keys_[e * key_size_ + i] != key[i]) {
        found = false;
        break;
      }
    }
    if (found) {
      return e;
    }
    // Continue searching
    index++;
    if (index == capacity_) {
      index = 0;
    }
  }
}

}  // namespace caffe
