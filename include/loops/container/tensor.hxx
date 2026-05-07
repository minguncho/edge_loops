#pragma once

#include <loops/container/formats.hxx>
#include <loops/container/detail/convert.hxx>
#include <loops/container/vector.hxx>
#include <loops/error.hxx>
#include <loops/memory.hxx>

#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>

namespace loops {

using namespace memory;

template <typename index_t,
          typename value_t,
          memory_space_t space = memory_space_t::device>
struct tensor_t {
  std::string name;
  vector_t<char, space> ranks;
  vector_t<std::size_t, space> dims;
  std::size_t nnzs;

  vector_t<index_t, space> indices;
  vector_t<value_t, space> values;

  /**
   * @brief Construct a new tensor object with everything initialized to zero.
   *
   */
  tensor_t() : name(""), ranks(), dims(), nnzs(0), indices(), values() {}

  /**
   * @brief Construct a new tensor object.
   *
   * @param name Name of tensor
   * @param ranks List of rank labels
   * @param dims List of size if each dimension
   * @param nnzs Number of non-zero elements.
   * @param indices List of coordinates of each rank to a value (Stored in 1-D
   * vector, stride by values.size())
   * @param values List of values
   */
  tensor_t(std::string name,
           vector_t<char, space>& ranks,
           vector_t<std::size_t, space>& dims,
           std::size_t nnzs,
           vector_t<index_t, space>& indices,
           vector_t<value_t, space>& values)
      : name(name),
        ranks(ranks),
        dims(dims),
        nnzs(nnzs),
        indices(indices),
        values(values) {}

  /**
   * @brief Construct a new tensor from another tensor object on host/device.
   *
   * @param rhs tensor_t<index_t, value_t, rhs_space>
   */
  template <auto rhs_space>
  tensor_t(const tensor_t<index_t, value_t, rhs_space>& rhs)
      : name(rhs.name),
        ranks(rhs.ranks),
        dims(rhs.dims),
        nnzs(rhs.nnzs),
        indices(rhs.indices),
        values(rhs.values) {}

  /**
   * @brief Construct a new tensor from coordinate format (COO).
   *
   * @param coo coo_t<index_t, value_t, auto>
   */
  template <auto rhs_space>
  tensor_t(std::string name,
           const coo_t<index_t, value_t, rhs_space>& coo,
           vector_t<char, rhs_space>& r)
      : name(name), nnzs(coo.nnzs), values(coo.values) {
    ranks.resize(2);
    ranks[0] = r[0];
    ranks[1] = r[1];

    dims.resize(2);
    dims[0] = coo.rows;
    dims[1] = coo.cols;

    // Copy indices from COO
    indices.resize(2 * values.size());
    thrust::copy(coo.row_indices.begin(), coo.row_indices.end(),
                 indices.begin());
    thrust::copy(coo.col_indices.begin(), coo.col_indices.end(),
                 indices.begin() + values.size());
  }

  /**
   * @brief Construct a new tensor from a fully dense vector.
   *
   * @param vec vec_t<value_t, auto>
   */
  template <typename vec_t>
  tensor_t(std::string name, const vec_t& vec, char r)
      : name(name), nnzs(vec.size()), values(vec) {
    ranks.resize(1);
    ranks[0] = r;

    dims.resize(1);
    dims[0] = vec.size();

    indices.resize(vec.size());
    thrust::sequence(indices.begin(), indices.end(), 0);

    // thrust::copy(vec.begin(), vec.end(), values.begin());
  }

  /**
   * @brief Returns index offset for given rank
   */
  std::size_t get_index_offset(char rank) {
    for (std::size_t it = 0; it < ranks.size(); it++) {
      if (ranks[it] == rank) {
        return (it * values.size());
      }
    }

    throw error::exception_t(std::string("Tensor ") + name +
                             " get_index_offset(): rank '" + rank +
                             "' not found!\n");
  }

  /**
   * @brief Print out tensor information
   */
  void print() {
    std::cout << "Tensor " << name << std::endl;
    std::cout << "  Ranks: [";
    for (auto it = ranks.begin(); it != ranks.end(); ++it) {
      std::cout << *it;
      if (std::next(it) != ranks.end()) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
    std::cout << "  Dimensions: ";
    for (auto it = dims.begin(); it != dims.end(); ++it) {
      std::cout << *it;
      if (std::next(it) != dims.end()) {
        std::cout << " x ";
      }
    }
    std::cout << std::endl;
    std::cout << "  NNZs: " << nnzs << std::endl;
    std::cout << "  Coordinates: " << std::endl;
    for (auto rank : ranks) {
      std::cout << "    " << rank << ": [";
      std::size_t offset = get_index_offset(rank);
      for (std::size_t val_idx = 0; val_idx < values.size(); val_idx++) {
        std::cout << indices[offset + val_idx];
        if (val_idx != values.size() - 1)
          std::cout << ", ";
      }
      std::cout << "]" << std::endl;
    }
    std::cout << "  Values: [";
    for (auto it = values.begin(); it != values.end(); ++it) {
      std::cout << *it;
      if (std::next(it) != values.end()) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

};  // struct tensor_t

}  // namespace loops