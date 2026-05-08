#pragma once

#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/error.hxx>
#include <loops/memory.hxx>

namespace loops {

using namespace memory;

/**
 * @brief Sample coordinate container for matrix (2-D tensor)
 *
 */  
template <typename index_t> 
struct matrix_coords {
  index_t r0;
  index_t r1;

  friend std::ostream& operator<<(std::ostream& os, const matrix_coords& c) {
    os << "(" << c.r0 << ", " << c.r1 << ")";
    return os;
  }
};

/**
 * @brief Sample coordinate container for vector (1-D tensor)
 *
 */  
template <typename index_t> 
struct vector_coords {
  index_t r0;

  friend std::ostream& operator<<(std::ostream& os, const vector_coords& v) {
    os << "(" << v.r0 << ")";
    return os;
  }
};

template <typename index_t,
          typename value_t,
          typename coord_t,
          memory_space_t space = memory_space_t::device>
struct tensor_t {
  std::string name;
  vector_t<char, space> ranks;
  vector_t<std::size_t, space> dims;
  std::size_t nnzs;

  vector_t<coord_t, space> indices;
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
   * @param indices List of coordinates of each rank to a value
   * @param values List of values
   */
  tensor_t(std::string name,
           vector_t<char, space>& ranks,
           vector_t<std::size_t, space>& dims,
           std::size_t nnzs,
           vector_t<coord_t, space>& indices,
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
   * @param rhs tensor_t<index_t, value_t, coord_t, rhs_space>
   */
  template <auto rhs_space>
  tensor_t(const tensor_t<index_t, value_t, coord_t, rhs_space>& rhs)
      : name(rhs.name),
        ranks(rhs.ranks),
        dims(rhs.dims),
        nnzs(rhs.nnzs),
        indices(rhs.indices),
        values(rhs.values) {}

  /**
   * @brief Construct a new tensor from coordinate format (COO) on the host.
   *
   * @param coo coo_t<index_t, value_t, auto>
   */
  template <auto rhs_space>
  tensor_t(std::string name,
           const coo_t<index_t, value_t, rhs_space>& coo,
           vector_t<char, memory_space_t::host>& r)
      : name(name), ranks(r), nnzs(coo.nnzs), values(coo.values) {

    std::vector<std::size_t> h_dims = {coo.rows, coo.cols};
    dims = vector_t<std::size_t, memory_space_t::host> (h_dims.begin(),
                                                        h_dims.end());

    // Copy indices from COO
    coo_t<index_t, value_t, memory_space_t::host> h_coo(coo); 
    vector_t<coord_t, memory_space_t::host> h_indices(nnzs);
    for (std::size_t nz = 0; nz < nnzs; nz++) {
      h_indices[nz].r0 = h_coo.row_indices[nz];
      h_indices[nz].r1 = h_coo.col_indices[nz];
    }
    indices = vector_t<coord_t, memory_space_t::host>(h_indices.begin(),
                                                     h_indices.end());
  }

  /**
   * @brief Construct a new tensor from a fully dense vector on the host.
   *
   * @param vec vec_t<value_t, auto>
   */
  template <typename vec_t>
  tensor_t(std::string name, const vec_t& vec, char r)
      : name(name), nnzs(vec.size()), values(vec) {
    
    std::vector<char> h_ranks = {r};
    ranks = vector_t<char, memory_space_t::host>(h_ranks.begin(),
                                                 h_ranks.end());
    
    std::vector<std::size_t> h_dims = {vec.size()};
    dims = vector_t<std::size_t, memory_space_t::host> (h_dims.begin(),
                                                        h_dims.end());

    vector_t<coord_t, memory_space_t::host> h_indices(nnzs);
    for (std::size_t nz = 0; nz < nnzs; nz++) {
      h_indices[nz].r0 = nz;
    }
    indices = vector_t<coord_t, memory_space_t::host>(h_indices.begin(),
                                                      h_indices.end());

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
    std::cout << "  Coordinates: ";
    for (std::size_t val_idx = 0; val_idx < values.size(); val_idx++) {
      std::cout << indices[val_idx];
      if (val_idx != values.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
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