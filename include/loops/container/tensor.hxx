/**
 * @file tensor.hxx
 * @author
 * @brief
 * @version
 * @date
 *
 * @copyright
 *
 */

#pragma once

#include <loops/container/formats.hxx>
#include <loops/container/matrix.cuh>
#include <loops/container/vector.hxx>
#include <loops/error.hxx>
#include <loops/memory.hxx>

namespace loops {

using namespace memory;

/**
 * @brief Coordinate container for N-dimension tensor
 *
 * @tparam index_t Type of index.
 * @tparam N       Value of N.
 */
template <typename index_t, std::size_t N>
struct coords {
  index_t r[N];
  __host__ __device__ static constexpr std::size_t get_N() { return N; }
  __host__ __device__ index_t& operator[](std::size_t i) { return r[i]; }
  __host__ __device__ bool operator==(const coords& c) {
    for (std::size_t i = 0; i < N; ++i) {
      if (r[i] != c.r[i]) {
        return false;
      }
    }
    return true;
  }
  __host__ friend std::ostream& operator<<(std::ostream& os, const coords& c) {
    os << "(";
    for (std::size_t i = 0; i < N; ++i) {
      os << c.r[i];
      if (i != (N - 1)) {
        os << ", ";
      }
    }
    os << ")";
    return os;
  }
};

/**
 * @brief Tensor container
 *
 * @tparam index_t Type of index.
 * @tparam value_t Type of non-zero value.
 * @tparam N       Value of N.
 */
template <typename index_t,
          typename value_t,
          typename coord_t,
          memory_space_t space = memory_space_t::device>
struct tensor_t {
  std::string name;                   /// Name of tensor.
  vector_t<char, space> ranks;        /// List of rank labels.
  vector_t<std::size_t, space> dims;  /// List of size of each dimension.
  std::size_t nnzs;                   /// Number of non-zero elements.

  vector_t<coord_t, space>
      coords;  /// List of coordinates of each rank to a value.
  vector_t<value_t, space> values;  /// List of values.

  /**
   * @brief Construct a new tensor object with everything initialized to zero.
   *
   */
  tensor_t() : name(""), ranks(), dims(), nnzs(0), coords(), values() {}

  /**
   * @brief Construct a new tensor object.
   *
   * @param name   Name of tensor.
   * @param ranks  List of rank labels.
   * @param dims   List of size of each dimension.
   * @param nnzs   Number of non-zero elements.
   * @param coords List of coordinates of each rank to a value.
   * @param values List of values.
   */
  tensor_t(std::string name,
           vector_t<char, space>& ranks,
           vector_t<std::size_t, space>& dims,
           std::size_t nnzs,
           vector_t<coord_t, space>& coords,
           vector_t<value_t, space>& values)
      : name(name),
        ranks(ranks),
        dims(dims),
        nnzs(nnzs),
        coords(coords),
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
        coords(rhs.coords),
        values(rhs.values) {}

  /**
   * @brief Construct a new tensor from coordinate format (COO) on the host.
   *
   * @param name  Name of tensor.
   * @param coo   coo_t<index_t, value_t, auto>
   * @param ranks List of rank labels.
   */
  template <auto rhs_space>
  tensor_t(std::string name,
           const coo_t<index_t, value_t, rhs_space>& coo,
           vector_t<char, space>& ranks)
      : name(name), ranks(ranks), nnzs(coo.nnzs), values(coo.values) {
    error::throw_if_exception(
        (coord_t::get_N() != 2),
        "tensor_t(): Construction with COO, coord_t's N is not 2!");
    std::vector<std::size_t> h_dims = {coo.rows, coo.cols};
    dims = vector_t<std::size_t, space>(h_dims.begin(), h_dims.end());

    // Copy coords from COO
    coo_t<index_t, value_t, memory_space_t::host> h_coo(coo);
    h_coo.sort_by_row();
    std::vector<coord_t> h_coords(nnzs);
    for (std::size_t nz = 0; nz < nnzs; nz++) {
      h_coords[nz].r[0] = h_coo.row_indices[nz];
      h_coords[nz].r[1] = h_coo.col_indices[nz];
    }
    coords = vector_t<coord_t, space>(h_coords.begin(), h_coords.end());
  }

  /**
   * @brief Construct a new tensor from a fully dense vector on the host.
   *
   * @param name Name of tensor.
   * @param vec  vec_t<value_t, auto>
   * @param rank Rank label.
   */
  template <typename vec_t>
  tensor_t(std::string name, const vec_t& vec, char rank)
      : name(name), nnzs(vec.size()), values(vec) {
    error::throw_if_exception(
        (coord_t::get_N() != 1),
        "tensor_t(): Construction with vector, coord_t's N is not 1!");
    std::vector<char> h_ranks = {rank};
    ranks = vector_t<char, space>(h_ranks.begin(), h_ranks.end());

    std::vector<std::size_t> h_dims = {vec.size()};
    dims = vector_t<std::size_t, space>(h_dims.begin(), h_dims.end());

    std::vector<coord_t> h_coords(nnzs);
    for (std::size_t nz = 0; nz < nnzs; nz++) {
      h_coords[nz].r[0] = nz;
    }
    coords = vector_t<coord_t, space>(h_coords.begin(), h_coords.end());
  }

  /**
   * @brief Construct a new tensor from a fully dense matrix on the host.
   *
   * @param name  Name of tensor.
   * @param mat   matrix_t<value_t, auto>
   * @param ranks List of rank labels.
   */
  template <auto rhs_space>
  tensor_t(std::string name,
           const matrix_t<value_t, rhs_space>& mat,
           vector_t<char, space>& ranks)
      : name(name), ranks(ranks), nnzs(mat.rows * mat.cols), values(mat.m_data) {
    error::throw_if_exception(
        (coord_t::get_N() != 2),
        "tensor_t(): Construction with matrix, coord_t's N is not 2!");

    std::vector<std::size_t> h_dims = {mat.rows, mat.cols};
    dims = vector_t<std::size_t, space>(h_dims.begin(), h_dims.end());

    // Copy coords from mat
    std::vector<coord_t> h_coords(nnzs);
    for (std::size_t r = 0; r < mat.rows; r++) {
      for (std::size_t c = 0; c < mat.cols; c++) {
        h_coords[(r * mat.cols) + c].r[0] = r;
        h_coords[(r * mat.cols) + c].r[1] = c;
      }
    }
    coords = vector_t<coord_t, space>(h_coords.begin(), h_coords.end());
  }

  /**
   * @brief Find rank idx of given rank
   *
   * @param rank Desired rank to find the rank idx.
   */
  __host__ __device__ std::size_t get_rank_idx(char rank) {
    for (std::size_t i = 0; i < ranks.size(); ++i) {
      if (ranks[i] == rank) {
        return i;
      }
    }

#ifdef __CUDA_ARCH__  // Device side error handling
    printf("get_rank_idx(): rank '%c' not found!\n", rank);
    return 0;
#else  // Host side error handling
    throw error::exception_t(std::string("get_rank_idx(): rank '") + rank +
                             "' not found!\n");
#endif
  }

  /**
   * @brief Get dimension of given rank
   *
   * @param rank Desired rank to get dimension.
   */
  __host__ __device__ std::size_t get_dim(char rank) {
    std::size_t rank_idx = get_rank_idx(rank);
    return dims[rank_idx];
  }

  /**
   * @brief Find coordinates that equal to the value of shared ranks
   * @note To be used on the host side
   *
   * @param shared_rank Rank that is shared between other tensor.
   * @param val         Value of coordinate of the shared rank.
   */
  __host__ vector_t<coord_t, memory_space_t::host> find_shared_coords(
      char shared_rank,
      value_t val) {
    std::size_t rank_idx = get_rank_idx(shared_rank);

    std::vector<coord_t> shared_coords;
    vector_t<coord_t, memory_space_t::host> h_coords = coords;

    for (auto& coord : h_coords) {
      if (coord[rank_idx] == val) {
        shared_coords.push_back(coord);
      }
    }

    return shared_coords;
  }

  /**
   * @brief Print out tensor information on the host side
   *
   */
  __host__ void print() {
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
      std::cout << coords[val_idx];
      if (val_idx != values.size() - 1)
        std::cout << ", ";
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