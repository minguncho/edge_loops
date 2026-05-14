/**
 * @file layout_edge.hxx
 * @author
 * @brief
 * @version
 * @date
 *
 * @copyright
 *
 */

#pragma once

#include <cstddef>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace loops {
namespace layout {

/**
 * @brief Layout view for the Edge container.
 *
 * Current version only supports partition in coordinate space.
 * Tile correspond to partitioned "block" in coordinate space.
 *
 * @tparam tile_id_type Index type for tiles (e.g., row id).
 * @tparam atom_id_type Index type for atoms (e.g., flat nnz position).
 */

template <typename tile_id_type, typename atom_id_type>
struct edge {
  using tile_id_t = tile_id_type;
  using atom_id_t = atom_id_type;
  using tile_end_iterator_t = atom_id_t const*;

  atom_id_t const*
      offsets_;  /// length num_tiles + 1, monotonically non-decreasing.
  tile_id_t n_tiles_;
  atom_id_t n_atoms_;

  __host__ __device__ edge() : n_tiles_(0), n_atoms_(0) {}
  __host__ __device__ edge(atom_id_t const* offsets,
                           tile_id_t num_tiles,
                           atom_id_t num_atoms)
      : offsets_(offsets), n_tiles_(num_tiles), n_atoms_(num_atoms) {}

  __host__ __device__ tile_id_t num_tiles() const { return n_tiles_; }
  __host__ __device__ atom_id_t num_atoms() const { return n_atoms_; }

  __host__ __device__ atom_id_t tile_begin(tile_id_t t) const {
    return offsets_[t];
  }
  __host__ __device__ atom_id_t tile_end(tile_id_t t) const {
    return offsets_[t + 1];
  }
  __host__ __device__ atom_id_t tile_size(tile_id_t t) const {
    return offsets_[t + 1] - offsets_[t];
  }

  __host__ __device__ tile_end_iterator_t tile_end_iter() const {
    return offsets_ + 1;
  }
};

}  // namespace layout
}  // namespace loops

#include <loops/container/partitioning.hxx>