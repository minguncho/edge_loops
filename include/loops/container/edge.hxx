/**
 * @file edge.hxx
 * @author
 * @brief
 * @version
 * @date
 *
 * Edge container stores input and output tensors from the
 * edge expression. Note that the current version supports
 * a single edge expression, 2 input tensors (A, B) and
 * 1 output tensor (Z).
 *
 * Edge container is responsible for building a list of
 * iteration points by traversing through the coordinates
 * of A and B. It also supports performing a partition
 * in coordinate space.
 *
 * @copyright
 *
 */

#pragma once

#include <loops/error.hxx>
#include <unordered_set>
#include <map>

namespace loops {

using namespace memory;

/**
 * @brief Edge container.
 *
 * @tparam index_t      Type of index.
 * @tparam value_t      Type of non-zero value.
 * @tparam A_coord_t    Type of coordinate container of tensor A.
 * @tparam B_coord_t    Type of coordinate container of tensor B.
 * @tparam Z_coord_t    Type of coordinate container of tensor Z.
 * @tparam expr_coord_t Type of coordinate container of iteration
 *                      space of edge expression.
 * @tparam space        Memory space (host/device) for storage.
 */
template <typename index_t,
          typename value_t,
          typename A_coord_t,
          typename B_coord_t,
          typename Z_coord_t,
          typename expr_coord_t,
          memory_space_t space = memory_space_t::device>
struct edge_t {
  tensor_t<index_t, value_t, A_coord_t, space> A;
  tensor_t<index_t, value_t, B_coord_t, space> B;
  tensor_t<index_t, value_t, Z_coord_t, space> Z;

  vector_t<char, space> ranks;  /// List of ranks of the entire expression
  vector_t<std::size_t, space>
      dims;  /// List of dimensions corresponding to each rank

  vector_t<expr_coord_t, space>
      coords;  /// List of coordinates of each rank in the iteration space
  vector_t<std::size_t, space> tile_offsets;  /// Tile offsets collected from
                                              /// partition_coordinate_space()

  edge_t() : A(), B(), Z(), ranks(), dims(), coords(), tile_offsets() {}

  /**
   * @brief Construct a new edge_t from another edge_t on host/device.
   *
   * @param rhs tensor_t<index_t, value_t, A_coord_t, B_coord_t,
   *                     Z_coord_t, expr_coord_t, rhs_space>
   */
  template <auto rhs_space>
  edge_t(const edge_t<index_t,
                      value_t,
                      A_coord_t,
                      B_coord_t,
                      Z_coord_t,
                      expr_coord_t,
                      rhs_space>& rhs)
      : A(rhs.A),
        B(rhs.B),
        Z(rhs.Z),
        ranks(rhs.ranks),
        dims(rhs.dims),
        coords(rhs.coords),
        tile_offsets(rhs.tile_offsets) {}

  /**
   * @brief Constructor of edge_t
   * @note Current version only supports 2 input tensors
   *
   * @param A Input tensor A
   * @param B Input tensor B
   * @param Z Output tensor Z
   * @param ranks List of ranks of the entire expression
   * @param dims List of dimensions corresponding to each rank
   */
  edge_t(tensor_t<index_t, value_t, A_coord_t, space>& A,
         tensor_t<index_t, value_t, B_coord_t, space>& B,
         tensor_t<index_t, value_t, Z_coord_t, space>& Z,
         vector_t<char, space>& ranks,
         vector_t<std::size_t, space>& dims)
      : A(A), B(B), Z(Z), ranks(ranks), dims(dims), coords(), tile_offsets() {
    error::throw_if_exception((ranks.size() != dims.size()),
                              "edge_expr_t(): ranks.size() != dims.size()!\n");
  }

  /**
   * @brief Find rank idx of given rank
   *
   * @param rank Desired rank to find the rank idx
   */
  __host__ __device__ std::size_t get_expr_rank_idx(char rank) {
    for (std::size_t i = 0; i < ranks.size(); ++i) {
      if (ranks[i] == rank) {
        return i;
      }
    }

#ifdef __CUDA_ARCH__  // Device side error handling
    printf("get_expr_rank_idx(): rank '%c' not found!\n", rank);
    return 0;
#else  // Host side error handling
    throw error::exception_t(std::string("get_expr_rank_idx(): rank '") + rank +
                             "' not found!\n");
#endif
  }

  /**
   * @brief Build a list of interation points on the host.
   * @note Current version only supports SpMV, SpMM, SpGEMM
   *
   * Go through each input tensors' coordinates and
   * build iteration point using shared ranks of A and B.
   */
  void expand_iteration_points() {
    // Get tensors on the host side
    tensor_t<index_t, value_t, A_coord_t, memory_space_t::host> h_A = A;
    tensor_t<index_t, value_t, B_coord_t, memory_space_t::host> h_B = B;

    // Find shared ranks of A and B
    std::vector<char> shared_ranks;
    std::unordered_set<char> setA(h_A.ranks.begin(), h_A.ranks.end());
    for (char r : h_B.ranks) {
      if (setA.erase(r)) {  // erase returns 1 if element existed
        shared_ranks.push_back(r);
      }
    }

    error::throw_if_exception(
        (shared_ranks.size() == 0),
        "expand_iteration_points(): A and B share no shared ranks!\n");

    error::throw_if_exception((shared_ranks.size() != 1),
                              "expand_iteration_points(): Current version only "
                              "supports 1 shared rank!\n");

    std::vector<expr_coord_t> h_coords;
    for (auto s_rank : shared_ranks) {
      // Go through every coordinates of A
      for (auto& A_coord : h_A.coords) {
        // Shared rank idx on A
        std::size_t s_rank_idx = h_A.get_rank_idx(s_rank);

        // Using the value of shared rank of current coord of A,
        // collect coordinates of B that match the value of shared rank
        auto shared_coords =
            h_B.find_shared_coords(s_rank, A_coord[s_rank_idx]);

        for (auto& B_coord : shared_coords) {
          expr_coord_t expr_coord;

          // Insert A_coord
          for (auto A_rank : h_A.ranks) {
            std::size_t A_rank_idx = h_A.get_rank_idx(A_rank);
            std::size_t rank_idx = get_expr_rank_idx(A_rank);
            expr_coord[rank_idx] = A_coord[A_rank_idx];
          }

          // Insert B_coord
          for (auto B_rank : h_B.ranks) {
            std::size_t B_rank_idx = h_B.get_rank_idx(B_rank);
            std::size_t rank_idx = get_expr_rank_idx(B_rank);
            expr_coord[rank_idx] = B_coord[B_rank_idx];
          }

          h_coords.push_back(expr_coord);
        }
      }
    }
    coords = vector_t<expr_coord_t, memory_space_t::host>(h_coords.begin(),
                                                          h_coords.end());
  }

  /**
   * @brief Perform partition on coordinate space on the host
   * @note Current version only supports SpMV, SpMM, SpGEMM
   *
   * @param part_sizes List of partition size for each rank
   */
  void partition_coordinate_space(std::vector<std::size_t> part_sizes) {
    error::throw_if_exception(
        (part_sizes.size() != dims.size()),
        "partition_coordinate_space(): Invalid size of part_size()! Not equal "
        "to number of unique ranks\n");

    vector_t<expr_coord_t, memory_space_t::host> h_coords = coords;

    // Compute number of tiles per rank
    std::vector<std::size_t> grid_dims(dims.size());
    for (std::size_t i = 0; i < dims.size(); i++) {
      grid_dims[i] = (dims[i] + part_sizes[i] - 1) / part_sizes[i];
    }

    std::map<std::size_t, std::vector<expr_coord_t>> tiles;

    for (auto& coord : h_coords) {
      std::size_t flat_tile_id = 0;
      std::size_t multiplier = 1;

      // Flatten the Tile ID (Calculated in reverse for row-major order)
      for (int i = dims.size() - 1; i >= 0; --i) {
        std::size_t local_tile_coord =
            static_cast<std::size_t>(coord[i]) / part_sizes[i];
        flat_tile_id += local_tile_coord * multiplier;
        multiplier *= grid_dims[i];
      }
      tiles[flat_tile_id].push_back(coord);
    }

    std::vector<expr_coord_t> h_flattened_coords;
    h_flattened_coords.reserve(h_coords.size());
    std::vector<std::size_t> h_tile_offsets;
    h_tile_offsets.push_back(0);

    for (auto& tile : tiles) {
      h_flattened_coords.insert(h_flattened_coords.end(), tile.second.begin(),
                                tile.second.end());
      h_tile_offsets.push_back(h_flattened_coords.size());
    }

    coords = vector_t<expr_coord_t, memory_space_t::host>(
        h_flattened_coords.begin(), h_flattened_coords.end());
    tile_offsets = vector_t<std::size_t, memory_space_t::host>(
        h_tile_offsets.begin(), h_tile_offsets.end());
  }

};  // struct edge_expr

}  // namespace loops