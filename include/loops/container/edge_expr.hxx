#pragma once

#include <loops/error.hxx>
#include <unordered_set>
#include <map>

namespace loops {

using namespace memory;

/**
 * @brief Edge Expression container.
 *
 */
template <typename index_t,
          typename value_t,
          typename A_coord_t,
          typename B_coord_t,
          typename Z_coord_t,
          typename expr_coord_t,
          memory_space_t space = memory_space_t::device>
struct edge_expr_t {
  // Z = A * B
  tensor_t<index_t, value_t, A_coord_t, space> A;
  tensor_t<index_t, value_t, B_coord_t, space> B;
  tensor_t<index_t, value_t, Z_coord_t, space> Z;

  vector_t<char, space> ranks;
  vector_t<std::size_t, space> dims;

  vector_t<expr_coord_t, space>
      indices;  /// List of coordinates of each rank in the iteration space

  edge_expr_t() : A(), B(), Z(), ranks(), dims(), indices() {}

  template <auto rhs_space>
  edge_expr_t(const edge_expr_t<index_t,
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
        indices(rhs.indices) {}

  /**
   * @brief Constructor of edge_expr_t
   * @note Current version only supports 2 input tensors
   */
  edge_expr_t(tensor_t<index_t, value_t, A_coord_t, space>& A,
              tensor_t<index_t, value_t, B_coord_t, space>& B,
              tensor_t<index_t, value_t, Z_coord_t, space>& Z,
              vector_t<char, space>& ranks,
              vector_t<std::size_t, space>& dims)
      : A(A), B(B), Z(Z), ranks(ranks), dims(dims) {
    error::throw_if_exception((ranks.size() != dims.size()),
                              "edge_expr_t(): ranks.size() != dims.size()!\n");
  }

  /**
   * @brief Find rank idx of given rank
   * @note Can be used on device side, but preferably on host side
   *
   * @param rank Desired rank to rank idx
   */
  std::size_t get_expr_rank_idx(char rank) {
    vector_t<char, memory_space_t::host> h_ranks = ranks;
    for (std::size_t i = 0; i < h_ranks.size(); ++i) {
      if (h_ranks[i] == rank) {
        return i;
      }
    }

    throw error::exception_t(std::string("get_rank_idx(): rank '") + rank +
                             "' not found!\n");
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

    std::vector<expr_coord_t> h_indices;
    for (auto s_rank : shared_ranks) {
      // Go through every coordinates of A
      for (auto& A_coord : h_A.indices) {
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

          h_indices.push_back(expr_coord);
        }
      }
    }
    indices = vector_t<expr_coord_t, memory_space_t::host>(h_indices.begin(),
                                                           h_indices.end());

    for (auto i : indices)
      std::cout << i << std::endl;
  }

  /**
   * @brief Perform partition on coordinate space on the host
   * @note Current version only supports SpMV, SpMM, SpGEMM
   *
   * @param
   */
  void partition_coordinate_space(std::vector<std::size_t> part_sizes) {
    error::throw_if_exception(
        (part_sizes.size() != dims.size()),
        "partition_coordinate_space(): Invalid size of part_size()! Not equal "
        "to number of unique ranks\n");

    vector_t<expr_coord_t, memory_space_t::host> h_indices = indices;

    // Compute number of tiles per rank
    std::vector<std::size_t> grid_dims(dims.size());
    for (std::size_t i = 0; i < dims.size(); i++) {
      grid_dims[i] = (dims[i] + part_sizes[i] - 1) / part_sizes[i];
    }

    std::map<std::size_t, std::vector<expr_coord_t>> tiles;

    for (auto& coord : h_indices) {
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

    for (std::size_t i = 0; i < tiles.size(); i++) {
      std::cout << "Tile #" << i << "\n  ";
      for (auto& coord : tiles[i]) {
        std::cout << coord << " ";
      }
      std::cout << "\n";
    }
  }

};  // struct edge_expr

}  // namespace loops