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
#include <algorithm>
#include <unordered_set>
#include <map>
#include <unordered_map>

namespace loops {

using namespace memory;

/**
 * @brief Edge container.
 *
 * @tparam index_t       Type of index.
 * @tparam value_t       Type of non-zero value.
 * @tparam space         Memory space (host/device) for storage.
 * @tparam expr_coord_t  Type of coordinate container of iteration
 *                       space of edge expression.
 * @tparam Z_coord_t     Type of coordinate container of tensor Z.
 * @tparam input_coord_t Type of coordinate container of input tensors.
 */
template <typename index_t,
          typename value_t,
          memory_space_t space,
          typename expr_coord_t,
          typename Z_coord_t,
          typename... input_coord_t>
struct edge_t {
  vector_t<char, space> ranks;  /// List of ranks of the entire expression
  vector_t<std::size_t, space>
      dims;  /// List of dimensions corresponding to each rank

  tensor_t<index_t, value_t, Z_coord_t, space> Z;  /// Output Tensor Z
  std::tuple<tensor_t<index_t, value_t, input_coord_t, space>...>
      input_tensors;  /// Input Tensors

  // Work Atoms
  vector_t<expr_coord_t, space>
      coords;  /// List of coordinates of each rank in the iteration space
  std::size_t num_atoms;

  // Work Tiles
  vector_t<std::size_t, space>
      tile_offsets;  /// Tile offsets collected from partitioning
  std::size_t num_tiles;

  /**
   * @brief Validate input tensors on host.
   *
   */
  void validate_input_tensors() {
    std::tuple<
        tensor_t<index_t, value_t, input_coord_t, memory_space_t::host>...>
        h_input_tensors = input_tensors;
    std::unordered_set<std::string> tensor_names;

    std::apply(
        [&](auto&... tensor) {
          (
              [&](auto& t) {
                if (tensor_names.find(t.name) != tensor_names.end())
                  throw error::exception_t(
                      std::string("validate_input_tensors(): tensor name '") +
                      t.name + "' already exists!\n");
                else
                  tensor_names.insert(t.name);
              }(tensor),
              ...);
        },
        h_input_tensors);

    error::throw_if_exception(
        tensor_names.empty(),
        "validate_input_tensors(): Empty set of tensor names!\n");
  }

  /**
   * @brief Default Constructor
   *
   */
  edge_t()
      : ranks(),
        dims(),
        Z(),
        input_tensors(),
        coords(),
        num_atoms(0),
        tile_offsets(),
        num_tiles(0) {}

  /**
   * @brief Construct a new edge_t from another edge_t on host/device.
   *
   * @param rhs edge_t<index_t, value_t, rhs_space, expr_coord_t,
   *                     Z_coord_t, input_coord_t...>
   */
  template <auto rhs_space>
  edge_t(const edge_t<index_t,
                      value_t,
                      rhs_space,
                      expr_coord_t,
                      Z_coord_t,
                      input_coord_t...>& rhs)
      : ranks(rhs.ranks),
        dims(rhs.dims),
        Z(rhs.Z),
        input_tensors(rhs.input_tensors),
        coords(rhs.coords),
        num_atoms(rhs.num_atoms),
        tile_offsets(rhs.tile_offsets),
        num_tiles(rhs.num_tiles) {
    validate_input_tensors();
  }

  /**
   * @brief Constructor of edge_t
   *
   * @param ranks         List of ranks of the entire expression
   * @param dims          List of dimensions corresponding to each rank
   * @param Z             Output tensor Z
   * @param input_tensors Input tensors
   *
   */
  edge_t(vector_t<char, space>& ranks,
         vector_t<std::size_t, space>& dims,
         tensor_t<index_t, value_t, Z_coord_t, space>& Z,
         tensor_t<index_t, value_t, input_coord_t, space>&... input_tensors)
      : ranks(ranks),
        dims(dims),
        Z(Z),
        input_tensors(input_tensors...),
        coords(),
        num_atoms(0),
        tile_offsets(),
        num_tiles(0) {
    error::throw_if_exception((ranks.size() != dims.size()),
                              "edge_expr_t(): ranks.size() != dims.size()!\n");
    error::throw_if_exception((ranks.size() != expr_coord_t::get_N()),
                              "edge_expr_t(): ranks.size() != expr_coord_t::get_N()!\n");
    validate_input_tensors();
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
   * @brief Collects all valid unique coordinate values
   * for a target rank, filtered by the dimensions that have
   * already been bound in the global expression space.
   * @note To be used on the host side
   *
   * @param tensor             Current tensor object.
   * @param target_rank        Current targeted rank.
   * @param current_expr_coord The global workspace tracking
   *                           currently bound coordinate values.
   */
  template <typename cur_tensor_t>
  __host__ std::unordered_set<index_t> get_tensor_active_coordinates(
      cur_tensor_t& tensor,
      char target_rank,
      expr_coord_t& current_expr_coord) {
    std::unordered_set<index_t> valid_values;
    std::size_t target_local_idx = 0;
    bool has_target_rank = false;

    // Identify which of this tensor's local ranks have already been bound
    // globally.
    struct bound_mapping_t {
      std::size_t local_idx;
      std::size_t global_idx;
    };
    std::vector<bound_mapping_t> active_filters;

    for (std::size_t i = 0; i < tensor.ranks.size(); ++i) {
      char local_rank = tensor.ranks[i];

      if (local_rank == target_rank) {
        target_local_idx = i;
        has_target_rank = true;
        continue;
      }

      // Check if this other rank has already been bound in the global
      // expression engine. We know its rank index in the global expression
      // space via get_expr_rank_idx.
      std::size_t global_idx = get_expr_rank_idx(local_rank);

      // Safety check: Does the backtracking engine consider this rank "already
      // processed"? Since we process ranks sequentially (0 to N-1), if a rank's
      // global index mapping is less than the global index mapping of our
      // target_rank, it means it has already been bound.
      if (global_idx < get_expr_rank_idx(target_rank)) {
        active_filters.push_back({i, global_idx});
      }
    }

    if (!has_target_rank) {
      return valid_values;
    }

    valid_values = tensor.get_active_coordinates(
        target_local_idx, active_filters, current_expr_coord);

    return valid_values;
  }

  /**
   * @brief Build a list of intersected interation points on the host.
   *
   * Go through each input tensors' coordinates and
   * collect intersected iteration points. These are
   * coordinates that all input tensors have a nz value.
   */
  void expand_intersected_iteration_points() {
    std::tuple<
        tensor_t<index_t, value_t, input_coord_t, memory_space_t::host>...>
        h_input_tensors = input_tensors;
    vector_t<char, memory_space_t::host> h_ranks = ranks;

    // Build Constraint Map
    // For every global rank, find which tensors contain it
    std::unordered_map<char, std::vector<std::string>> constraints;
    for (char rank : h_ranks) {
      std::apply(
          [&](auto&... tensor) {
            (
                [&](auto& t) {
                  auto it = std::find(t.ranks.begin(), t.ranks.end(), rank);
                  if (it != t.ranks.end()) {
                    constraints[rank].push_back(t.name);
                  }
                }(tensor),
                ...);
          },
          h_input_tensors);
    }

    // A working buffer representing the current multidimensional coordinate we
    // are building
    expr_coord_t current_expr_coord;
    std::vector<expr_coord_t> h_coords;

    // Recursive Backtracking Engine
    // rank_idx: the index of the global rank we are currently resolving
    auto IntersectAndExpand = [&](auto& self, std::size_t rank_idx) -> void {
      // Base Case: All ranks resolved! We found a valid iteration point.
      if (rank_idx == ranks.size()) {
        h_coords.push_back(current_expr_coord);
        return;
      }

      char current_rank = h_ranks[rank_idx];
      auto& active_tensor_names = constraints[current_rank];

      // Find the intersection of valid coordinate values for 'current_rank'
      // across all tensors that share this rank.
      std::unordered_set<index_t> valid_values;
      bool first_tensor = true;

      for (const auto& ref_name : active_tensor_names) {
        std::unordered_set<index_t> local_values;

        std::apply(
            [&](auto&... tensor) {
              (
                  [&](auto& t) {
                    if (ref_name == t.name) {
                      local_values = get_tensor_active_coordinates(
                          t, current_rank, current_expr_coord);
                    }
                  }(tensor),
                  ...);
            },
            h_input_tensors);

        if (first_tensor) {
          valid_values = std::move(local_values);
          first_tensor = false;
        } else {
          // Intersect with existing valid values
          for (auto it = valid_values.begin(); it != valid_values.end();) {
            if (local_values.find(*it) == local_values.end()) {
              it = valid_values.erase(
                  it);  // Not present in this tensor, discard
            } else {
              ++it;
            }
          }
        }
        if (valid_values.empty())
          break;  // Early termination if intersection is empty
      }

      // Bind each valid intersecting coordinate value and recurse to the next
      // dimension
      for (index_t value : valid_values) {
        std::size_t global_idx = get_expr_rank_idx(current_rank);
        current_expr_coord[global_idx] = value;

        self(self, rank_idx + 1);  // Recurse to next rank
      }
    };

    // Kick off the multi-dimensional join starting at global rank index 0
    IntersectAndExpand(IntersectAndExpand, 0);

    // Save back to class storage
    coords = vector_t<expr_coord_t, space>(h_coords.begin(), h_coords.end());
    num_atoms = h_coords.size();
  }

  /**
   * @brief Build a list of union interation points on the host.
   *
   * Go through each input tensors' coordinates and
   * collect union iteration points. These are
   * coordinates that any input tensors have a nz value.
   */
  void expand_union_iteration_points() {
    std::tuple<
        tensor_t<index_t, value_t, input_coord_t, memory_space_t::host>...>
        h_input_tensors = input_tensors;
    std::vector<expr_coord_t> h_coords;
    vector_t<char, memory_space_t::host> h_ranks = ranks;

    std::apply(
        [&](auto&... tensor) {
          (
              [&](auto& t) {
                // Check tensor ranks == expression ranks
                for (std::size_t rank_id = 0; rank_id < h_ranks.size(); rank_id++) {
                  error::throw_if_exception((h_ranks[rank_id] != t.ranks[rank_id]),
                    "expand_union_iteration_points(): Tensor rank mismatch!\n");
                }
                for (auto& coord : t.coords) {
                  if (std::find(h_coords.begin(), h_coords.end(), coord) == h_coords.end())
                    h_coords.push_back(coord);
                }
              }(tensor),
              ...);
        },
        h_input_tensors);
    
    coords = vector_t<expr_coord_t, space>(h_coords.begin(), h_coords.end());
  }

  /**
   * @brief Perform partition on coordinate space on the host
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
    num_tiles = h_tile_offsets.size() - 1;
  }

  /**
   * @brief Print out Edge container information on the host side
   *
   */
  void print() {
    vector_t<char, memory_space_t::host> h_ranks = ranks;
    vector_t<std::size_t, memory_space_t::host> h_dims = dims;
    vector_t<expr_coord_t, memory_space_t::host> h_coords = coords;
    vector_t<std::size_t, memory_space_t::host> h_tile_offsets = tile_offsets;
    std::cout << "------Input Tensors------" << std::endl;
    std::apply([](auto&... tensor) { (..., (tensor.print())); }, input_tensors);
    std::cout << "------Output Tensor------" << std::endl;
    Z.print();
    std::cout << "-----EDGE Expression------" << std::endl;
    std::cout << "Expression ranks & dims:" << std::endl;
    for (std::size_t rank_id = 0; rank_id < h_ranks.size(); rank_id++) {
      std::cout << "  " << h_ranks[rank_id] << ": " << h_dims[rank_id]
                << std::endl;
    }
    std::cout << "Number of work atoms: " << num_atoms << std::endl;
    std::cout << "Number of work tiles: " << num_tiles << std::endl;
    for (std::size_t tile_id = 0; tile_id < num_tiles; tile_id++) {
      std::size_t start = h_tile_offsets[tile_id];
      std::size_t end = h_tile_offsets[tile_id + 1];

      std::cout << "  Tile #" << tile_id << std::endl;
      std::cout << "    Coordinates: ";
      for (; start < end; start++) {
        std::cout << h_coords[start] << " ";
      }
      std::cout << std::endl;
    }
  }

};  // struct edge_expr

}  // namespace loops