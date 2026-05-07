#pragma once

#include <loops/range.hxx>
#include <loops/error.hxx>
#include <set>

/*
 * Note: Current version only supports the following Einsum expressions:
 *       - Z[m] = A[m, k] * B[k] (SpMV)
 *       - Z[m,n] = A[m, k] * B[k, n] (SpMM, SpGEMM)
 */

namespace loops {

/**
 * @brief Class for work quarks.
 *  Stores a single point in the iteration space.
 */
/*template <typename index_t>
class WorkQuark {
 public:

  __host__ WorkQuark(index_t m, index_t k, index_t n)
    : m(m), k(k), n(n), atom_idx(0) {}

 private:

  vector_t<vector_t<index_t, space>> indices; // Need map? on storing
coordinates size_t atom_idx;
}*/

/**
 * @brief Class for work atoms. Stores assigned work quarks.
 */
/*template <typename index_t>
class WorkAtom {
 public:
  index_t m;
  index_t k;
  index_t n;
  size_t tile_idx;

  __host__ WorkAtom() : quarks(nullptr), num_quarks(0), m(0), k(0), n(0),
tile_idx(0) {}
  __host__ WorkAtom(WorkQuark<index_t>* quarks, std::size_t num_quarks,
                    index_t m, index_t k, index_t n)
    : quarks(quarks), num_quarks(num_quarks), m(m), k(k), n(n), tile_idx(0) {};

  __host__ void update_quarks(WorkQuark<index_t>* new_quarks)  { quarks =
new_quarks; }
  __host__ WorkQuark<index_t>* get_quarks() const { return quarks; }
  __host__ __device__ std::size_t get_num_quarks() const { return num_quarks; }
  __host__ __device__ WorkQuark<index_t>* begin() const { return quarks; };
  __host__ __device__ WorkQuark<index_t>* end() const { return quarks +
num_quarks; };

 private:
  WorkQuark<index_t>* quarks;
  std::size_t num_quarks;
};*/

/**
 * @brief Class for work tiles. Stores assigned work atoms.
 */
/*template <typename index_t>
class WorkTile {
 public:

  __host__ WorkTile() : atoms(nullptr), num_atoms(0) {}
  __host__ WorkTile(WorkAtom<index_t>* atoms, std::size_t num_atoms)
    : atoms(atoms), num_atoms(num_atoms) {};

  __host__ void update_atoms(WorkAtom<index_t>* new_atoms)  { atoms = new_atoms;
}
  __host__ __device__ std::size_t get_num_atoms() const { return num_atoms; }
  __host__ __device__ WorkAtom<index_t>* begin() const { return atoms; };
  __host__ __device__ WorkAtom<index_t>* end() const { return atoms + num_atoms;
};

 private:
  WorkAtom<index_t>* atoms;
  std::size_t num_atoms;
};*/

/**
 * @brief Partitioner Class.
 * Partitions a given matrix into work tiles, atoms, and tiles.
 */
template <typename index_t, typename value_t>
class Partitioner {
 public:
  using h_tensor_t = tensor_t<index_t, value_t, memory_space_t::host>;
  using d_tensor_t = tensor_t<index_t, value_t, memory_space_t::device>;

  __host__ Partitioner(
      vector_t<h_tensor_t, memory_space_t::host>& input_tensors,
      h_tensor_t& Z)  // Z is output tensor
      : input_tensors(input_tensors), Z(Z), rank_flattened(false) {
    // Collect ranks
    for (auto tensor : input_tensors) {
      for (auto rank : tensor.ranks) {
        all_ranks.insert(rank);
      }
    }

    for (auto rank : all_ranks) {
      printf("Rank: %s\n");
    }
  };

  // Perform an expansion on work quarks
  __host__ void expand_quarks() {
    // Reset quarks
    /*work_quarks.clear();

    for (std::size_t A_nz = 0; A_nz < A.nnzs; A_nz++) {
      index_t A_row = A.row_indices[A_nz]; // m
      index_t A_col = A.col_indices[A_nz]; // k

      for (std::size_t B_nz = 0; B_nz < B.nnzs; B_nz++) {
        index_t B_row = B.row_indices[B_nz]; // k
        index_t B_col = B.col_indices[B_nz]; // n

        if (A_col == B_row) {
          work_quarks.push_back(WorkAtom<index_t>(A_row, A_col, B_col));
        }
      }
    }*/
  }

  // Partition quarks into atoms in coordinate space (including zero entries)
  /*__host__ void partition_atoms_coordinate_space(std::size_t M0,
                                                 std::size_t K0,
                                                 std::size_t N0) {

    // Validate input parameters
    error::throw_if_exception(((M0 <= 0 || K0 <= 0) || N0 <= 0),
      "partition_atoms_coordinate_space(): Invalid partition size, cannot be <=
  0!\n");

    error::throw_if_exception(((M0 > A.rows || K0 > A.cols) || N0 > B.cols),
      std::string("partition_atoms_coordinate_space(): Invalid partition size,
  exceeding limit!\n")
      + "  {M0: " + std::to_string(M0) + "} > {A_rows: " +
  std::to_string(A.rows) + "}\n"
      + "  {K0: " + std::to_string(K0) + "} > {A_cols: " +
  std::to_string(A.cols) + "}\n"
      + "  {N0: " + std::to_string(N0) + "} > {B_cols: " +
  std::to_string(B.cols) + "}\n");

    error::throw_if_exception(work_quarks.size() == 0,
      "partition_atoms_coordinate_space(): Empty work quarks!");

    num_atoms_m = (A.rows + M0 - 1) / M0;
    num_atoms_k = (A.cols + K0 - 1) / K0;
    num_atoms_n = (B.cols + N0 - 1) / N0;
    num_atoms = num_atoms_m * num_atoms_k * num_atoms_n;

    // Reset work atoms
    work_atoms.clear();
    work_atoms.resize(num_atoms);

    // Get number of quarks for each atom
    vector_t<std::size_t, memory_space_t::host> atoms_num_quarks(num_atoms, 0);
    for (std::size_t quark_idx = 0; quark_idx < quarks.size(); quark_idx++) {
      std::size_t atom_idx_m = work_quarks[quark_idx].m / M0;
      std::size_t atom_idx_k = work_quarks[quark_idx].k / K0;
      std::size_t atom_idx_n = work_quarks[quark_idx].n / N0;

      std::size_t atom_idx = (atom_idx_m * num_atoms_k * num_atoms_n) +
                        (atom_idx_k * num_atoms_n) + atom_idx_n;

      atoms_num_quarks[atom_idx]++;
      work_quarks[quark_idx].atom_idx = atom_idx;
    }

    // Prefix sum for atoms_num_quarks to get starting position of work quark
  for each atom vector_t<std::size_t, memory_space_t::host> offsets(num_atoms +
  1, 0); for (std::size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      offsets[atom_idx + 1] = offsets[atom_idx] + atoms_num_quarks[atom_idx];
    }

    // Reorder work quarks based on their atom idx
    vector_t<WorkAtom<index_t>, memory_space_t::host>
  reordered_work_quarks(quarks.size()); vector_t<std::size_t,
  memory_space_t::host> current_atom_pos = offsets; for (std::size_t quark_idx =
  0; quark_idx < quarks.size(); quark_idx++) { std::size_t atom_idx =
  quarks[quark_idx].atom_idx; std::size_t dest_idx =
  current_atom_pos[atom_idx]++; reordered_work_quarks[dest_idx] =
  quarks[quark_idx];
    }

    // Replace the work quarks with the reordered list
    work_quarks = std::move(reordered_work_quarks);

    // Assign each atom with corresponding range of quarks
    for (std::size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      std::size_t atom_idx_m = atom_idx / (num_atoms_k * num_atoms_n);
      std::size_t atom_idx_k = (atom_idx / num_atoms_n) % num_atoms_k;
      std::size_t atom_idx_n = atom_idx % num_atoms_n;

      work_atoms[atom_idx] = WorkAtom<index_t>(&work_quarks[offsets[atom_idx]],
                                                atoms_num_quarks[atom_idx],
                                                atom_idx_m, atom_idx_k,
  atom_idx_n);
    }
  }*/

  // Partition atoms into tiles in coordinate space (including zero entries)
  /*__host__ void partition_tiles_coordinate_space(std::size_t M1,
                                                 std::size_t K1,
                                                 std::size_t N1) {

    // Validate input parameters
    error::throw_if_exception(((M1 <= 0 || K1 <= 0) || N1 <= 0),
      "partition_tiles_coordinate_space(): Invalid partition size, cannot be <=
  0!\n");

    error::throw_if_exception(((M1 > num_atoms_m || K1 > num_atoms_k) || N1 >
  num_atoms_n), std::string("partition_tiles_coordinate_space(): Invalid
  partition size, exceeding limit!\n")
      + "  {M1: " + std::to_string(M1) + "} > {num_atoms_m: " +
  std::to_string(num_atoms_m) + "}\n"
      + "  {K1: " + std::to_string(K1) + "} > {num_atoms_k: " +
  std::to_string(num_atoms_k) + "}\n"
      + "  {N1: " + std::to_string(N1) + "} > {num_atoms_n: " +
  std::to_string(num_atoms_n) + "}\n");

    error::throw_if_exception(work_atoms.size() == 0,
      "partition_tiles_coordinate_space(): Empty work atoms!\n");

    num_tiles_m = (num_atoms_m + M1 - 1) / M1;
    num_tiles_k = (num_atoms_k + K1 - 1) / K1;
    num_tiles_n = (num_atoms_n + N1 - 1) / N1;
    num_tiles = num_tiles_m * num_tiles_k * num_tiles_n;

    // Reset work tiles
    work_tiles.clear();
    work_tiles.resize(num_tiles);

    // Get number of atoms for each tile
    vector_t<std::size_t, memory_space_t::host> tiles_num_atoms(num_tiles, 0);
    for (std::size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      std::size_t tile_idx_m = work_atoms[atom_idx].m / M1;
      std::size_t tile_idx_k = work_atoms[atom_idx].k / K1;
      std::size_t tile_idx_n = work_atoms[atom_idx].n / N1;

      std::size_t tile_idx = (tile_idx_m * num_tiles_k * num_tiles_n) +
                        (tile_idx_k * num_tiles_n) + tile_idx_n;

      tiles_num_atoms[tile_idx]++;
      work_atoms[atom_idx].tile_idx = tile_idx;
    }

    // Prefix sum for tiles_num_atoms to get starting position of work atom for
  each tile vector_t<std::size_t, memory_space_t::host> offsets(num_tiles + 1,
  0); for (std::size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      offsets[tile_idx + 1] = offsets[tile_idx] + tiles_num_atoms[tile_idx];
    }

    // Reorder the work atoms based on their tile idx
    vector_t<WorkAtom<index_t>, memory_space_t::host>
  reordered_work_atoms(num_atoms); vector_t<std::size_t, memory_space_t::host>
  current_tile_pos = offsets; for (std::size_t atom_idx = 0; atom_idx <
  num_atoms; atom_idx++) { size_t tile_idx = work_atoms[atom_idx].tile_idx;
      size_t dest_idx = current_tile_pos[tile_idx]++;
      reordered_work_atoms[dest_idx] = work_atoms[atom_idx];
    }

    // Replace the work atoms with the reordered list
    work_atoms = std::move(reordered_work_atoms);

    // Assign each tile with corresponding range of work atoms
    for (std::size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      std::size_t tile_idx_m = tile_idx / (num_tiles_k * num_tiles_n);
      std::size_t tile_idx_k = (tile_idx / num_tiles_n) % num_tiles_k;
      std::size_t tile_idx_n = tile_idx % num_tiles_n;

      work_tiles[tile_idx] =
  WorkTile<index_t>(&work_atoms[tiles_offsets[tile_idx]],
                                                tiles_num_atoms[tile_idx]);
    }
  }*/

  // Partition quarks into atoms in position space (only including nonzero
  // entries)
  /*__host__ void partition_atoms_position_space(std::size_t M0,
                                               std::size_t K0,
                                               std::size_t N0) {
    // Validate input parameters
    error::throw_if_exception(((M0 <= 0 || K0 <= 0) || N0 <= 0),
      "partition_atoms_position_space(): Invalid partition size, cannot be <=
  0!\n");

    error::throw_if_exception(work_quarks.size() == 0,
      "partition_atoms_position_space(): Empty work quarks!");

    // Reset work atoms
    work_atoms.clear();

    // Prepare CSR format of A
    using offset_t = index_t;
    csr_t<offset_t, index_t, value_t, memory_space_t::host> A_csr(A);
    vector_t<offset_t, memory_space_t::host> active_rows;
    std::size_t global_quark_ptr = 0;

    // Build range of rows for given M0
    for (std::size_t row = 0; row < A_csr.rows; row++) {
      if (A_csr.offsets[row + 1] > A_csr.offsets[row]) { // Row is not empty
        active_rows.push_back(row);
      }
    }

    for (std::size_t b_start = 0; b_start < active_rows.size(); b_start += M0) {
      std::size_t b_end = std::min(b_start + M0, active_rows.size());

      vector_t<std::size_t, memory_space_t::host> row_local_offsets(b_end -
  b_start, 0); bool block_has_remaining_quarks = true; std::size_t atom_k_idx =
  0;

      while (block_has_remaining_quarks) {
        block_has_remaining_quarks = false;

        for (std::size_t atom_n_idx = 0; atom_n_idx < B.cols; atom_n_idx++) { //
  N_COUNT_LIMIT depends on your data depth vector_t<std::size_t,
  memory_space_t::host> current_atom_indices;

          for (std::size_t i = 0; i < (b_end - b_start); i++) {
            std::size_t actual_row = active_rows[b_start + i];
            std::size_t row_start_nnz = A_csr.offsets[actual_row];
            std::size_t total_in_row = A_csr.offsets[actual_row + 1] -
  row_start_nnz;

            std::size_t taken = 0;
            // We take K0 * N0 elements for a 3D "volume" atom
            while (taken < (K0 * N0) && row_local_offsets[i] < total_in_row) {
              current_atom_indices.push_back(row_start_nnz +
  row_local_offsets[i]); row_local_offsets[i]++; taken++;
            }

            if (row_local_offsets[i] < total_in_row) {
              block_has_remaining_quarks = true;
            }
          }

          if (!current_atom_indices.empty()) {
            std::size_t atom_start_ptr = global_quark_ptr;
            for (auto idx : current_atom_indices) {
              quarks[global_quark_ptr++] = idx;
            }

            // Push WorkAtom with 3D coordinate metadata
            work_atoms.push_back(WorkAtom<quarks_t>(
              &quarks[atom_start_ptr],
              current_atom_indices.size(),
              b_start / M0, // m_idx
              atom_k_idx,   // k_idx
              atom_n_idx    // n_idx
            ));

            // Track global dimensions for the atom grid
            num_atoms_m = std::max(num_atoms_m, (b_start / M0) + 1);
            num_atoms_k = std::max(num_atoms_k, atom_k_idx + 1);
            num_atoms_n = std::max(num_atoms_n, atom_n_idx + 1);
          }
          else {
            break; // No more data in this N-slice
          }
        }
      atom_k_idx++;
    }

    num_atoms = work_atoms.size();
  }*/

  // Partition atoms into tiles in position space (only including nonzero
  // entries)
  /*__host__ void partition_tiles_position_space(std::size_t M1, std::size_t K1)
  {
    // Validate input parameters
    error::throw_if_exception(!atoms_partitioned,
      "partition_tiles_position_space(): Need to partition work atoms
  first.\n");

    error::throw_if_exception((M1 <= 0 || K1 <= 0),
      "partition_tiles_position_space(): Invalid size of tile, cannot be <=
  0!\n");

    // Reset work tiles
    num_tiles_x = 0;
    num_tiles_y = 0;
    num_tiles = 0;
    work_tiles.clear();

    vector_t<vector_t<std::size_t, memory_space_t::host>, memory_space_t::host>
  atom_grid(num_atoms_x, vector_t<int, memory_space_t::host>(num_atoms_y, -1));
    vector_t<WorkAtom<quarks_t>, memory_space_t::host> tile_atoms;
    for (std::size_t atom_id = 0; atom_id < num_atoms; atom_id++) {
      atom_grid[work_atoms[atom_id].get_x_idx()][work_atoms[atom_id].get_y_idx()]
  = atom_id;
    }

    for (std::size_t i = 0; i < num_atoms_x; i += M1) {
      for (std::size_t j = 0; j < num_atoms_y; j += K1) {

        std::size_t tile_start_idx = tile_atoms.size();
        std::size_t num_atoms_in_tile = 0;

        // Collect all valid atoms within this M1 x K1 block
        for (std::size_t r = i; r < std::min(i + M1, num_atoms_x); r++) {
          for (std::size_t c = j; c < std::min(j + K1, num_atoms_y); c++) {
            int atom_idx = atom_grid[r][c];
            if (atom_idx != -1) {
              tile_atoms.push_back(work_atoms[atom_idx]);
              num_atoms_in_tile++;
            }
          }
        }

        if (num_atoms_in_tile > 0) {
          work_tiles.push_back(WorkTile<quarks_t>(
            &tile_atoms[tile_start_idx],
            num_atoms_in_tile
          ));

          num_tiles_x = std::max(num_tiles_x, (i / M1) + 1);
          num_tiles_y = std::max(num_tiles_y, (j / K1) + 1);
        }

      }
    }

    // Replace the work atoms with new order
    work_atoms = std::move(tile_atoms);

    num_tiles = work_tiles.size();
    tiles_partitioned = true;
  }*/

  // Partition quarks into atoms in position space with flatten rank
  /*__host__ void partition_atoms_position_space_flatten(std::size_t
  nnzs_per_atom) {

    // Validate input parameters
    error::throw_if_exception((nnzs_per_atom == 0),
      "partition_atoms_position_space_flatten(): nnzs_per_atom cannot be
  zero!\n");

    error::throw_if_exception((nnzs_per_atom > A.nnzs),
      "partition_atoms_position_space_flatten(): nnzs_per_atom cannot be greater
  than NNZ of A!\n");

    num_atoms = (A.nnzs + nnzs_per_atom - 1) / nnzs_per_atom;

    // Reset work atoms
    work_atoms.clear();
    work_atoms.resize(num_atoms);

    for (std::size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      std::size_t start_idx = atom_idx * nnzs_per_atom;

      std::size_t real_nnzs = std::min(nnzs_per_atom, A.nnzs - start_idx);
      work_atoms[atom_idx] = WorkAtom<quarks_t>(&quarks[start_idx], real_nnzs);

      std::size_t end_idx = start_idx + real_nnzs;
      for (; start_idx < end_idx; start_idx++) {
        quarks[start_idx] = start_idx;
      }
    }

    atoms_partitioned = true;
    rank_flattened = true;
  }*/

  // Partition atoms into tiles in position space with flatten rank
  /*__host__ void partition_tiles_position_space_flatten(std::size_t
  num_atoms_per_tile) {

    error::throw_if_exception(!atoms_partitioned,
      "partition_tiles_position_space_flatten(): Need to partition work atoms
  first.\n");

    error::throw_if_exception(!rank_flattened,
      "partition_tiles_position_space_flatten(): Need to partition work atoms
  using flatten method\n");

    num_tiles = (num_atoms + num_atoms_per_tile - 1) / num_atoms_per_tile;

    // Reset work tiles
    work_tiles.clear();
    work_tiles.resize(num_tiles);

    for (std::size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      std::size_t start_idx = tile_idx * num_atoms_per_tile;

      std::size_t real_num_atoms = std::min(num_atoms_per_tile, num_atoms -
  start_idx); work_tiles[tile_idx] = WorkTile<quarks_t>(&work_atoms[start_idx],
  real_num_atoms);
    }

    tiles_partitioned = true;
  }*/

  /*__host__ void prepare_gpu() {

    error::throw_if_exception((!atoms_partitioned || !tiles_partitioned),
      "prepare_gpu(): Need to partition work atoms and tiles first.\n");

    // Prepare quarks for device
    d_quarks = quarks;

    // Prepare atoms for device, rewrite address for quarks
    vector_t<WorkAtom<quarks_t>, memory_space_t::host> temp_atoms = work_atoms;
    quarks_t* d_quarks_ptr = thrust::raw_pointer_cast(d_quarks.data());
    for (size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      temp_atoms[atom_idx].update_quarks(d_quarks_ptr +
  (work_atoms[atom_idx].get_quarks() - &quarks[0]));
    }
    d_work_atoms = temp_atoms;

    // Prepare tiles for device, rewrite address for atoms
    vector_t<WorkTile<quarks_t>, memory_space_t::host> temp_tiles = work_tiles;
    WorkAtom<quarks_t>* d_atoms_ptr =
  thrust::raw_pointer_cast(d_work_atoms.data()); size_t atoms_offset = 0; for
  (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      temp_tiles[tile_idx].update_atoms(d_atoms_ptr + atoms_offset);
      atoms_offset += temp_tiles[tile_idx].get_num_atoms();
    }
    d_work_tiles = temp_tiles;
  }*/

  /*__host__ vector_t<WorkTile<index_t>>& get_work_tiles() { return
  d_work_tiles; }
  __host__ std::size_t get_num_atoms() { return num_atoms; }
  __host__ std::size_t get_num_atoms_m() { return num_atoms_m; }
  __host__ std::size_t get_num_atoms_k() { return num_atoms_k; }
  __host__ std::size_t get_num_atoms_n() { return num_atoms_n; }
  __host__ std::size_t get_num_tiles() { return num_tiles; }
  __host__ std::size_t get_num_tiles_m() { return num_tiles_m; }
  __host__ std::size_t get_num_tiles_k() { return num_tiles_k; }
  __host__ std::size_t get_num_tiles_n() { return num_tiles_n; }*/

 private:
  // Tensors
  vector_t<h_tensor_t, memory_space_t::host> input_tensors;
  h_tensor_t Z;
  std::set<std::string> all_ranks;

  // Data structure and variables needed for quark to atom partition
  /*vector_t<index_t, memory_space_t::host> work_quarks;
  vector_t<WorkAtom<index_t>, memory_space_t::host> work_atoms;
  std::size_t num_atoms_m, num_atoms_k, num_atoms_n, num_atoms;

  // Data structure and variables needed for atom to tile partition
  vector_t<WorkTile<index_t>, memory_space_t::host> work_tiles;
  std::size_t num_tiles_m, num_tiles_k, num_tiles_n, num_tiles;*/

  // Flags for partitioned status
  bool rank_flattened;

  // Data structures needed for GPU
  /*vector_t<quarks_t> d_quarks;
  vector_t<WorkAtom<quarks_t>> d_work_atoms;
  vector_t<WorkTile<quarks_t>> d_work_tiles;*/
};

}  // namespace loops