#pragma once

#include <loops/range.hxx>
#include <loops/error.hxx>

/*
 * Note: Current version only supports work quark as a non-zero entry of A.
 *        Works with only SpMV at the moment.
 */

namespace loops {

/**
 * @brief Class for work atoms. Stores work quarks.
 */
template <typename quarks_type, typename quark_size_type = std::size_t>
class WorkAtom {
 public:
  using quarks_iterator_t = quarks_type*;
  using quark_size_t = quark_size_type;

  __host__ WorkAtom() : quarks(nullptr), num_quarks(0), x_idx(0), y_idx(0) {}
  __host__ WorkAtom(quarks_iterator_t quarks, quark_size_t num_quarks)
      : quarks(quarks), num_quarks(num_quarks), x_idx(0), y_idx(0) {};
  __host__ WorkAtom(quarks_iterator_t quarks,
                    quark_size_t num_quarks,
                    quark_size_t x_idx,
                    quark_size_t y_idx)
      : quarks(quarks), num_quarks(num_quarks), x_idx(x_idx), y_idx(y_idx) {};

  __host__ void update_quarks(quarks_iterator_t new_quarks) {
    quarks = new_quarks;
  }
  __host__ quarks_iterator_t get_quarks() const { return quarks; }
  __host__ __device__ quark_size_t get_num_quarks() const { return num_quarks; }
  __host__ __device__ quark_size_t get_x_idx() const { return x_idx; }
  __host__ __device__ quark_size_t get_y_idx() const { return y_idx; }
  __host__ __device__ quarks_iterator_t begin() const { return quarks; };
  __host__ __device__ quarks_iterator_t end() const {
    return quarks + num_quarks;
  };

 private:
  quarks_iterator_t quarks;
  quark_size_t num_quarks;
  quark_size_t x_idx, y_idx;
};

/**
 * @brief Class for work tiles. Stores work atoms.
 */
template <typename quarks_type, typename atom_size_type = std::size_t>
class WorkTile {
 public:
  using atoms_iterator_t = WorkAtom<quarks_type>*;
  using atom_size_t = atom_size_type;

  __host__ WorkTile() : atoms(nullptr), num_atoms(0) {}
  __host__ WorkTile(atoms_iterator_t atoms, atom_size_t num_atoms)
      : atoms(atoms), num_atoms(num_atoms) {};

  __host__ void update_atoms(atoms_iterator_t new_atoms) { atoms = new_atoms; }
  __host__ __device__ atom_size_t get_num_atoms() const { return num_atoms; }
  __host__ __device__ atoms_iterator_t begin() const { return atoms; };
  __host__ __device__ atoms_iterator_t end() const {
    return atoms + num_atoms;
  };

 private:
  atoms_iterator_t atoms;
  atom_size_t num_atoms;
};

/**
 * @brief Partitioner Class.
 * Partitions a given matrix into work tiles, atoms, and tiles.
 */
template <typename index_t, typename value_t, typename quarks_t>
class Partitioner {
 public:
  __host__ Partitioner(coo_t<index_t, value_t, memory_space_t::host>& A)
      : A(A),
        atoms_partitioned(false),
        tiles_partitioned(false),
        rank_flattened(false) {
    // Prepare for partition
    A.sort_by_row();
  };

  // Partition quarks into atoms in coordinate space (including zero entries)
  __host__ void partition_atoms_coordinate_space(std::size_t M0,
                                                 std::size_t K0) {
    // Validate input parameters
    error::throw_if_exception((M0 <= 0 || K0 <= 0),
                              "partition_atoms_coordinate_space(): Invalid "
                              "size of atom, cannot be <= 0!\n");

    error::throw_if_exception(
        (M0 > A.rows || K0 > A.cols),
        std::string("partition_atoms_coordinate_space(): M0 and K0 exceeding "
                    "limit!\n") +
            "  {M0: " + std::to_string(M0) + "} > {num_rows: " +
            std::to_string(A.rows) + "}\n" + "  {K0: " + std::to_string(K0) +
            "} > {num_cols: " + std::to_string(A.cols) + "}\n");

    // Reset quarks
    quarks.clear();
    quarks.resize(A.nnzs);

    num_atoms_x = (A.rows + M0 - 1) / M0;
    num_atoms_y = (A.cols + K0 - 1) / K0;
    num_atoms = num_atoms_x * num_atoms_y;

    // Reset work atoms
    work_atoms.clear();
    work_atoms.resize(num_atoms);

    // Get nnz for each atom
    vector_t<std::size_t, memory_space_t::host> atoms_nnz(num_atoms, 0);
    for (std::size_t quark_idx = 0; quark_idx < A.nnzs; quark_idx++) {
      size_t atom_idx = (A.row_indices[quark_idx] / M0) * num_atoms_y +
                        (A.col_indices[quark_idx] / K0);
      atoms_nnz[atom_idx]++;
    }

    // Prefix sum for atoms_nnz to get starting position for each atom
    vector_t<std::size_t, memory_space_t::host> atoms_offsets(num_atoms + 1, 0);
    for (std::size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      atoms_offsets[atom_idx + 1] =
          atoms_offsets[atom_idx] + atoms_nnz[atom_idx];
    }

    // Fill in the address for atom's assigned quarks
    vector_t<std::size_t, memory_space_t::host> current_atom_pos =
        atoms_offsets;
    for (std::size_t quark_idx = 0; quark_idx < A.nnzs; quark_idx++) {
      size_t atom_idx = (A.row_indices[quark_idx] / M0) * num_atoms_y +
                        (A.col_indices[quark_idx] / K0);
      size_t dest_idx = current_atom_pos[atom_idx]++;
      quarks[dest_idx] = quark_idx;
    }

    // Assign each atom with corresponding range of quarks
    for (std::size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      work_atoms[atom_idx] = WorkAtom<quarks_t>(
          &quarks[atoms_offsets[atom_idx]], atoms_nnz[atom_idx],
          (atom_idx / num_atoms_y), (atom_idx % num_atoms_y));
    }

    atoms_partitioned = true;
  }

  // Partition atoms into tiles in coordinate space (including zero entries)
  __host__ void partition_tiles_coordinate_space(std::size_t M1,
                                                 std::size_t K1) {
    // Validate input parameters
    error::throw_if_exception(!atoms_partitioned,
                              "partition_tiles_coordinate_space(): Need to "
                              "partition work atoms first.\n");

    error::throw_if_exception((M1 <= 0 || K1 <= 0),
                              "partition_tiles_coordinate_space(): Invalid "
                              "size of tile, cannot be <= 0!\n");

    error::throw_if_exception(
        (M1 > num_atoms_x || K1 > num_atoms_y),
        std::string("partition_tiles_coordinate_space(): M1 and K1 exceeding "
                    "limit!\n") +
            "  {M1: " + std::to_string(M1) +
            "} > {num_atoms_x: " + std::to_string(num_atoms_x) + "}\n" +
            "  {K1: " + std::to_string(K1) +
            "} > {num_atoms_y: " + std::to_string(num_atoms_y) + "}\n");

    num_tiles_x = (num_atoms_x + M1 - 1) / M1;
    num_tiles_y = (num_atoms_y + K1 - 1) / K1;
    num_tiles = num_tiles_x * num_tiles_y;

    // Reset work tiles
    work_tiles.clear();
    work_tiles.resize(num_tiles);

    // Get number of atoms per tiles
    vector_t<std::size_t, memory_space_t::host> tiles_num_atoms(num_tiles, 0);
    for (std::size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      size_t tile_idx = (work_atoms[atom_idx].get_x_idx() / M1) * num_tiles_y +
                        (work_atoms[atom_idx].get_y_idx() / K1);
      tiles_num_atoms[tile_idx]++;
    }

    // Prefix sum for tiles_num_atoms to get starting position for each tile
    vector_t<std::size_t, memory_space_t::host> tiles_offsets(num_tiles + 1, 0);
    for (std::size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      tiles_offsets[tile_idx + 1] =
          tiles_offsets[tile_idx] + tiles_num_atoms[tile_idx];
    }

    // Reorder the work atoms based on their tile idx
    vector_t<WorkAtom<quarks_t>, memory_space_t::host> sorted_work_atoms(
        num_atoms);
    vector_t<std::size_t, memory_space_t::host> current_tile_pos =
        tiles_offsets;
    for (std::size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      size_t tile_idx = (work_atoms[atom_idx].get_x_idx() / M1) * num_tiles_y +
                        (work_atoms[atom_idx].get_y_idx() / K1);
      size_t dest_idx = current_tile_pos[tile_idx]++;
      sorted_work_atoms[dest_idx] = work_atoms[atom_idx];
    }

    // Replace the work atoms with the sorted
    work_atoms = std::move(sorted_work_atoms);

    // Assign each tile with corresponding range of work atoms
    for (std::size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      work_tiles[tile_idx] = WorkTile<quarks_t>(
          &work_atoms[tiles_offsets[tile_idx]], tiles_num_atoms[tile_idx]);
    }

    tiles_partitioned = true;
  }

  // Partition quarks into atoms in position space (only including nonzero
  // entries)
  __host__ void partition_atoms_position_space(std::size_t M0, std::size_t K0) {
    // Validate input parameters
    error::throw_if_exception((M0 <= 0 || K0 <= 0),
                              "partition_atoms_position_space(): Invalid size "
                              "of atom, cannot be <= 0!\n");

    // Reset quarksW
    quarks.clear();
    quarks.resize(A.nnzs);

    // Reset work atoms
    num_atoms_x = 0;
    num_atoms_y = 0;
    num_atoms = 0;
    work_atoms.clear();

    // Prepare CSR format of A
    using offset_t = index_t;
    csr_t<offset_t, index_t, value_t, memory_space_t::host> A_csr(A);
    vector_t<offset_t, memory_space_t::host> active_rows;
    std::size_t global_quark_ptr = 0;

    // Build range of rows for given M0
    for (std::size_t row = 0; row < A_csr.rows; row++) {
      if (A_csr.offsets[row + 1] > A_csr.offsets[row]) {  // Row is not empty
        active_rows.push_back(row);
      }
    }

    for (std::size_t b_start = 0; b_start < active_rows.size(); b_start += M0) {
      std::size_t b_end = std::min(b_start + M0, active_rows.size());

      vector_t<std::size_t, memory_space_t::host> row_local_offsets(
          b_end - b_start, 0);
      bool block_has_remaining_nnz = true;
      std::size_t atom_in_block_idx = 0;

      while (block_has_remaining_nnz) {
        block_has_remaining_nnz = false;
        vector_t<std::size_t, memory_space_t::host> current_atom_indices;

        // Try to collect K0 non-zeros for the current atom
        for (std::size_t i = 0; i < (b_end - b_start); i++) {
          std::size_t actual_row = active_rows[b_start + i];
          std::size_t row_start_nnz = A_csr.offsets[actual_row];
          std::size_t row_end_nnz = A_csr.offsets[actual_row + 1];
          std::size_t total_in_row = row_end_nnz - row_start_nnz;

          std::size_t taken = 0;
          while (taken < K0 && row_local_offsets[i] < total_in_row) {
            current_atom_indices.push_back(row_start_nnz +
                                           row_local_offsets[i]);
            row_local_offsets[i]++;
            taken++;
          }

          // Check if this row still has more for the next pass
          if (row_local_offsets[i] < total_in_row) {
            block_has_remaining_nnz = true;
          }
        }

        if (!current_atom_indices.empty()) {
          // Commit indices to the global quarks array
          std::size_t atom_start_ptr = global_quark_ptr;
          for (auto idx : current_atom_indices) {
            quarks[global_quark_ptr++] = idx;
          }

          work_atoms.push_back(WorkAtom<quarks_t>(
              &quarks[atom_start_ptr], current_atom_indices.size(),
              b_start / M0, atom_in_block_idx++));

          num_atoms_x = std::max(num_atoms_x, (b_start / M0) + 1);
          num_atoms_y = std::max(num_atoms_y, atom_in_block_idx);
        }
      }
    }

    num_atoms = work_atoms.size();
    atoms_partitioned = true;
  }

  // Partition atoms into tiles in position space (only including nonzero
  // entries)
  __host__ void partition_tiles_position_space(std::size_t M1, std::size_t K1) {
    // Validate input parameters
    error::throw_if_exception(!atoms_partitioned,
                              "partition_tiles_position_space(): Need to "
                              "partition work atoms first.\n");

    error::throw_if_exception((M1 <= 0 || K1 <= 0),
                              "partition_tiles_position_space(): Invalid size "
                              "of tile, cannot be <= 0!\n");

    // Reset work tiles
    num_tiles_x = 0;
    num_tiles_y = 0;
    num_tiles = 0;
    work_tiles.clear();

    vector_t<vector_t<std::size_t, memory_space_t::host>, memory_space_t::host>
        atom_grid(num_atoms_x,
                  vector_t<int, memory_space_t::host>(num_atoms_y, -1));
    vector_t<WorkAtom<quarks_t>, memory_space_t::host> tile_atoms;
    for (std::size_t atom_id = 0; atom_id < num_atoms; atom_id++) {
      atom_grid[work_atoms[atom_id].get_x_idx()]
               [work_atoms[atom_id].get_y_idx()] = atom_id;
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
          work_tiles.push_back(WorkTile<quarks_t>(&tile_atoms[tile_start_idx],
                                                  num_atoms_in_tile));

          num_tiles_x = std::max(num_tiles_x, (i / M1) + 1);
          num_tiles_y = std::max(num_tiles_y, (j / K1) + 1);
        }
      }
    }

    // Replace the work atoms with new order
    work_atoms = std::move(tile_atoms);

    num_tiles = work_tiles.size();
    tiles_partitioned = true;
  }

  // Partition quarks into atoms in position space with flatten rank
  __host__ void partition_atoms_position_space_flatten(
      std::size_t nnzs_per_atom) {
    // Validate input parameters
    error::throw_if_exception((nnzs_per_atom == 0),
                              "partition_atoms_position_space_flatten(): "
                              "nnzs_per_atom cannot be zero!\n");

    error::throw_if_exception(
        (nnzs_per_atom > A.nnzs),
        "partition_atoms_position_space_flatten(): nnzs_per_atom cannot be "
        "greater than NNZ of A!\n");

    // Reset quarks
    quarks.clear();
    quarks.resize(A.nnzs);

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
  }

  // Partition atoms into tiles in position space with flatten rank
  __host__ void partition_tiles_position_space_flatten(
      std::size_t num_atoms_per_tile) {
    error::throw_if_exception(!atoms_partitioned,
                              "partition_tiles_position_space_flatten(): Need "
                              "to partition work atoms first.\n");

    error::throw_if_exception(!rank_flattened,
                              "partition_tiles_position_space_flatten(): Need "
                              "to partition work atoms using flatten method\n");

    num_tiles = (num_atoms + num_atoms_per_tile - 1) / num_atoms_per_tile;

    // Reset work tiles
    work_tiles.clear();
    work_tiles.resize(num_tiles);

    for (std::size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      std::size_t start_idx = tile_idx * num_atoms_per_tile;

      std::size_t real_num_atoms =
          std::min(num_atoms_per_tile, num_atoms - start_idx);
      work_tiles[tile_idx] =
          WorkTile<quarks_t>(&work_atoms[start_idx], real_num_atoms);
    }

    tiles_partitioned = true;
  }

  __host__ void prepare_gpu() {
    error::throw_if_exception(
        (!atoms_partitioned || !tiles_partitioned),
        "prepare_gpu(): Need to partition work atoms and tiles first.\n");

    // Prepare quarks for device
    d_quarks = quarks;

    // Prepare atoms for device, rewrite address for quarks
    vector_t<WorkAtom<quarks_t>, memory_space_t::host> temp_atoms = work_atoms;
    quarks_t* d_quarks_ptr = thrust::raw_pointer_cast(d_quarks.data());
    for (size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      temp_atoms[atom_idx].update_quarks(
          d_quarks_ptr + (work_atoms[atom_idx].get_quarks() - &quarks[0]));
    }
    d_work_atoms = temp_atoms;

    // Prepare tiles for device, rewrite address for atoms
    vector_t<WorkTile<quarks_t>, memory_space_t::host> temp_tiles = work_tiles;
    WorkAtom<quarks_t>* d_atoms_ptr =
        thrust::raw_pointer_cast(d_work_atoms.data());
    size_t atoms_offset = 0;
    for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      temp_tiles[tile_idx].update_atoms(d_atoms_ptr + atoms_offset);
      atoms_offset += temp_tiles[tile_idx].get_num_atoms();
    }
    d_work_tiles = temp_tiles;
  }

  __host__ vector_t<WorkTile<quarks_t>>& get_work_tiles() {
    return d_work_tiles;
  }
  __host__ std::size_t get_num_atoms() { return num_atoms; }
  __host__ std::size_t get_num_atoms_x() { return num_atoms_x; }
  __host__ std::size_t get_num_atoms_y() { return num_atoms_y; }
  __host__ std::size_t get_num_tiles() { return num_tiles; }
  __host__ std::size_t get_num_tiles_x() { return num_tiles_x; }
  __host__ std::size_t get_num_tiles_y() { return num_tiles_y; }

 private:
  // Input matrix A in COO format
  coo_t<index_t, value_t, memory_space_t::host>& A;

  // Data structure and variables needed for atom partition
  vector_t<quarks_t, memory_space_t::host> quarks;
  vector_t<WorkAtom<quarks_t>, memory_space_t::host> work_atoms;
  std::size_t num_atoms_x, num_atoms_y, num_atoms;

  // Data structure and variables needed for tile partition
  vector_t<WorkTile<quarks_t>, memory_space_t::host> work_tiles;
  std::size_t num_tiles_x, num_tiles_y, num_tiles;

  // Flags for partitioned status
  bool atoms_partitioned;
  bool tiles_partitioned;
  bool rank_flattened;

  // Data structures needed for GPU
  vector_t<quarks_t> d_quarks;
  vector_t<WorkAtom<quarks_t>> d_work_atoms;
  vector_t<WorkTile<quarks_t>> d_work_tiles;
};

}  // namespace loops