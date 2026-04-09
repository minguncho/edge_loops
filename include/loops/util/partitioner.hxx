#pragma once

#include <loops/range.hxx>
#include <loops/error.hxx>

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
  __host__ WorkAtom(quarks_iterator_t quarks, quark_size_t num_quarks,
                    quark_size_t x_idx, quark_size_t y_idx)
    : quarks(quarks), num_quarks(num_quarks), x_idx(x_idx), y_idx(y_idx) {};

  __host__ void update_quarks(quarks_iterator_t new_quarks)  { quarks = new_quarks; }
  __host__ __device__ quark_size_t get_num_quarks() const { return num_quarks; }
  __host__ __device__ quark_size_t get_x_idx() const { return x_idx; }
  __host__ __device__ quark_size_t get_y_idx() const { return y_idx; }
  __host__ __device__ quarks_iterator_t begin() const { return quarks; };
  __host__ __device__ quarks_iterator_t end() const { return quarks + num_quarks; };

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

  __host__ void update_atoms(atoms_iterator_t new_atoms)  { atoms = new_atoms; }
  __host__ __device__ atom_size_t get_num_atoms() const { return num_atoms; }
  __host__ __device__ atoms_iterator_t begin() const { return atoms; };
  __host__ __device__ atoms_iterator_t end() const { return atoms + num_atoms; };

 private:
  atoms_iterator_t atoms;
  atom_size_t num_atoms;
};

/**
 * @brief Partitioner Class. 
 * Partitions a given matrix into work tiles, atoms, and tiles.
 */
template <typename index_t,
          typename value_t,
          typename quarks_t>
class Partitioner {
 public:

  __host__ Partitioner(coo_t<index_t, value_t, memory_space_t::host> &A)
    : A(A), is_partitioned(false) { 

    // Prepare for partition
    A.sort_by_row(); 
  };

  __host__ void partition_coordinate_space(std::size_t M0, std::size_t K0,
                                           std::size_t M1, std::size_t K1) {

    /**
     * Step 1: Partition quarks into atoms in coordinate space.
     */
    // Validate input parameters
    error::throw_if_exception((M0 <= 0 || K0 <= 0) || (M1 <= 0 || K1 <= 0), 
      "partition_coordinate_space(): Invalid size of atom or tile, cannot be <= 0!\n");

    error::throw_if_exception((M0 > A.rows || K0 > A.cols),
      std::string("partition_coordinate_space(): M0 and K0 exceeding limit!\n")
      + "  {M0: " + std::to_string(M0) + "} > {num_rows: " + std::to_string(A.rows) + "}\n"
      + "  {K0: " + std::to_string(K0) + "} > {num_cols: " + std::to_string(A.cols) + "}\n");

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
      size_t atom_idx = (A.row_indices[quark_idx] / M0) * num_atoms_y + (A.col_indices[quark_idx] / K0);
      atoms_nnz[atom_idx]++;
    }

    // Prefix sum for atoms_nnz to get starting position for each atom
    vector_t<std::size_t, memory_space_t::host> atoms_offsets(num_atoms + 1, 0);
    for (std::size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      atoms_offsets[atom_idx + 1] = atoms_offsets[atom_idx] + atoms_nnz[atom_idx];
    }

    // Fill in the address for atom's assigned quarks
    vector_t<std::size_t, memory_space_t::host> current_atom_pos = atoms_offsets;
    for (std::size_t quark_idx = 0; quark_idx < A.nnzs; quark_idx++) {
      size_t atom_idx = (A.row_indices[quark_idx] / M0) * num_atoms_y + (A.col_indices[quark_idx] / K0);
      size_t dest_idx = current_atom_pos[atom_idx]++;
      quarks[dest_idx] = quark_idx;
    }

    // Assign each atom with corresponding range of quarks
    for (std::size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      work_atoms[atom_idx] = WorkAtom<quarks_t>(&quarks[atoms_offsets[atom_idx]], atoms_nnz[atom_idx],
                                                (atom_idx / num_atoms_y), (atom_idx % num_atoms_y));
    }

    /**
     * Step 2: Partition atoms into tiles in coordinate space.
     */
    // Validate input parameters
    error::throw_if_exception((M1 > num_atoms_x || K1 > num_atoms_y),
      std::string("partition_coordinate_space(): M1 and K1 exceeding limit!\n")
      + "  {M1: " + std::to_string(M1) + "} > {num_atoms_x: " + std::to_string(num_atoms_x) + "}\n"
      + "  {K1: " + std::to_string(K1) + "} > {num_atoms_y: " + std::to_string(num_atoms_y) + "}\n");

    num_tiles_x = (num_atoms_x + M1 - 1) / M1;
    num_tiles_y = (num_atoms_y + K1 - 1) / K1;
    num_tiles = num_tiles_x * num_tiles_y;

    // Reset work tiles
    work_tiles.clear();
    work_tiles.resize(num_tiles);

    // Get number of atoms per tiles
    vector_t<std::size_t, memory_space_t::host> tiles_num_atoms(num_tiles, 0);
    for (std::size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      size_t tile_idx = (work_atoms[atom_idx].get_x_idx() / M1) * num_tiles_y + (work_atoms[atom_idx].get_y_idx() / K1);
      tiles_num_atoms[tile_idx]++;
    }

    // Prefix sum for tiles_num_atoms to get starting position for each tile
    vector_t<std::size_t, memory_space_t::host> tiles_offsets(num_tiles + 1, 0);
    for (std::size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      tiles_offsets[tile_idx + 1] = tiles_offsets[tile_idx] + tiles_num_atoms[tile_idx];
    }

    // Reorder the work atoms based on their tile idx
    vector_t<WorkAtom<quarks_t>, memory_space_t::host> sorted_work_atoms(num_atoms);
    vector_t<std::size_t, memory_space_t::host> current_tile_pos = tiles_offsets;
    for (std::size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      size_t tile_idx = (work_atoms[atom_idx].get_x_idx() / M1) * num_tiles_y + (work_atoms[atom_idx].get_y_idx() / K1);
      size_t dest_idx = current_tile_pos[tile_idx]++;
      sorted_work_atoms[dest_idx] = work_atoms[atom_idx];
    }

    // Replace the work atoms with the sorted 
    work_atoms = std::move(sorted_work_atoms);

    // Assign each tile with corresponding range of work atoms
    for (std::size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      work_tiles[tile_idx] = WorkTile<quarks_t>(&work_atoms[tiles_offsets[tile_idx]], tiles_num_atoms[tile_idx]);
    }

    is_partitioned = true;
  }

  __host__ void partition_position_space(std::size_t nnzs_per_atom, 
                                         std::size_t num_atoms_per_tile) {
    
    /**
     * Step 1: Partition quarks into atoms in position space.
     */
    // Validate input parameters
    error::throw_if_exception((nnzs_per_atom == 0), 
      "partition_position_space(): nnzs_per_atom cannot be zero!\n");

    error::throw_if_exception((nnzs_per_atom > A.nnzs), 
      "partition_position_space(): nnzs_per_atom cannot be greater than NNZ of A!\n");

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

    /**
     * Step 2: Partition atoms into tiles in position space.
     */

    num_tiles = (num_atoms + num_atoms_per_tile - 1) / num_atoms_per_tile;
    
    // Reset work tiles
    work_tiles.clear();
    work_tiles.resize(num_tiles);

    for (std::size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      std::size_t start_idx = tile_idx * num_atoms_per_tile;

      std::size_t real_num_atoms = std::min(num_atoms_per_tile, num_atoms - start_idx);
      work_tiles[tile_idx] = WorkTile<quarks_t>(&work_atoms[start_idx], real_num_atoms);
    }

    is_partitioned = true;
  }

  __host__ void prepare_gpu() {

    error::throw_if_exception(!is_partitioned, 
      "prepare_gpu(): Need to partition work atoms and tiles first.\n");

    // Prepare quarks for device
    d_quarks = quarks;

    // Prepare atoms for device, rewrite address for quarks
    vector_t<WorkAtom<quarks_t>, memory_space_t::host> temp_atoms = work_atoms;
    quarks_t* d_quarks_ptr = thrust::raw_pointer_cast(d_quarks.data());
    size_t quarks_offset = 0;
    for (size_t atom_idx = 0; atom_idx < num_atoms; atom_idx++) {
      temp_atoms[atom_idx].update_quarks(d_quarks_ptr + quarks_offset);
      quarks_offset += temp_atoms[atom_idx].get_num_quarks();
    }
    d_work_atoms = temp_atoms;

    // Prepare tiles for device, rewrite address for atoms
    vector_t<WorkTile<quarks_t>, memory_space_t::host> temp_tiles = work_tiles;
    WorkAtom<quarks_t>* d_atoms_ptr = thrust::raw_pointer_cast(d_work_atoms.data());
    size_t atoms_offset = 0;
    for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      temp_tiles[tile_idx].update_atoms(d_atoms_ptr + atoms_offset);
      atoms_offset += temp_tiles[tile_idx].get_num_atoms();
    }
    d_work_tiles = temp_tiles;
  }

  __host__ vector_t<WorkTile<quarks_t>>& get_work_tiles() { return d_work_tiles; }
  __host__ std::size_t get_num_tiles() { return num_tiles; }

 private:
  // Input matrix A in COO format
  coo_t<index_t, value_t, memory_space_t::host> &A;

  // Data structure and variables needed for atom partition
  vector_t<quarks_t, memory_space_t::host> quarks;
  vector_t<WorkAtom<quarks_t>, memory_space_t::host> work_atoms;
  std::size_t num_atoms_x, num_atoms_y, num_atoms;

  // Data structure and variables needed for tile partition
  vector_t<WorkTile<quarks_t>, memory_space_t::host> work_tiles;
  std::size_t num_tiles_x, num_tiles_y, num_tiles;

  bool is_partitioned; // Indicates if partition happened

  // Data structures needed for GPU
  vector_t<quarks_t> d_quarks;
  vector_t<WorkAtom<quarks_t>> d_work_atoms;
  vector_t<WorkTile<quarks_t>> d_work_tiles;
};

} // namespace loops