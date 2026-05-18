/**
 * Matrix multiplication with 3 tensors.
 * Testing the Edge container with
 * any number of input tensors.
 *
 * Z[m, v] = A[m, k] * B[k, n] * C[n, v]
 */

#include "helpers.hxx"
#include <iostream>
#include <loops/container/edge.hxx>
#include <loops/container/layout_edge.hxx>
#include <loops/container/tensor.hxx>
#include <loops/container/vector.hxx>
#include <loops/memory.hxx>
#include <loops/schedule.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/util/tracker.hxx>

using namespace loops;

template <typename setup_t,
          typename index_t,
          typename value_t,
          typename expr_coord_t,
          typename Z_coord_t,
          typename A_coord_t,
          typename B_coord_t,
          typename C_coord_t>
__global__ void __edge_thread_mapped(setup_t config,
                                     expr_coord_t* expr_coords,
                                     Z_coord_t* Z_coords,
                                     A_coord_t* A_coords,
                                     B_coord_t* B_coords,
                                     C_coord_t* C_coords,
                                     value_t* Z_values,
                                     value_t* A_values,
                                     value_t* B_values,
                                     value_t* C_values,
                                     std::size_t Z_nnzs,
                                     std::size_t A_nnzs,
                                     std::size_t B_nnzs,
                                     std::size_t C_nnzs,
                                     int* coords_tid) {
  for (auto tile_idx : config.tiles()) {
    for (auto atom : config.atoms(tile_idx)) {
      index_t m = expr_coords[atom][0];
      index_t k = expr_coords[atom][1];
      index_t n = expr_coords[atom][2];
      index_t v = expr_coords[atom][3];

      value_t A_val = 0;
      for (std::size_t nz = 0; nz < A_nnzs; nz++) {
        if (A_coords[nz] == A_coord_t{m, k}) {
          A_val = A_values[nz];
          break;
        }
      }

      value_t B_val = 0;
      for (std::size_t nz = 0; nz < B_nnzs; nz++) {
        if (B_coords[nz] == B_coord_t{k, n}) {
          B_val = B_values[nz];
          break;
        }
      }

      value_t C_val = 0;
      for (std::size_t nz = 0; nz < C_nnzs; nz++) {
        if (C_coords[nz] == C_coord_t{n, v}) {
          C_val = C_values[nz];
          break;
        }
      }

      for (std::size_t nz = 0; nz < Z_nnzs; nz++) {
        if (Z_coords[nz] == Z_coord_t{m, v}) {
          atomicAdd(&Z_values[nz], A_val * B_val * C_val);
          break;
        }
      }
      coords_tid[atom] = (blockIdx.x * blockDim.x) + threadIdx.x;
    }
  }
}

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using value_t = float;

  parameters_t parameters(argc, argv);

  // Check filename requirements
  if (parameters.filenames.size() != 3) {
    std::cout << "Invalid number of Matrix files, Correct number: 3"
              << std::endl;
    std::exit(0);
  }

  matrix_market_t<index_t, offset_t, value_t> mtx;
  coo_t<index_t, value_t> A_coo = mtx.load(parameters.filenames[0]);
  std::string A_name = mtx.dataset;
  coo_t<index_t, value_t> B_coo = mtx.load(parameters.filenames[1]);
  std::string B_name = mtx.dataset;
  coo_t<index_t, value_t> C_coo = mtx.load(parameters.filenames[2]);
  std::string C_name = mtx.dataset;

  // Check dimension requirement
  if (A_coo.cols != B_coo.rows) {
    std::cout << "Invalid dimension, K must be equal! " << A_coo.cols
              << " != " << B_coo.rows << std::endl;
    std::exit(0);
  }

  if (B_coo.cols != C_coo.rows) {
    std::cout << "Invalid dimension, N must be equal! " << B_coo.cols
              << " != " << C_coo.rows << std::endl;
    std::exit(0);
  }

  // Define dimensions
  std::size_t M = A_coo.rows;
  std::size_t K = A_coo.cols;
  std::size_t N = B_coo.cols;
  std::size_t V = C_coo.cols;

  using A_coord_t = coords<index_t, 2>;
  using B_coord_t = coords<index_t, 2>;
  using C_coord_t = coords<index_t, 2>;
  using Z_coord_t = coords<index_t, 2>;
  using expr_coord_t = coords<index_t, 4>;

  thrust::device_vector<char> A_ranks = {'M', 'K'};
  tensor_t<index_t, value_t, A_coord_t> A("A", A_coo, A_ranks);

  thrust::device_vector<char> B_ranks = {'K', 'N'};
  tensor_t<index_t, value_t, B_coord_t> B("B", B_coo, B_ranks);

  thrust::device_vector<char> C_ranks = {'N', 'V'};
  tensor_t<index_t, value_t, C_coord_t> C("C", C_coo, C_ranks);

  thrust::device_vector<char> Z_ranks = {'M', 'V'};
  matrix_t<value_t> Z_mat(M, V);
  tensor_t<index_t, value_t, Z_coord_t> Z("Z", Z_mat, Z_ranks);

  thrust::device_vector<char> expr_ranks = {'M', 'K', 'N', 'V'};
  thrust::device_vector<std::size_t> expr_dims = {M, K, N, V};

  using edge_expr_t =
      edge_t<index_t, value_t, memory_space_t::device, expr_coord_t, Z_coord_t,
             A_coord_t, B_coord_t, C_coord_t>;

  edge_expr_t edge_expr(expr_ranks, expr_dims, Z, A, B, C);
  edge_expr.expand_intersected_iteration_points();
  edge_expr.partition_coordinate_space({2, 2, 2, 2});

  using tile_id_t = std::size_t;
  using atom_id_t = std::size_t;
  using edge_layout_t = layout::edge<tile_id_t, atom_id_t>;

  using setup_t =
      schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1, tile_id_t,
                      atom_id_t, std::size_t, std::size_t, edge_layout_t>;

  edge_layout_t lay(edge_expr.tile_offsets.data().get(),
                    static_cast<tile_id_t>(edge_expr.num_tiles),
                    static_cast<atom_id_t>(edge_expr.num_atoms));
  setup_t config(lay);

  constexpr std::size_t block_size = 128;
  std::size_t grid_size = math::ceil_div(edge_expr.num_tiles, block_size);
  cudaStream_t stream = 0;

  tracker_t<atom_id_t> tracker(edge_expr.num_atoms, block_size * grid_size);

  util::timer_t timer;
  timer.start();

  launch::non_cooperative(
      stream,
      __edge_thread_mapped<setup_t, index_t, value_t, expr_coord_t, Z_coord_t,
                           A_coord_t, B_coord_t, C_coord_t>,
      grid_size, block_size, config, edge_expr.coords.data().get(),
      Z.coords.data().get(), A.coords.data().get(), B.coords.data().get(),
      C.coords.data().get(), Z.values.data().get(), A.values.data().get(),
      B.values.data().get(), C.values.data().get(), Z.nnzs, A.nnzs, B.nnzs,
      C.nnzs, tracker.coord_tid.data().get());
  cudaStreamSynchronize(stream);
  timer.stop();

  if (parameters.validate) {
    vector_t<expr_coord_t, memory_space_t::host> h_coords = edge_expr.coords;
    thrust::host_vector<char> h_Z_ranks = {'M', 'V'};
    tensor_t<index_t, value_t, A_coord_t, memory_space_t::host> h_A = A;
    tensor_t<index_t, value_t, B_coord_t, memory_space_t::host> h_B = B;
    tensor_t<index_t, value_t, B_coord_t, memory_space_t::host> h_C = C;
    tensor_t<index_t, value_t, Z_coord_t, memory_space_t::host> h_Z(
        "h_Z", Z_mat, h_Z_ranks);
    for (auto& coord : h_coords) {
      index_t m = coord[0];
      index_t k = coord[1];
      index_t n = coord[2];
      index_t v = coord[3];

      value_t A_val = 0;
      for (std::size_t nz = 0; nz < h_A.nnzs; nz++) {
        if (h_A.coords[nz] == A_coord_t{m, k}) {
          A_val = h_A.values[nz];
          break;
        }
      }

      value_t B_val = 0;
      for (std::size_t nz = 0; nz < h_B.nnzs; nz++) {
        if (h_B.coords[nz] == B_coord_t{k, n}) {
          B_val = h_B.values[nz];
          break;
        }
      }

      value_t C_val = 0;
      for (std::size_t nz = 0; nz < h_C.nnzs; nz++) {
        if (h_C.coords[nz] == C_coord_t{n, v}) {
          C_val = h_C.values[nz];
          break;
        }
      }

      for (std::size_t nz = 0; nz < h_Z.nnzs; nz++) {
        if (h_Z.coords[nz] == Z_coord_t{m, v}) {
          h_Z.values[nz] += A_val * B_val * C_val;
          break;
        }
      }
    }
    if (h_Z.values.size() != Z.values.size()) {
      std::cout << "Number of elems mismatch! " << h_Z.values.size()
                << " != " << Z.values.size() << std::endl;
    } else {
      std::size_t errors = util::equal(
          Z.values.data().get(), h_Z.values.data(), h_Z.values.size(),
          [](const value_t a, const value_t b) {
            return std::abs(a - b) > 1e-2;
          },
          parameters.verbose);

      std::cout << "Errors:\t\t" << errors << std::endl;
    }
  }

  std::cout << "multi_tensors_mult_thread_mapped_coordinate," << A_name << ".mtx," << B_name << ".mtx,"
            << C_name << ".mtx,M=" << M << ",K=" << K << ",N=" << N
            << ",V=" << V << ",time(ms)=" << timer.milliseconds() << std::endl;

  tracker.generate_output<edge_expr_t, expr_coord_t>(edge_expr, "multi_tensors_mult_thread_mapped_coordinate");
}
