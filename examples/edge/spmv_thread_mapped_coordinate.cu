/**
 * Generated Loops code for SpMV.
 * Partition in coordinate space.
 * To be used as a reference, this is
 * what the generated code should look like.
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
          typename B_coord_t>
__global__ void __edge_thread_mapped(setup_t config,
                                     expr_coord_t* expr_coords,
                                     Z_coord_t* Z_coords,
                                     A_coord_t* A_coords,
                                     B_coord_t* B_coords,
                                     value_t* Z_values,
                                     value_t* A_values,
                                     value_t* B_values,
                                     std::size_t Z_nnzs,
                                     std::size_t A_nnzs,
                                     std::size_t B_nnzs,
                                     int* coords_tid) {
  for (auto tile_idx : config.tiles()) {
    for (auto atom : config.atoms(tile_idx)) {
      index_t m = expr_coords[atom][0];
      index_t k = expr_coords[atom][1];

      value_t A_val = 0;
      for (std::size_t nz = 0; nz < A_nnzs; nz++) {
        if (A_coords[nz] == A_coord_t{m, k}) {
          A_val = A_values[nz];
          break;
        }
      }

      value_t B_val = 0;
      for (std::size_t nz = 0; nz < B_nnzs; nz++) {
        if (B_coords[nz] == B_coord_t{k}) {
          B_val = B_values[nz];
          break;
        }
      }

      for (std::size_t nz = 0; nz < Z_nnzs; nz++) {
        if (Z_coords[nz] == Z_coord_t{m}) {
          atomicAdd(&Z_values[nz], A_val * B_val);
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
  if (parameters.filenames.size() != 1) {
    std::cout << "Invalid number of Matrix files, Correct number: 1"
              << std::endl;
    std::exit(0);
  }

  matrix_market_t<index_t, offset_t, value_t> mtx;
  coo_t<index_t, value_t> A_coo = mtx.load(parameters.filenames[0]);

  // Define dimensions
  std::size_t M = A_coo.rows;
  std::size_t K = A_coo.cols;

  using A_coord_t = coords<index_t, 2>;
  using B_coord_t = coords<index_t, 1>;
  using Z_coord_t = coords<index_t, 1>;
  using expr_coord_t = coords<index_t, 2>;

  thrust::device_vector<char> A_ranks = {'M', 'K'};
  tensor_t<index_t, value_t, A_coord_t> A("A", A_coo, A_ranks);

  vector_t<value_t> B_vec(K);
  if (parameters.using_seed) {
    std::cout << "Using seed value: " << parameters.seed_value << std::endl;
    generate::random::uniform_distribution(B_vec.begin(), B_vec.end(), 1, 10,
                                           parameters.seed_value);
  } else {
    generate::random::uniform_distribution(B_vec.begin(), B_vec.end(), 1, 10);
  }
  tensor_t<index_t, value_t, B_coord_t> B("B", B_vec, 'K');

  vector_t<value_t> Z_vec(M);
  tensor_t<index_t, value_t, Z_coord_t> Z("Z", Z_vec, 'M');

  thrust::device_vector<char> expr_ranks = {'M', 'K'};
  thrust::device_vector<std::size_t> expr_dims = {M, K};

  using edge_expr_t = edge_t<index_t, value_t, memory_space_t::device,
                             expr_coord_t, Z_coord_t, A_coord_t, B_coord_t>;

  edge_expr_t edge_expr(expr_ranks, expr_dims, Z, A, B);
  edge_expr.expand_intersected_iteration_points();
  edge_expr.partition_coordinate_space({2, 2});

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
                           A_coord_t, B_coord_t>,
      grid_size, block_size, config, edge_expr.coords.data().get(),
      Z.coords.data().get(), A.coords.data().get(), B.coords.data().get(),
      Z.values.data().get(), A.values.data().get(), B.values.data().get(),
      Z.nnzs, A.nnzs, B.nnzs, tracker.coord_tid.data().get());
  cudaStreamSynchronize(stream);
  timer.stop();

  if (parameters.validate) {
    vector_t<expr_coord_t, memory_space_t::host> h_coords = edge_expr.coords;
    tensor_t<index_t, value_t, A_coord_t, memory_space_t::host> h_A = A;
    tensor_t<index_t, value_t, B_coord_t, memory_space_t::host> h_B = B;
    tensor_t<index_t, value_t, Z_coord_t, memory_space_t::host> h_Z("h_Z",
                                                                    Z_vec, 'M');
    for (auto& coord : h_coords) {
      index_t m = coord[0];
      index_t k = coord[1];

      value_t A_val = 0;
      for (std::size_t nz = 0; nz < h_A.nnzs; nz++) {
        if (h_A.coords[nz] == A_coord_t{m, k}) {
          A_val = h_A.values[nz];
          break;
        }
      }

      value_t B_val = 0;
      for (std::size_t nz = 0; nz < h_B.nnzs; nz++) {
        if (h_B.coords[nz] == B_coord_t{k}) {
          B_val = h_B.values[nz];
          break;
        }
      }

      for (std::size_t nz = 0; nz < h_Z.nnzs; nz++) {
        if (h_Z.coords[nz] == Z_coord_t{m}) {
          h_Z.values[nz] += A_val * B_val;
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

  std::cout << "spmv_thread_mapped_coordinate," << mtx.dataset << ".mtx,M=" << M
            << ",K=" << K << ",";
  if (parameters.using_seed)
    std::cout << "seed=" << parameters.seed_value << ",";
  std::cout << "time(ms)=" << timer.milliseconds() << std::endl;

  tracker.generate_output<edge_expr_t, expr_coord_t>(edge_expr, "spmv_thread_mapped_coordinate");
}
