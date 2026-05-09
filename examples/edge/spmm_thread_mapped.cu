/**
 * Generated Loops code (WIP)
 */

#include "helpers.hxx"
#include <iostream>
#include <loops/container/edge_expr.hxx>
#include <loops/container/layout_edge.hxx>
#include <loops/container/tensor.hxx>
#include <loops/container/matrix.cuh>
#include <loops/memory.hxx>
#include <loops/schedule.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/util/tracker.hxx>

using namespace loops;

/*template <typename setup_t, typename index_t, typename type_t>
__global__ void thread_mapped_edge(setup_t config, const index_t *row_indices,
                                   const index_t *col_indices,
                                   const type_t *values, const type_t *B,
                                   type_t *Z, size_t *nz_tid) {
  for (auto tile_idx : config.tiles()) {
    for (auto atom : config.atoms(tile_idx)) {
      if (atom->get_num_quarks() == 0) {
        continue;
      }

      for (auto quark : config.quarks(atom)) {
        atomicAdd(&(Z[row_indices[*quark]]),
                  values[*quark] * B[col_indices[*quark]]);
        nz_tid[*quark] = (blockIdx.x * blockDim.x) + threadIdx.x;
      }
    }
  }
}*/

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using value_t = float;
  using quarks_t = std::size_t;

  parameters_t parameters(argc, argv);

  // Check filename requirements
  if (parameters.filenames.size() != 1) {
    std::cout << "Invalid number of Matrix files, Correct number: 1" << std::endl;
    std::exit(0);
  }

  matrix_market_t<index_t, offset_t, value_t> mtx;
  coo_t<index_t, value_t> A_coo = mtx.load(parameters.filenames[0]);

  // Define dimensions
  std::size_t M = A_coo.rows;
  std::size_t K = A_coo.cols;
  std::size_t N = 4;

  thrust::device_vector<char> A_ranks = {'M', 'K'};
  tensor_t<index_t, value_t, coords<index_t, 2>> A(
      "A", A_coo, A_ranks);

  thrust::device_vector<char> B_ranks = {'K', 'N'};
  matrix_t<value_t> B_mat(K, N);
  generate::random::uniform_distribution(B_mat.m_data.begin(), B_mat.m_data.end(), 1, 10);
  tensor_t<index_t, value_t, coords<index_t, 2>> B(
      "B", B_mat, B_ranks);

  thrust::device_vector<char> Z_ranks = {'M', 'N'};
  matrix_t<value_t> Z_mat(M, N);
  tensor_t<index_t, value_t, coords<index_t, 2>> Z(
      "Z", Z_mat, Z_ranks);

  A.print();
  B.print();
  Z.print();

  thrust::device_vector<char> expr_ranks = {'M', 'K', 'N'};
  thrust::device_vector<std::size_t> expr_dims = {M, K, N};
  edge_expr_t<index_t, 
              value_t, 
              coords<index_t, 2>, 
              coords<index_t, 2>, 
              coords<index_t, 2>,
              coords<index_t, 3>> edge_expr(A, B, Z, expr_ranks, expr_dims);
  edge_expr.expand_iteration_points();
  edge_expr.partition_coordinate_space({2, 2, 2});
}
