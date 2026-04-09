/**
 * Generated Loops code
 */

#include "helpers.hxx"
#include <iostream>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/memory.hxx>
#include <loops/schedule_edge.hxx>
#include <loops/util/device.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/partitioner.hxx>
#include <loops/util/tracker.hxx>

using namespace loops;

template <typename setup_t, typename index_t, typename type_t>
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
}

int main(int argc, char **argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;
  using quarks_t = std::size_t;

  parameters_t parameters(argc, argv);

  matrix_market_t<index_t, offset_t, type_t> mtx;
  coo_t<index_t, type_t, memory_space_t::host> A =
      mtx.load(parameters.filename);
  coo_t<index_t, type_t> A_device(A);

  vector_t<type_t> B(A.cols);
  vector_t<type_t> Z(A.rows);

  generate::random::uniform_distribution(B.begin(), B.begin(), 1, 10);

  Partitioner<index_t, type_t, quarks_t> partitioner(A);
  partitioner.partition_coordinate_space(1, 1, 1, A.cols);
  //partitioner.partition_position_space(2, 2);
  partitioner.prepare_gpu();

  using setup_t =
      schedule_edge::setup<schedule_edge::algorithms_t::thread_mapped, 1, 1,
                           WorkTile<quarks_t>>;
  setup_t config(partitioner.get_work_tiles().data().get(),
                 partitioner.get_num_tiles());

  constexpr std::size_t block_size = 128;
  std::size_t grid_size =
      (partitioner.get_num_tiles() + block_size - 1) / block_size;
  cudaStream_t stream = 0;

  Tracker tracker(A.nnzs, block_size * grid_size);

  util::timer_t timer;
  timer.start();

  launch::non_cooperative(
      stream, thread_mapped_edge<setup_t, index_t, type_t>, grid_size,
      block_size, config, A_device.row_indices.data().get(),
      A_device.col_indices.data().get(), A_device.values.data().get(),
      B.data().get(), Z.data().get(), tracker.get_nz_tid().data().get());

  cudaStreamSynchronize(stream);
  timer.stop();

  if (parameters.validate) {
    csr_t<index_t, offset_t, type_t> A_csr(A);
    cpu::validate(parameters, A_csr, B, Z);
  }

  std::cout << "SpMV,"
            << "thread_mapped_edge" << mtx.dataset << "," << A.rows << ","
            << A.cols << "," << A.nnzs << "," << timer.milliseconds()
            << std::endl;

  tracker.generate_output("thread_mapped_edge");
}
