/**
 * @file tracker.hxx
 * @author
 * @brief
 * @version
 * @date
 *
 * @copyright
 *
 */

#pragma once

#include <loops/container/vector.hxx>

#include <fstream>
#include <vector>
#include <algorithm>
#include <iostream>

namespace loops {

/**
 * @brief Tracker Container.
 * Tracks which GPU global thread has worked on which work atom.
 *
 * @tparam atom_id_t Index type for atoms.
 * @tparam space     Memory space (host/device) for storage.
 *
 */
template <typename atom_id_t, memory_space_t space = memory_space_t::device>
struct tracker_t {
  atom_id_t num_atoms;
  std::size_t num_threads;
  vector_t<int, space> coord_tid;

  /**
   * @brief Default constructor
   *
   */
  tracker_t() : num_atoms(0), num_threads(0) {};

  /**
   * @brief Construct of tracker_t
   *
   * @param num_atoms   Number of work atoms.
   * @param num_threads Total number of GPU threads.
   *
   */
  tracker_t(atom_id_t num_atoms, std::size_t num_threads)
      : num_atoms(num_atoms), num_threads(num_threads) {
    coord_tid.resize(num_atoms);
  };

  /**
   * @brief Generate an output file
   *
   * @param edge_expr Edge expression container.
   * @param file_name Name of output file.
   *
   */
  template <typename edge_expr_t, typename expr_coord_t>
  void generate_output(const edge_expr_t& edge_expr,
                       const std::string& file_name) {
    vector_t<std::size_t, memory_space_t::host> h_coord_tid = coord_tid;
    vector_t<expr_coord_t, memory_space_t::host> h_coords = edge_expr.coords;

    std::vector<std::vector<std::size_t>> thr_coords(num_threads);

    // Gather coordinate processed by each tid
    for (std::size_t coord_id = 0; coord_id < h_coords.size(); coord_id++) {
      int tid = h_coord_tid[coord_id];
      thr_coords[tid].push_back(coord_id);
    }

    // Write to an output file
    std::string filename = "output_" + file_name + "_track_report.txt";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
      std::cerr << "Error: Could not open file " << filename << " for writing."
                << std::endl;
      return;
    }

    edge_expr.print(outfile);
    outfile << "------Tracker Report------" << std::endl;
    outfile << "Global TID | Coordinate processed\n";
    for (std::size_t tid = 0; tid < num_threads; ++tid) {
      outfile << tid << ":";
      for (const auto& coord_idx : thr_coords[tid]) {
        outfile << " " << h_coords[coord_idx];
      }
      outfile << "\n";
    }

    outfile.close();
    std::cout << "Tracker output generated: " << filename << std::endl;
  }
};

}  // namespace loops