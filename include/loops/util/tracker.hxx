#pragma once

#include <fstream>
#include <vector>
#include <algorithm>
#include <iostream>

namespace loops {

/**
 * @brief Trackers Class. 
 * Tracks which GPU global thread has worked on which nonzero entry.
 */
class Tracker {
 public:

  __host__ Tracker(std::size_t nnzs, std::size_t num_threads)
    : nnzs(nnzs), num_threads(num_threads) { 

    nz_tid.resize(nnzs);
  };

  __host__ void generate_output(std::string alg_name) {
    vector_t<std::size_t, memory_space_t::host> h_nz_tid = nz_tid;

    std::vector<std::vector<std::size_t>> thread_buckets(num_threads);

    // Gather nz entry processed by each tid
    for (std::size_t nz_idx = 0; nz_idx < nnzs; nz_idx++) {
      std::size_t tid = h_nz_tid[nz_idx];
      thread_buckets[tid].push_back(nz_idx);
    }

    // Write to an output file
    std::string filename = "output_ " + alg_name + "_track_report.txt";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
      std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
      return;
    }

    outfile << "Global TID | NZ entries processed\n";
    for (std::size_t tid = 0; tid < num_threads; ++tid) {
      outfile << tid << ":";
      for (const auto& nz_idx : thread_buckets[tid]) {
        outfile << " " << nz_idx;
      }
      outfile << "\n";
    }

    outfile.close();
    std::cout << "Tracker output generated: " << filename << std::endl;
  }

  __host__ vector_t<std::size_t>& get_nz_tid() { return nz_tid; }

 private:
  std::size_t nnzs;
  std::size_t num_threads;

  vector_t<std::size_t> nz_tid;
};

} // namespace loops