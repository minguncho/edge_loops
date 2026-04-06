/**
 * @file 
 * @author 
 * @brief 
 * @version 
 * @date 
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <cstddef>

namespace loops {
namespace schedule_edge {

/**
 * @brief Load balancing algorithms.
 *
 */
enum algorithms_t {
  thread_mapped,        /// < Thread mapped scheduling algorithm.
};

template <algorithms_t scheme, typename tiles_t, typename tile_size_t>
class tile_traits;

/**
 * @brief Schedule's setup interface.
 *
 * @tparam scheme The scheduling algorithm.
 * @tparam threads_per_block Number of threads per block.
 * @tparam threads_per_tile Number of threads per tile.
 * @tparam tiles_t Type of the tiles.
 * @tparam tile_size_t Type of the tile size (default: std::size_t).
 */
template <algorithms_t scheme,
          std::size_t threads_per_block,
          std::size_t threads_per_tile,
          typename tiles_t,
          typename tile_size_t = std::size_t>
class setup;

}  // namespace schedule_edge
}  // namespace loops

#include <loops/schedule_edge/thread_mapped.hxx>