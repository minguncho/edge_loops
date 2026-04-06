/**
 * @file thread_mapped_einsum.hxx
 */

#pragma once

#include <loops/stride_ranges.hxx>
#include <loops/schedule.hxx>

namespace loops {
namespace schedule_edge {

/**
 * @brief Traits for Tile.
 *
 * @todo Implement an tile iterator, right now it is based on CSR only. Can be
 * abstracted very simply by allowing UDF iterators.
 *
 * @tparam tiles_type Type of the tiles.
 * @tparam tile_size_type Type of the tile size (default: std::size_t).
 */
template <typename tiles_type, typename tile_size_type>
class tile_traits<algorithms_t::thread_mapped, tiles_type, tile_size_type> {
 public:
  using tiles_t = tiles_type;
  using tiles_iterator_t = tiles_t*;
  using tile_size_t = tile_size_type;

  __host__ __device__ tile_traits() : size_(0), tiles_(nullptr) {}
  __host__ __device__ tile_traits(tile_size_t size, tiles_iterator_t tiles)
      : size_(size), tiles_(tiles) {}

  __host__ __device__ tile_size_t size() const { return size_; }
  __host__ __device__ tiles_iterator_t begin() { return tiles_; };
  __host__ __device__ tiles_iterator_t end() { return tiles_ + size_; };

 private:
  tile_size_t size_;
  tiles_iterator_t tiles_;
};

/**
 * @brief Thread-mapped schedule's setup interface.
 *
 * @tparam tiles_type Type of the tiles.
 * @tparam tile_size_type Type of the tile size.
 */
template <typename tiles_type,
          typename tile_size_type>
class setup<algorithms_t::thread_mapped,
            1,
            1,
            tiles_type,
            tile_size_type> : public tile_traits<algorithms_t::thread_mapped,
                                                 tiles_type,
                                                 tile_size_type> {
 public:
  using tiles_t = tiles_type;          /// Tile Type
  using tiles_iterator_t = tiles_t*;   /// Tile Iterator Type
  using tile_size_t = tile_size_type;  /// Tile Size Type

  using tile_traits_t =
      tile_traits<algorithms_t::thread_mapped, tiles_type, tile_size_type>;

  /**
   * @brief Default constructor.
   *
   */
  __host__ __device__ setup() : tile_traits_t() {}

  /**
   * @brief Construct a setup object for load balance schedule.
   *
   * @param tiles Tiles iterator.
   * @param num_tiles Number of tiles.
   */
  __host__ __device__ setup(tiles_iterator_t tiles,
                            tile_size_t num_tiles)
      : tile_traits_t(num_tiles, tiles) {}

  /**
   * @brief Creates a range of tile indices to process within a for loop.
   *
   * @example The following code snippet shows how to use this function.
   * \code{.cpp}
   * for (auto t : config.tiles()) {
   *  // Process tile t.
   * }
   * \endcode
   *
   * @return grid_stride_range<tile_size_t> Range of tile indices to process.
   */
  __device__ auto tiles() const {
    return grid_stride_range(tile_size_t(0), tile_traits_t::size());
  }

  /**
   * @brief Creates a range of atoms (object) to process within a given tile index.
   *
   * @param tile_idx Tile Index for which to create the atom range for.
   * @return Range of atoms to process.
   */
  __device__ auto atoms(const tile_size_t tile_idx) {
    return range(tile_traits_t::begin()[tile_idx].begin(), tile_traits_t::begin()[tile_idx].end());
  }

  /**
   * @brief Creates a range of quarks (nz indices) to process within a given atom object.
   *
   * @param atom Atom object for which to create the quarks range for.
   * @return Range of quarks to process.
   */
  __device__ auto quarks(auto atom) {
    return range(atom->begin(), atom->end());
  }

};

}  // namespace schedule_edge
}  // namespace loops
