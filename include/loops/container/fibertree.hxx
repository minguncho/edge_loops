/**
 * @file fibertree.hxx
 * @author
 * @brief
 * @version
 * @date
 *
 * @copyright
 *
 */

#pragma once

#include <iostream>
#include <map>
#include <queue>

namespace loops {

using namespace memory;

/**
 * @brief Node container for FiberTree
 *
 * @tparam index_t Type of index.
 */
template <typename index_t>
struct FiberNode {
  std::map<index_t, std::unique_ptr<FiberNode<index_t>>> children;
};

/**
 * @brief FiberTree container
 *
 * @tparam index_t      Type of index.
 * @tparam expr_coord_t Type of coordinate container of iteration
 *                      space of edge expression.
 */
template <typename index_t, typename expr_coord_t>
struct FiberTree {
  FiberNode<index_t> root;
  std::size_t max_depth = expr_coord_t::get_N();

  /**
   * @brief Construct a FiberTree using coordinates in iteration space.
   *
   * @param ranks List of coordinates in iteration space (on host side).
   */
  FiberTree(vector_t<expr_coord_t, memory_space_t::host> h_coords) {
    // Insert each coordinate to a tree
    for (auto& coord : h_coords) {
      FiberNode<index_t>* currentNode = &root;

      // Get each coordinate
      for (std::size_t i = 0; i < max_depth; ++i) {
        index_t c = coord[i];

        if (currentNode->children.find(c) == currentNode->children.end()) {
          currentNode->children[c] = std::make_unique<FiberNode<index_t>>();
        }
        currentNode = currentNode->children[c].get();
      }
    }
  }

  /**
   * @brief Recursive helper for DFS traversal of FiberTree.
   *        Used for gathering coordinates based on position
   *        space partitioning.
   *
   * @param current_node    Current Node of FiberTree
   * @param current_depth   Current depth of FiberTree
   * @param part_sizes      List of partition size for each rank
   * @param current_path    Current collected path/coordinate
   * @param current_tile_id TileID corresponding to current_path
   * @param tile_map        TileID : List of associated coordinates
   */
  void gather_coord_recursive(
      FiberNode<index_t>* current_node,
      std::size_t current_depth,
      std::vector<std::size_t>& part_sizes,
      std::vector<index_t>& current_path,
      std::vector<index_t>& current_tile_id,
      std::map<std::vector<index_t>, std::vector<std::vector<index_t>>>&
          tile_map) {
    if (current_depth == part_sizes.size() || current_node->children.empty()) {
      tile_map[current_tile_id].push_back(current_path);
      return;
    }

    std::size_t P = part_sizes[current_depth];
    std::size_t child_ord_ind = 0;

    for (const auto& [index, childNode] : current_node->children) {
      index_t depthTileId = static_cast<index_t>(child_ord_ind / P);
      current_path.push_back(index);
      current_tile_id.push_back(depthTileId);

      // Recurse deeper
      gather_coord_recursive(childNode.get(), current_depth + 1, part_sizes,
                             current_path, current_tile_id, tile_map);

      // Backtrack state
      current_tile_id.pop_back();
      current_path.pop_back();

      // Advance to the next sequential child position
      child_ord_ind++;
    }
  }

  /**
   * @brief Gathers coordinates based on position space partitioning.
   *
   * @param part_sizes List of partition size for each rank
   */
  std::map<std::vector<index_t>, std::vector<std::vector<index_t>>>
  gather_position_space(std::vector<std::size_t>& part_sizes) {
    std::map<std::vector<index_t>, std::vector<std::vector<index_t>>> tile_map;
    std::vector<index_t> current_path;
    std::vector<index_t> current_tile_id;

    // Start DFS from the root at Depth 0
    gather_coord_recursive(&root, 0, part_sizes, current_path, current_tile_id,
                           tile_map);

    return tile_map;
  }

  /**
   * @brief Print out FiberTree on the host side.
   *
   */
  void print() {
    std::queue<const FiberNode<index_t>*> parentQueue;
    parentQueue.push(&root);
    std::size_t currentDepth = 0;

    while (!parentQueue.empty()) {
      std::size_t parentsAtThisLevel = parentQueue.size();

      // Check if this level actually has any children to display
      bool hasChildren = false;
      std::queue<const FiberNode<index_t>*> checkCopy = parentQueue;
      while (!checkCopy.empty()) {
        if (!checkCopy.front()->children.empty()) {
          hasChildren = true;
          break;
        }
        checkCopy.pop();
      }

      if (!hasChildren)
        break;

      std::cout << "Depth: " << currentDepth << std::endl;

      // Iterate through every parent container at the current depth
      for (std::size_t i = 0; i < parentsAtThisLevel; ++i) {
        const FiberNode<index_t>* currentParent = parentQueue.front();
        parentQueue.pop();

        // If this parent has children, bundle and print them together
        if (!currentParent->children.empty()) {
          std::cout << "(";
          bool first = true;

          for (const auto& [index, childNode] : currentParent->children) {
            if (!first)
              std::cout << ", ";
            std::cout << index;
            first = false;

            // Queue this child up to act as a parent container in the next
            // depth level
            parentQueue.push(childNode.get());
          }

          std::cout << ") ";
        }
      }

      std::cout << "\n";
      currentDepth++;
    }
  }
};

}  // namespace loops