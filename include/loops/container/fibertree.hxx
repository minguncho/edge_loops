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
  std::map<std::vector<index_t>, std::unique_ptr<FiberNode<index_t>>> children;
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

private:
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

    for (const auto& [indices, childNode] : current_node->children) {
      index_t depthTileId = static_cast<index_t>(child_ord_ind / P);

      for (auto& index : indices) {
        current_path.push_back(index);
      }
      current_tile_id.push_back(depthTileId);

      // Recurse deeper
      gather_coord_recursive(childNode.get(), current_depth + 1, part_sizes,
                             current_path, current_tile_id, tile_map);

      // Backtrack state
      current_tile_id.pop_back();
      current_path.resize(current_path.size() - indices.size());

      // Advance to the next sequential child position
      child_ord_ind++;
    }
  }

  /**
   * @brief Recursive helper to flatten specific contiguous depths in the tree.
   * 
   * @param current_node  Pointer to the current node being processed.
   * @param current_depth Current depth in the tree.
   * @param target_depth  The depth level that needs to be absorbed into its parent.
   */
  void flatten_depths_recursive(FiberNode<index_t>* current_node, 
                                std::size_t current_depth, 
                                std::size_t target_depth) {
    if (current_node->children.empty()) {
      return;
    }

    // If we are exactly one level above the target depth, we perform the fusion
    if (current_depth == target_depth - 1) {
      std::map<std::vector<index_t>, std::unique_ptr<FiberNode<index_t>>> flattened_children;

      for (auto& [parent_indices, child_node] : current_node->children) {
        for (auto& [child_indices, grandchild_node] : child_node->children) {
          
          // Create a fused key: concatenate parent indices and child indices
          std::vector<index_t> fused_key = parent_indices;
          fused_key.insert(fused_key.end(), child_indices.begin(), child_indices.end());

          // Move the grandchild node up to become a direct child
          flattened_children[fused_key] = std::move(grandchild_node);
        }
      }

      // Swap the old unflattened map with our newly fused map
      current_node->children = std::move(flattened_children);
      return;
    }

    // Otherwise, keep drilling down to reach the target depth
    for (auto& [indices, child_node] : current_node->children) {
      flatten_depths_recursive(child_node.get(), current_depth + 1, target_depth);
    }
  }

public:
  /**
   * @brief Default Constructor
   *
   */
  FiberTree(): root() {};

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

        if (currentNode->children.find({c}) == currentNode->children.end()) {
          currentNode->children[{c}] = std::make_unique<FiberNode<index_t>>();
        }
        currentNode = currentNode->children[{c}].get();
      }
    }
  }

  /**
   * @brief Indicates whether FiberTree has been formed.
   *
   */
  bool is_empty() {
    return root.children.empty();
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
   * @brief Flattens specified depth levels by fusing parent-child indices.
   *
   * @param target_depths List of depth indices to absorb into their respective parent levels.
   */
  void flatten_depths(std::vector<std::size_t>& target_depths) {
    error::throw_if_exception((target_depths.size() >  max_depth),
            "flatten_depths(): Number of depths to flatten exceeding max!\n");

    error::throw_if_exception((target_depths.size() == 0),
            "flatten_depths(): Empty input depths!\n");

    // Sort depths in descending order to avoid messing up index positions during modification        
    std::vector<std::size_t> sorted_depths = target_depths;
    std::sort(sorted_depths.rbegin(), sorted_depths.rend());

    for (std::size_t depth : sorted_depths) {
      if (depth == 0) {
        // Depth 0 cannot be absorbed into a parent because it's the root level
        continue; 
      }
      
      // Perform the structural compression starting from the root
      flatten_depths_recursive(&root, 0, depth);
      
      // Decrement the max_depth tracking as the tree is now one level shorter
      max_depth--;
    }
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

          for (const auto& [indices, childNode] : currentParent->children) {
            if (!first)
              std::cout << ", ";

            std::cout << "(";
            bool ind_first = true;
            for (auto& index : indices) {
              if (!ind_first)
                std::cout << ", ";
              std::cout << index;
              ind_first = false;
            }
            std::cout << ")";
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