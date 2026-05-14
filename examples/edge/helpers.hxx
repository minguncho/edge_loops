/**
 * @file helpers.hxx
 * @author
 * @brief
 * @version
 * @date
 *
 * @copyright
 *
 */

#pragma once

#include <loops/util/generate.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/container/market.hxx>
#include <loops/util/filepath.hxx>
#include <loops/util/equal.hxx>
#include <loops/util/device.hxx>
#include <loops/util/timer.hxx>
#include <loops/memory.hxx>
#include <cxxopts.hpp>

#include <algorithm>
#include <iostream>

struct parameters_t {
  std::vector<std::string> filenames;
  bool validate;
  bool verbose;
  bool using_seed;
  unsigned int seed_value;
  cxxopts::Options options;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv)
      : options(argv[0], "Edge Expression in Loops") {
    // Add command line options
    options.add_options()("h,help", "Print help")  // help
        ("m,market", "Matrix file(s) (can be specified multiple times)",
         cxxopts::value<std::vector<std::string>>())  // mtx(s)
        ("validate", "CPU validation")                // validate
        ("v,verbose", "Verbose output")               // verbose
        ("s,seed", "Seed value for random value generation",
         cxxopts::value<unsigned int>());  // seed

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help") || (result.count("market") == 0)) {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    if (result.count("market") > 0) {
      auto files = result["market"].as<std::vector<std::string>>();

      for (const auto& file : files) {
        if (loops::is_market(file)) {
          filenames.push_back(file);
        } else {
          std::cerr << "Error: File '" << file
                    << "' is not a valid market file." << std::endl;
          std::exit(0);
        }
      }
    }
    if (filenames.empty()) {
      std::cerr << "No valid matrix market files provided." << std::endl;
      std::exit(0);
    }

    if (result.count("validate") == 1) {
      validate = true;
    } else {
      validate = false;
    }

    if (result.count("verbose") == 1) {
      verbose = true;
    } else {
      verbose = false;
    }

    if (result.count("seed") == 1) {
      using_seed = true;
      seed_value = result["seed"].as<unsigned int>();
    } else {
      using_seed = false;
    }
  }
};