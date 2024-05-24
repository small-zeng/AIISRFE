#ifndef _ASTAR2_H
#define _ASTAR2_H

#include <Eigen/Eigen>
#include <iostream>
#include <map>
#include <ros/console.h>
#include <ros/ros.h>
#include <string>
#include <unordered_map>
#include "plan_env/edt_environment.h"
#include <boost/functional/hash.hpp>
#include <queue>
#include <path_searching/matrix_hash.h>
#include <unordered_map>

namespace fast_planner {
// Define the NodeInfo structure
struct NodeInfo {
    double path_length;
    double total_gain;
    int num_points;
    // Add other required information
};

// Define a custom hash function for Eigen::Vector3d
struct EigenVector3dHash {
    std::size_t operator()(const Eigen::Vector3d& vector) const {
        std::hash<double> hash_func;
        std::size_t seed = 0;
        for (int i = 0; i < 3; ++i) {
            seed ^= hash_func(vector[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};




class Node {
public:
  Eigen::Vector3i index;
  Eigen::Vector3d position;
  double g_score, f_score;
  Node* parent;

  /* -------------------- */
  Node() {
    parent = NULL;
  }
  ~Node(){};
};
typedef Node* NodePtr;

class NodeComparator0 {
public:
  bool operator()(NodePtr node1, NodePtr node2) {
    return node1->f_score > node2->f_score;
  }
};

class Astar {
public:
  Astar();
  ~Astar();
  enum { REACH_END = 1, NO_PATH = 2 };

  void init(ros::NodeHandle& nh, const EDTEnvironment::Ptr& env);
  void reset();
  int search(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt);
  void setResolution(const double& res);
  static double pathLength(const vector<Eigen::Vector3d>& path);

  std::vector<Eigen::Vector3d> getPath();
  std::vector<Eigen::Vector3d> getVisited();
  double getEarlyTerminateCost();

  double lambda_heu_;
  double max_search_time_;

  // EVPP NBV
  int search_evpp( const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt, 
                std::function<double(const Eigen::Vector3d&)> get_gain);
  double lamda;
  // Create the unordered_map with the custom hash function
  std::unordered_map<Eigen::Vector3d, NodeInfo, EigenVector3dHash> node_info_map;


private:
  void backtrack(const NodePtr& end_node, const Eigen::Vector3d& end);
  void posToIndex(const Eigen::Vector3d& pt, Eigen::Vector3i& idx);
  double getDiagHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2);
  double getManhHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2);
  double getEuclHeu(const Eigen::Vector3d& x1, const Eigen::Vector3d& x2);

  // main data structure
  vector<NodePtr> path_node_pool_;
  int use_node_num_, iter_num_;
  std::priority_queue<NodePtr, std::vector<NodePtr>, NodeComparator0> open_set_;
  std::unordered_map<Eigen::Vector3i, NodePtr, matrix_hash<Eigen::Vector3i>> open_set_map_;
  std::unordered_map<Eigen::Vector3i, int, matrix_hash<Eigen::Vector3i>> close_set_map_;
  std::vector<Eigen::Vector3d> path_nodes_;
  double early_terminate_cost_;

  EDTEnvironment::Ptr edt_env_;

  // parameter
  double margin_;
  int allocate_num_;
  double tie_breaker_;
  double resolution_, inv_resolution_;
  Eigen::Vector3d map_size_3d_, origin_;
};

}  // namespace fast_planner

#endif