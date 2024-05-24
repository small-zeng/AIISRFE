#ifndef _FRONTIER_FINDER_H_
#define _FRONTIER_FINDER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <list>
#include <utility>
#include <visualization_msgs/MarkerArray.h>
#include <string>

using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;
using std::list;
using std::pair;

using namespace std;
class RayCaster;

namespace fast_planner {
class EDTEnvironment;
class PerceptionUtils;

// Viewpoint to cover a frontier cluster
struct Viewpoint {
  // Position and heading
  Vector3d pos_;
  double pitch_;
  double yaw_;
  // Fraction of the cluster that can be covered
  // double fraction_;
  int visib_num_;
  double gain_;
  
};

// A frontier cluster, the viewpoints to cover it
struct Frontier {
  // Complete voxels belonging to the cluster
  vector<Vector3d> cells_;
  // down-sampled voxels filtered by voxel grid filter
  vector<Vector3d> filtered_cells_;
  // Average position of all voxels
  Vector3d average_;
  // Idx of cluster
  int id_;
  // Viewpoints that can cover the cluster
  vector<Viewpoint> viewpoints_;
  // Bounding box of cluster, center & 1/2 side length
  Vector3d box_min_, box_max_;
  // Path and cost from this cluster to other clusters
  list<vector<Vector3d>> paths_;
  list<double> costs_;
};

class FrontierFinder {
public:
  FrontierFinder(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh);
  ~FrontierFinder();

  void searchFrontiers(bool is_3D = true);
  void computeFrontiersToVisit(bool is_3D = true);

  void getFrontiers(vector<vector<Vector3d>>& clusters);
  void getDormantFrontiers(vector<vector<Vector3d>>& clusters);
  void getFrontierBoxes(vector<pair<Vector3d, Vector3d>>& boxes);
  // Get viewpoint with highest coverage for each frontier
  void getTopViewpointsInfo(const Vector3d& cur_pos, vector<Eigen::Vector3d>& points, vector<double>& pitchs, 
                            vector<double>& yaws,vector<Eigen::Vector3d>& averages);
  // Get several viewpoints for a subset of frontiers
  void getViewpointsInfo(const Vector3d& cur_pos, const vector<int>& ids, const int& view_num, 
                                       const double& max_decay,vector<vector<Eigen::Vector3d>>& points, 
                                       vector<vector<double>>& pitchs, vector<vector<double>>& yaws);
  void updateFrontierCostMatrix();
  void getFullCostMatrix(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
                         Eigen::MatrixXd& mat);
  void getPathForTour(const Vector3d& pos, const vector<int>& frontier_ids, vector<Vector3d>& path);
  void drawTraj(const vector<Vector3d>& plan_path);

  void setNextFrontier(const int& id);
  bool isFrontierCovered();
  void wrapYaw(double& yaw);

  

  shared_ptr<PerceptionUtils> percep_utils_;

  ros::Publisher views_vis_pub, path_views_vis_pub;
  visualization_msgs::MarkerArray mk_array, path_views_mks;
  bool isNearUnknown(const Vector3d& pos);
  void drawFirstViews();
  void drawAllViews();
  void drawPathViews(vector<Vector3d> & path_pos, vector<double> & path_pitchs, vector<double> & path_yaws);
  // void drawFOV(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2);
  void findGlobalTour(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw, vector<int>& indices);

  // Data
  vector<char> frontier_flag_;
  list<Frontier> frontiers_, dormant_frontiers_, tmp_frontiers_;
  vector<int> removed_ids_;
  list<Frontier>::iterator first_new_ftr_;
  Frontier next_frontier_;

  shared_ptr<EDTEnvironment> edt_env_;

  ros::Publisher traj_pub;

  string tsp_dir_;
  

private:
  void splitLargeFrontiers(list<Frontier>& frontiers, bool is_3D = true);
  bool splitHorizontally(const Frontier& frontier, list<Frontier>& splits);
  bool splitHorizontally_2D(const Frontier& frontier, list<Frontier>& splits);
  void mergeFrontiers(Frontier& ftr1, const Frontier& ftr2);
  bool isFrontierChanged(const Frontier& ft);
  bool haveOverlap(const Vector3d& min1, const Vector3d& max1, const Vector3d& min2,
                   const Vector3d& max2);
  void computeFrontierInfo(Frontier& frontier);
  void downsample(const vector<Vector3d>& cluster_in, vector<Vector3d>& cluster_out);
  void sampleViewpoints(Frontier& frontier, bool is_3D = true);

  int countVisibleCells(const Vector3d& pos, const double& pitch, const double& yaw, const vector<Vector3d>& cluster);
  vector<Eigen::Vector3i> sixNeighbors(const Eigen::Vector3i& voxel);
  vector<Eigen::Vector3i> tenNeighbors(const Eigen::Vector3i& voxel);
  vector<Eigen::Vector3i> allNeighbors(const Eigen::Vector3i& voxel);
  bool isNeighborUnknown(const Eigen::Vector3i& voxel);
  void expandFrontier(const Eigen::Vector3i& first /* , const int& depth, const int& parent_id */);

  // Wrapper of sdf map
  int toadr(const Eigen::Vector3i& idx);
  bool knownfree(const Eigen::Vector3i& idx);
  bool inmap(const Eigen::Vector3i& idx);

  // Deprecated
  Eigen::Vector3i searchClearVoxel(const Eigen::Vector3i& pt);
  bool isInBoxes(const vector<pair<Vector3d, Vector3d>>& boxes, const Eigen::Vector3i& idx);
  bool canBeMerged(const Frontier& ftr1, const Frontier& ftr2);
  void findViewpoints(const Vector3d& sample, const Vector3d& ftr_avg, vector<Viewpoint>& vps);

  // // Data
  // vector<char> frontier_flag_;
  // list<Frontier> frontiers_, dormant_frontiers_, tmp_frontiers_;
  // vector<int> removed_ids_;
  // list<Frontier>::iterator first_new_ftr_;
  // Frontier next_frontier_;
  

  // Params
  int cluster_min_;
  double cluster_size_xy_, cluster_size_z_;
  double candidate_rmax_, candidate_rmin_, candidate_dphi_, min_candidate_dist_,
      min_candidate_clearance_, search_radius_;
  int down_sample_;
  double min_view_finish_fraction_, resolution_;
  int min_visib_num_, candidate_rnum_;

  // Utils
  unique_ptr<RayCaster> raycaster_;
  
};

}  // namespace fast_planner
#endif