#include <active_perception/frontier_finder.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>
// #include <path_searching/astar2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <plan_env/edt_environment.h>
#include <active_perception/perception_utils.h>
#include <active_perception/graph_node.h>

// use PCL region growing segmentation
// #include <pcl/point_types.h>
// #include <pcl/search/search.h>
// #include <pcl/search/kdtree.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/segmentation/region_growing.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Eigenvalues>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <opencv2/opencv.hpp>
#include <random>




namespace fast_planner {
FrontierFinder::FrontierFinder(const EDTEnvironment::Ptr& edt, ros::NodeHandle& nh) {
  this->edt_env_ = edt;
  int voxel_num = edt->sdf_map_->getVoxelNum();
  frontier_flag_ = vector<char>(voxel_num, 0);
  fill(frontier_flag_.begin(), frontier_flag_.end(), 0);

  nh.param("frontier/cluster_min", cluster_min_, -1);
  nh.param("frontier/cluster_size_xy", cluster_size_xy_, -1.0);
  nh.param("frontier/cluster_size_z", cluster_size_z_, -1.0);
  nh.param("frontier/min_candidate_dist", min_candidate_dist_, -1.0);
  nh.param("frontier/min_candidate_clearance", min_candidate_clearance_, -1.0);
  nh.param("frontier/candidate_dphi", candidate_dphi_, -1.0);
  nh.param("frontier/candidate_rmax", candidate_rmax_, -1.0);
  nh.param("frontier/candidate_rmin", candidate_rmin_, -1.0);
  nh.param("frontier/candidate_rnum", candidate_rnum_, -1);
  nh.param("frontier/down_sample", down_sample_, -1);
  nh.param("frontier/min_visib_num", min_visib_num_, -1);
  nh.param("frontier/min_view_finish_fraction", min_view_finish_fraction_, -1.0);

  nh.param("frontier/search_radius", search_radius_, -1.0);

  views_vis_pub = nh.advertise<visualization_msgs::MarkerArray>("/planning_vis/all_views", 10);
  path_views_vis_pub = nh.advertise<visualization_msgs::MarkerArray>("/planning_vis/path_views", 10);
  traj_pub = nh.advertise<visualization_msgs::Marker>("/planning_vis/plan_traj", 10);

  raycaster_.reset(new RayCaster);
  resolution_ = edt_env_->sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  edt_env_->sdf_map_->getRegion(origin, size);
  raycaster_->setParams(resolution_, origin);

  percep_utils_.reset(new PerceptionUtils(nh));


}

FrontierFinder::~FrontierFinder() {
}

void FrontierFinder::searchFrontiers(bool is_3D) {
  ros::Time t1 = ros::Time::now();
  tmp_frontiers_.clear();

  // Bounding box of updated region
  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max, true);
  // std::cout << "[update param]: " << update_min << '\t' << update_max << std::endl;

  // Removed changed frontiers in updated map
  auto resetFlag = [&](list<Frontier>::iterator& iter, list<Frontier>& frontiers) {
    Eigen::Vector3i idx;
    for (auto cell : iter->cells_) {
      edt_env_->sdf_map_->posToIndex(cell, idx);
      frontier_flag_[toadr(idx)] = 0;
    }
    iter = frontiers.erase(iter);
  };

  std::cout << "******** Start frontier finder and planning ********" << std::endl;
  // std::cout << "Before remove: " << frontiers_.size() << std::endl;

  removed_ids_.clear();
  int rmv_idx = 0;
  for (auto iter = frontiers_.begin(); iter != frontiers_.end();) {
    // std::cout << "[frontiers param]: " << iter->box_min_ << '\t' << iter->box_max_ << std::endl;
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
        isFrontierChanged(*iter)) {
      resetFlag(iter, frontiers_);
      removed_ids_.push_back(rmv_idx);
    } else {
      ++rmv_idx;
      ++iter;
    }
  }
  // std::cout << "After remove: " << frontiers_.size() << std::endl;
  for (auto iter = dormant_frontiers_.begin(); iter != dormant_frontiers_.end();) {
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
        isFrontierChanged(*iter))
      resetFlag(iter, dormant_frontiers_);
    else
      ++iter;
  }

  // std::cout << "After remove 1: " << dormant_frontiers_.size() << std::endl;

  // Search new frontier within box slightly inflated from updated box
  Vector3d search_min = update_min - Vector3d(1.0, 1.0, 1.0);
  Vector3d search_max = update_max + Vector3d(1.0, 1.0, 1.0);
  Vector3d box_min, box_max;
  edt_env_->sdf_map_->getBox(box_min, box_max);
  for (int k = 0; k < 3; ++k) {
    search_min[k] = max(search_min[k], box_min[k]);
    search_max[k] = min(search_max[k], box_max[k]);
  }
  Eigen::Vector3i min_id, max_id;
  edt_env_->sdf_map_->posToIndex(search_min, min_id);
  edt_env_->sdf_map_->posToIndex(search_max, max_id);

  // cout << "min_id = " << min_id << "max_id = " << max_id << endl;

  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z) {
        // Scanning the updated region to find seeds of frontiers
        Eigen::Vector3i cur(x, y, z);
        // cout << "[Current Idx]: " << x << "\t" << y << "\t" << z << endl;

        Eigen::Vector3d cur_pos;
        edt_env_->sdf_map_->indexToPos(cur, cur_pos);
        // cout << "[Current Pos]: " << cur_pos[0] << "\t" << cur_pos[1] << "\t" << cur_pos[2] << endl;

        // if (cur_pos[0]*cur_pos[0] + cur_pos[2]*cur_pos[2] > search_radius_*search_radius_) continue;
        if (frontier_flag_[toadr(cur)] == 0 && knownfree(cur) && isNeighborUnknown(cur)) {
          // cout << "true" << endl;
          // Expand from the seed cell to find a complete frontier cluster
          expandFrontier(cur);
        }
      }
  splitLargeFrontiers(tmp_frontiers_, is_3D);

  //TODO

  std::cout << "searchFrontiers: " << frontiers_.size()<< std::endl;

  ROS_WARN_THROTTLE(5.0, "Frontier t: %lf", (ros::Time::now() - t1).toSec());
}

void FrontierFinder::expandFrontier(
    const Eigen::Vector3i& first /* , const int& depth, const int& parent_id */) {
  // std::cout << "depth: " << depth << std::endl;
  auto t1 = ros::Time::now();

  // Data for clustering
  queue<Eigen::Vector3i> cell_queue;
  vector<Eigen::Vector3d> expanded;
  Vector3d pos;

  edt_env_->sdf_map_->indexToPos(first, pos);
  expanded.push_back(pos);
  cell_queue.push(first);
  frontier_flag_[toadr(first)] = 1;

  // Search frontier cluster based on region growing (distance clustering)
  while (!cell_queue.empty()) {
    auto cur = cell_queue.front();
    cell_queue.pop();
    auto nbrs = allNeighbors(cur);
    for (auto nbr : nbrs) {
      // Qualified cell should be inside bounding box and frontier cell not clustered
      int adr = toadr(nbr);
      if (frontier_flag_[adr] == 1 || !edt_env_->sdf_map_->isInBox(nbr) ||
          !(knownfree(nbr) && isNeighborUnknown(nbr)))
        continue;

      edt_env_->sdf_map_->indexToPos(nbr, pos);
      // if (pos[2] < 0.4) continue;  // Remove noise close to ground
      expanded.push_back(pos);
      cell_queue.push(nbr);
      frontier_flag_[adr] = 1;
    }
  }

  // cout << "expanded.size() = " << expanded.size() << endl;
  if (expanded.size() > cluster_min_) {
    // Compute detailed info
    Frontier frontier;
    frontier.cells_ = expanded;
    computeFrontierInfo(frontier);
    tmp_frontiers_.push_back(frontier);
  }
}

void FrontierFinder::splitLargeFrontiers(list<Frontier>& frontiers, bool is_3D) {
  list<Frontier> splits, tmps;
  if (is_3D){
      for (auto it = frontiers.begin(); it != frontiers.end(); ++it) {
        // Check if each frontier needs to be split horizontally
        if (splitHorizontally(*it, splits)) {
          tmps.insert(tmps.end(), splits.begin(), splits.end());
          splits.clear();
        } else
          tmps.push_back(*it);
      }
      frontiers = tmps;
  }
  else{
      for (auto it = frontiers.begin(); it != frontiers.end(); ++it) {
        // Check if each frontier needs to be split horizontally
        if (splitHorizontally_2D(*it, splits)) {
          tmps.insert(tmps.end(), splits.begin(), splits.end());
          splits.clear();
        } else
          tmps.push_back(*it);
      }
      frontiers = tmps;
  }


}

bool FrontierFinder::splitHorizontally(const Frontier& frontier, list<Frontier>& splits) {
  // Split a frontier into small piece if it is too large
  auto mean = frontier.average_.head<3>();
  bool need_split = false;
  for (auto cell : frontier.filtered_cells_) {
    if ((cell.head<3>() - mean).norm() > cluster_size_xy_) {
      need_split = true;
      break;
    }
  }

  // cout << "need_split = " << need_split << endl;
  
  if (!need_split) return false;

  // Compute principal component
  // Covariance matrix of cells
  Eigen::Matrix3d cov;
  cov.setZero();
  for (auto cell : frontier.filtered_cells_) {
    Eigen::Vector3d diff = cell.head<3>() - mean;
    cov += diff * diff.transpose();
  }
  cov /= double(frontier.filtered_cells_.size());

  // Find eigenvector corresponds to maximal eigenvector
  Eigen::EigenSolver<Eigen::Matrix3d> es(cov);
  auto values = es.eigenvalues().real();
  auto vectors = es.eigenvectors().real();
  int max_idx;
  double max_eigenvalue = -1000000;
  for (int i = 0; i < values.rows(); ++i) {
    if (values[i] > max_eigenvalue) {
      max_idx = i;
      max_eigenvalue = values[i];
    }
  }
  Eigen::Vector3d first_pc = vectors.col(max_idx);
  // std::cout << "max idx: " << max_idx << std::endl;
  // std::cout << "mean: " << mean.transpose() << ", first pc: " << first_pc.transpose() << std::endl;

  // Split the frontier into two groups along the first PC
  Frontier ftr1, ftr2;
  for (auto cell : frontier.cells_) {
    if ((cell.head<3>() - mean).dot(first_pc) >= 0)
      ftr1.cells_.push_back(cell);
    else
      ftr2.cells_.push_back(cell);
  }
  computeFrontierInfo(ftr1);
  computeFrontierInfo(ftr2);

  // Recursive call to split frontier that is still too large
  list<Frontier> splits2;
  if (splitHorizontally(ftr1, splits2)) {
    splits.insert(splits.end(), splits2.begin(), splits2.end());
    splits2.clear();
  } else
    splits.push_back(ftr1);

  if (splitHorizontally(ftr2, splits2))
    splits.insert(splits.end(), splits2.begin(), splits2.end());
  else
    splits.push_back(ftr2);

  return true;
}

bool FrontierFinder::splitHorizontally_2D(const Frontier& frontier, list<Frontier>& splits) {
  // Split a frontier into small piece if it is too large
  Eigen::Vector2d mean;
  mean << frontier.average_(0), frontier.average_(2);
  bool need_split = false;
  Eigen::Vector2d cell_tmp;
  for (auto cell : frontier.filtered_cells_) {
    cell_tmp << cell(0), cell(2);
    if ((cell_tmp - mean).norm() > cluster_size_xy_) {
      need_split = true;
      break;
    }
  }
  if (!need_split) return false;

  // Compute principal component
  // Covariance matrix of cells
  Eigen::Matrix2d cov;
  cov.setZero();
  for (auto cell : frontier.filtered_cells_) {
    cell_tmp << cell(0), cell(2);
    Eigen::Vector2d diff = cell_tmp - mean;
    cov += diff * diff.transpose();
  }
  cov /= double(frontier.filtered_cells_.size());

  // Find eigenvector corresponds to maximal eigenvector
  Eigen::EigenSolver<Eigen::Matrix2d> es(cov);
  auto values = es.eigenvalues().real();
  auto vectors = es.eigenvectors().real();
  int max_idx;
  double max_eigenvalue = -1000000;
  for (int i = 0; i < values.rows(); ++i) {
    if (values[i] > max_eigenvalue) {
      max_idx = i;
      max_eigenvalue = values[i];
    }
  }
  Eigen::Vector2d first_pc = vectors.col(max_idx);
  // std::cout << "max idx: " << max_idx << std::endl;
  // std::cout << "mean: " << mean.transpose() << ", first pc: " << first_pc.transpose() << std::endl;

  // Split the frontier into two groups along the first PC
  Frontier ftr1, ftr2;
  for (auto cell : frontier.cells_) {
    cell_tmp << cell(0), cell(2);
    if ((cell_tmp - mean).dot(first_pc) >= 0)
      ftr1.cells_.push_back(cell);
    else
      ftr2.cells_.push_back(cell);
  }
  computeFrontierInfo(ftr1);
  computeFrontierInfo(ftr2);

  // Recursive call to split frontier that is still too large
  list<Frontier> splits2;
  if (splitHorizontally(ftr1, splits2)) {
    splits.insert(splits.end(), splits2.begin(), splits2.end());
    splits2.clear();
  } else
    splits.push_back(ftr1);

  if (splitHorizontally(ftr2, splits2))
    splits.insert(splits.end(), splits2.begin(), splits2.end());
  else
    splits.push_back(ftr2);

  return true;
}

bool FrontierFinder::isInBoxes(
    const vector<pair<Vector3d, Vector3d>>& boxes, const Eigen::Vector3i& idx) {
  Vector3d pt;
  edt_env_->sdf_map_->indexToPos(idx, pt);
  for (auto box : boxes) {
    // Check if contained by a box
    bool inbox = true;
    for (int i = 0; i < 3; ++i) {
      inbox = inbox && pt[i] > box.first[i] && pt[i] < box.second[i];
      if (!inbox) break;
    }
    if (inbox) return true;
  }
  return false;
}

void FrontierFinder::updateFrontierCostMatrix() {
  // std::cout << "cost mat size before remove: " << std::endl;
  // for (auto ftr : frontiers_)
  //   std::cout << "(" << ftr.costs_.size() << "," << ftr.paths_.size() << "), ";
  // std::cout << "" << std::endl;

  std::cout << "cost mat size remove: " << std::endl;
  if (!removed_ids_.empty()) {
    // Delete path and cost for removed clusters
    for (auto it = frontiers_.begin(); it != first_new_ftr_; ++it) {
      auto cost_iter = it->costs_.begin();
      auto path_iter = it->paths_.begin();
      int iter_idx = 0;
      for (int i = 0; i < removed_ids_.size(); ++i) {
        // Step iterator to the item to be removed
        while (iter_idx < removed_ids_[i]) {
          ++cost_iter;
          ++path_iter;
          ++iter_idx;
        }
        cost_iter = it->costs_.erase(cost_iter);
        path_iter = it->paths_.erase(path_iter);
      }
      std::cout << "(" << it->costs_.size() << "," << it->paths_.size() << "), ";
    }
    removed_ids_.clear();
  }
  std::cout << "" << std::endl;

  auto updateCost = [](const list<Frontier>::iterator& it1, const list<Frontier>::iterator& it2) {
    // std::cout << "(" << it1->id_ << "," << it2->id_ << "), ";
    // Search path from old cluster's top viewpoint to new cluster'
    Viewpoint& vui = it1->viewpoints_.front();
    Viewpoint& vuj = it2->viewpoints_.front();
    vector<Vector3d> path_ij;
    double cost_ij = ViewNode::computeCost(
        vui.pos_, vuj.pos_, vui.yaw_, vuj.yaw_, Vector3d(0, 0, 0), 0, path_ij);
    // Insert item for both old and new clusters
    it1->costs_.push_back(cost_ij);
    it1->paths_.push_back(path_ij);
    reverse(path_ij.begin(), path_ij.end());
    it2->costs_.push_back(cost_ij);
    it2->paths_.push_back(path_ij);
  };

  std::cout << "cost mat add: " << std::endl;
  // Compute path and cost between old and new clusters
  for (auto it1 = frontiers_.begin(); it1 != first_new_ftr_; ++it1)
    for (auto it2 = first_new_ftr_; it2 != frontiers_.end(); ++it2)
      updateCost(it1, it2);

  // Compute path and cost between new clusters
  for (auto it1 = first_new_ftr_; it1 != frontiers_.end(); ++it1)
    for (auto it2 = it1; it2 != frontiers_.end(); ++it2) {
      if (it1 == it2) {
        std::cout << "(" << it1->id_ << "," << it2->id_ << "), ";
        it1->costs_.push_back(0);
        it1->paths_.push_back({});
      } else
        updateCost(it1, it2);
    }
  std::cout << "" << std::endl;
  // std::cout << "cost mat size final: " << std::endl;
  // for (auto ftr : frontiers_)
  //   std::cout << "(" << ftr.costs_.size() << "," << ftr.paths_.size() << "), ";
  // std::cout << "" << std::endl;


  std::cout << "updateFrontierCostMatrix: " << frontiers_.size()<< std::endl;
}

void FrontierFinder::mergeFrontiers(Frontier& ftr1, const Frontier& ftr2) {
  // Merge ftr2 into ftr1
  ftr1.average_ =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  ftr1.cells_.insert(ftr1.cells_.end(), ftr2.cells_.begin(), ftr2.cells_.end());
  computeFrontierInfo(ftr1);
}

bool FrontierFinder::canBeMerged(const Frontier& ftr1, const Frontier& ftr2) {
  Vector3d merged_avg =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  // Check if it can merge two frontier without exceeding size limit
  for (auto c1 : ftr1.cells_) {
    auto diff = c1 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  for (auto c2 : ftr2.cells_) {
    auto diff = c2 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  return true;
}

bool FrontierFinder::haveOverlap(
    const Vector3d& min1, const Vector3d& max1, const Vector3d& min2, const Vector3d& max2) {
  // Check if two box have overlap part
  Vector3d bmin, bmax;
  for (int i = 0; i < 3; ++i) {
    bmin[i] = max(min1[i], min2[i]);
    bmax[i] = min(max1[i], max2[i]);
    if (bmin[i] > bmax[i] + 1e-3) return false;
  }
  return true;
}

bool FrontierFinder::isFrontierChanged(const Frontier& ft) {
  for (auto cell : ft.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!(knownfree(idx) && isNeighborUnknown(idx))) return true;
  }
  return false;
}

void FrontierFinder::computeFrontierInfo(Frontier& ftr) {
  // Compute average position and bounding box of cluster
  ftr.average_.setZero();
  ftr.box_max_ = ftr.cells_.front();
  ftr.box_min_ = ftr.cells_.front();
  for (auto cell : ftr.cells_) {
    ftr.average_ += cell;
    for (int i = 0; i < 3; ++i) {
      ftr.box_min_[i] = min(ftr.box_min_[i], cell[i]);
      ftr.box_max_[i] = max(ftr.box_max_[i], cell[i]);
    }
  }
  ftr.average_ /= double(ftr.cells_.size());

  // Compute downsampled cluster
  downsample(ftr.cells_, ftr.filtered_cells_);
}

void FrontierFinder::computeFrontiersToVisit(bool is_3D) {    //TODO
  cout << "[FrontierFinder]is_3D = " << (is_3D ? "true" : "false") << endl;
  first_new_ftr_ = frontiers_.end();
  int new_num = 0;
  int new_dormant_num = 0;
  // Try find viewpoints for each cluster and categorize them according to viewpoint number
  // cout << "tmp_frontiers_ : " << tmp_frontiers_.size() << endl; 
  for (auto& tmp_ftr : tmp_frontiers_) {
    // Search viewpoints around frontier
    // cout << "tmp_ftr come " << endl;
    sampleViewpoints(tmp_ftr, is_3D);
    // cout << "tmp_ftr.viewpoints_ = " << tmp_ftr.viewpoints_.size() << endl;
    if (!tmp_ftr.viewpoints_.empty()) {
      ++new_num;
      list<Frontier>::iterator inserted = frontiers_.insert(frontiers_.end(), tmp_ftr);
      // Sort the viewpoints by coverage fraction, best view in front
      sort(
          inserted->viewpoints_.begin(), inserted->viewpoints_.end(),
          [](const Viewpoint& v1, const Viewpoint& v2) { return v1.visib_num_ > v2.visib_num_; });
      if (first_new_ftr_ == frontiers_.end()) {
        first_new_ftr_ = inserted;
        // cout << "first_new_ftr_ = inserted " << endl;
      }
      
      
    } else {
      // Find no viewpoint, move cluster to dormant list
      dormant_frontiers_.push_back(tmp_ftr);
      ++new_dormant_num;
    }
  }
  // Reset indices of frontiers
  int idx = 0;
  for (auto& ft : frontiers_) {
    ft.id_ = idx++;
    std::cout << ft.id_ << ", ";
  }
  std::cout << "computeFrontiersToVisit: " << frontiers_.size()<< std::endl;
  std::cout << "new num: " << new_num << ", new dormant: " << new_dormant_num << std::endl;
  std::cout << "to visit: " << frontiers_.size() << ", dormant: " << dormant_frontiers_.size()
            << std::endl;
}

void FrontierFinder::getTopViewpointsInfo(
    const Vector3d& cur_pos, vector<Eigen::Vector3d>& points, vector<double>& pitchs, vector<double>& yaws,
    vector<Eigen::Vector3d>& averages) {
  points.clear();
  pitchs.clear();
  yaws.clear();
  averages.clear();
  for (auto frontier : frontiers_) {
    bool no_view = true;
    for (auto view : frontier.viewpoints_) {
      // Retrieve the first viewpoint that is far enough and has highest coverage
      if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;
      points.push_back(view.pos_);
      pitchs.push_back(view.pitch_);
      yaws.push_back(view.yaw_);
      averages.push_back(frontier.average_);
      no_view = false;
      break;
    }
    if (no_view) {
      // All viewpoints are very close, just use the first one (with highest coverage).
      auto view = frontier.viewpoints_.front();
      points.push_back(view.pos_);
      pitchs.push_back(view.pitch_);
      yaws.push_back(view.yaw_);
      averages.push_back(frontier.average_);
    }
  }
}

void FrontierFinder::getViewpointsInfo(
    const Vector3d& cur_pos, const vector<int>& ids, const int& view_num, const double& max_decay,
    vector<vector<Eigen::Vector3d>>& points, vector<vector<double>>& pitchs, vector<vector<double>>& yaws) {
  points.clear();
  pitchs.clear();
  yaws.clear();
  for (auto id : ids) {
    // Scan all frontiers to find one with the same id
    for (auto frontier : frontiers_) {
      if (frontier.id_ == id) {
        // Get several top viewpoints that are far enough
        vector<Eigen::Vector3d> pts;
        vector<double> ps, ys;
        int visib_thresh = frontier.viewpoints_.front().visib_num_ * max_decay;
        for (auto view : frontier.viewpoints_) {
          if (pts.size() >= view_num || view.visib_num_ <= visib_thresh) break;
          if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;
          pts.push_back(view.pos_);
          ps.push_back(view.pitch_);
          ys.push_back(view.yaw_);
        }
        if (pts.empty()) {
          // All viewpoints are very close, ignore the distance limit
          for (auto view : frontier.viewpoints_) {
            if (pts.size() >= view_num || view.visib_num_ <= visib_thresh) break;
            pts.push_back(view.pos_);
            ps.push_back(view.pitch_);
            ys.push_back(view.yaw_);
          }
        }
        points.push_back(pts);
        pitchs.push_back(ps);
        yaws.push_back(ys);
      }
    }
  }
}

void FrontierFinder::getFrontiers(vector<vector<Eigen::Vector3d>>& clusters) {
  clusters.clear();
  for (auto frontier : frontiers_)
    clusters.push_back(frontier.cells_);
  // clusters.push_back(frontier.filtered_cells_);
}

void FrontierFinder::getDormantFrontiers(vector<vector<Vector3d>>& clusters) {
  clusters.clear();
  for (auto ft : dormant_frontiers_)
    clusters.push_back(ft.cells_);
}

void FrontierFinder::getFrontierBoxes(vector<pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes) {
  boxes.clear();
  for (auto frontier : frontiers_) {
    Vector3d center = (frontier.box_max_ + frontier.box_min_) * 0.5;
    Vector3d scale = frontier.box_max_ - frontier.box_min_;
    boxes.push_back(make_pair(center, scale));
  }
}

void FrontierFinder::getPathForTour(
    const Vector3d& pos, const vector<int>& frontier_ids, vector<Vector3d>& path) {
  // Make an frontier_indexer to access the frontier list easier
  vector<list<Frontier>::iterator> frontier_indexer;
  for (auto it = frontiers_.begin(); it != frontiers_.end(); ++it)
    frontier_indexer.push_back(it);

  // Compute the path from current pos to the first frontier
  vector<Vector3d> segment;
  ViewNode::searchPath(pos, frontier_indexer[frontier_ids[0]]->viewpoints_.front().pos_, segment);
  path.insert(path.end(), segment.begin(), segment.end());

  // // Get paths of tour passing all clusters
  // for (int i = 0; i < frontier_ids.size() - 1; ++i) {
  //   // Move to path to next cluster
  //   auto path_iter = frontier_indexer[frontier_ids[i]]->paths_.begin();
  //   int next_idx = frontier_ids[i + 1];
  //   for (int j = 0; j < next_idx; ++j)
  //     ++path_iter;
  //   path.insert(path.end(), path_iter->begin(), path_iter->end());
  // }

  // Get paths of tour passing all clusters
  int N = min(int(frontier_ids.size()), 10);
  for (int i = 0; i < N - 1; ++i) {
    // Move to path to next cluster
    vector<Vector3d> segment;
    auto p1 = frontier_indexer[frontier_ids[i]]->viewpoints_.front().pos_;
    auto p2 = frontier_indexer[frontier_ids[i + 1]]->viewpoints_.front().pos_;
    double path_length = ViewNode::searchPath(p1, p2, segment);
    if(path_length == 5000) break;
    path.insert(path.end(), segment.begin(), segment.end());

  }


}

void FrontierFinder::getFullCostMatrix(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
    Eigen::MatrixXd& mat) {
    // Use Asymmetric TSP
    int dimen = frontiers_.size();
    mat.resize(dimen + 1, dimen + 1);
    // std::cout << "mat size: " << mat.rows() << ", " << mat.cols() << std::endl;
    // Fill block for clusters
    int i = 1, j = 1;
    // for (auto ftr : frontiers_) {
    //   for (auto cs : ftr.costs_) {
    //     // std::cout << "(" << i << ", " << j << ")"
    //     // << ", ";
    //     mat(i, j++) = cs;
    //   }
    //   ++i;
    //   j = 1;
    // }

    for (auto start_ftr : frontiers_) {
      for (auto end_ftr : frontiers_) {
        // std::cout << "(" << i << ", " << j << ")"
        // << ", ";
        vector<Vector3d> path;
        Viewpoint vi = start_ftr.viewpoints_.front();
        Viewpoint vj = end_ftr.viewpoints_.front();
        double cost_ij = ViewNode::computeCostNew(vi.pos_, vj.pos_, vi.yaw_, vj.yaw_, vi.pitch_, vj.pitch_, path);
        mat(i, j++) = cost_ij;
      }
      ++i;
      j = 1;
    }

    // std::cout << "" << std::endl;

    // Fill block from current state to clusters
    mat.leftCols<1>().setZero();
    for (auto ftr : frontiers_) {
      // std::cout << "(0, " << j << ")"
      // << ", ";
      Viewpoint vj = ftr.viewpoints_.front();
      vector<Vector3d> path;
      mat(0, j++) =
          ViewNode::computeCost(cur_pos, vj.pos_, cur_yaw[0], vj.yaw_, cur_vel, cur_yaw[1], path);
    }
    // std::cout << "" << std::endl;
  
}

void FrontierFinder::findViewpoints(
    const Vector3d& sample, const Vector3d& ftr_avg, vector<Viewpoint>& vps) {
  if (!edt_env_->sdf_map_->isInBox(sample) ||
      edt_env_->sdf_map_->getInflateOccupancy(sample) == 1 || isNearUnknown(sample))
    return;

  double left_angle_, right_angle_, vertical_angle_, ray_length_;

  // Central yaw is determined by frontier's average position and sample
  auto dir = ftr_avg - sample;
  double hc = atan2(dir[1], dir[0]);

  vector<int> slice_gains;
  // Evaluate info gain of different slices
  for (double phi_h = -M_PI_2; phi_h <= M_PI_2 + 1e-3; phi_h += M_PI / 18) {
    // Compute gain of one slice
    int gain = 0;
    for (double phi_v = -vertical_angle_; phi_v <= vertical_angle_; phi_v += vertical_angle_ / 3) {
      // Find endpoint of a ray
      Vector3d end;
      end[0] = sample[0] + ray_length_ * cos(phi_v) * cos(hc + phi_h);
      end[1] = sample[1] + ray_length_ * cos(phi_v) * sin(hc + phi_h);
      end[2] = sample[2] + ray_length_ * sin(phi_v);

      // Do raycasting to check info gain
      Vector3i idx;
      raycaster_->input(sample, end);
      while (raycaster_->nextId(idx)) {
        // Hit obstacle, stop the ray
        if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 || !edt_env_->sdf_map_->isInBox(idx))
          break;
        // Count number of unknown cells
        if (edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) ++gain;
      }
    }
    slice_gains.push_back(gain);
  }

  // Sum up slices' gain to get different yaw's gain
  vector<pair<double, int>> yaw_gains;
  for (int i = 0; i < 6; ++i)  // [-90,-10]-> [10,90], delta_yaw = 20, 6 groups
  {
    double yaw = hc - M_PI_2 + M_PI / 9.0 * i + right_angle_;
    int gain = 0;
    for (int j = 2 * i; j < 2 * i + 9; ++j)  // 80 degree hFOV, 9 slices
      gain += slice_gains[j];
    yaw_gains.push_back(make_pair(yaw, gain));
  }

  // Get several yaws with highest gain
  vps.clear();
  sort(
      yaw_gains.begin(), yaw_gains.end(),
      [](const pair<double, int>& p1, const pair<double, int>& p2) {
        return p1.second > p2.second;
      });
  for (int i = 0; i < 3; ++i) {
    if (yaw_gains[i].second < min_visib_num_) break;
    Viewpoint vp = { sample, yaw_gains[i].first, yaw_gains[i].second };
    while (vp.yaw_ < -M_PI)
      vp.yaw_ += 2 * M_PI;
    while (vp.yaw_ > M_PI)
      vp.yaw_ -= 2 * M_PI;
    vps.push_back(vp);
  }
}

// Sample viewpoints around frontier's average position, check coverage to the frontier cells
void FrontierFinder::sampleViewpoints(Frontier& frontier, bool is_3D) {
  // cout << "[FrontierFinder]is_3D = " << (is_3D ? "true" : "false") << endl;
  std::random_device rd;
  std::mt19937 gen(rd());
  double min_pitch, max_pitch;
  if (is_3D){
    min_pitch = -M_PI/2.0+M_PI/18; //  TODO
    max_pitch = M_PI/4.0-M_PI/18;
  }
  else{
    min_pitch = 0.0;
    max_pitch = M_PI/36.0;
  }

  // Evaluate sample viewpoints on circles, find ones that cover most cells
  for (double rc = candidate_rmin_, dr = (candidate_rmax_ - candidate_rmin_) / candidate_rnum_;
       rc <= candidate_rmax_ + 1e-3; rc += dr)
    for (double phi = -M_PI/3.0; phi < M_PI/3.0; phi += candidate_dphi_) {
      for (double theta = min_pitch; theta <= max_pitch; theta += M_PI/18.0){            // by zj

        Vector3d sample_pos = frontier.average_ + rc * Vector3d(cos(theta)*cos(phi),-sin(theta), cos(theta)*sin(phi));
        // add gauss noise to sample_pos
        // 创建一个随机数引擎
        std::normal_distribution<double> gaussian_dist(0.0, 0.05); // 均值为0，标准差为0.1，根据需要调整
        // 生成高斯噪声
        // double noise_x = gaussian_dist(gen);
        // double noise_y = gaussian_dist(gen);
        // double noise_z = gaussian_dist(gen);
        // sample_pos += Vector3d(noise_x, noise_y, noise_z);



        // if(sample_pos[1] > 1.9 || sample_pos[1]<0.7 ) continue;  // by zj   TODO
        // Qualified viewpoint is in bounding box and in safe region
        if (!edt_env_->sdf_map_->isInBox(sample_pos) ||
            edt_env_->sdf_map_->getInflateOccupancy(sample_pos) == 1 || isNearUnknown(sample_pos))
          continue;

        if (edt_env_->sdf_map_->getDistance(sample_pos) < 0.2) continue;   //TODO

        Eigen::Vector3d sample_pos_dir = sample_pos.normalized();
        double sample_yaw = atan2(sample_pos_dir[0], sample_pos_dir[2]);
        double sample_pitch = atan2(-sample_pos_dir[1], sqrt(sample_pos_dir[0]*sample_pos_dir[0]+sample_pos_dir[2]*sample_pos_dir[2]));
        wrapYaw(sample_yaw);
        // cout << "[Data now !] " << "[sample_yaw]: " << sample_yaw << ", [sample_pitch]: " << sample_pitch << endl;
        if (sample_pitch < -M_PI_2 || sample_pitch > M_PI_4)  continue;

        // cout << "[Accept !] " << "[sample_yaw]: " << sample_yaw << ", [sample_pitch]: " << sample_pitch << endl;

        auto& cells = frontier.filtered_cells_;
        double avg_yaw = 0.0;
        double avg_pitch = 0.0;
        for (int i = 0; i < cells.size(); ++i) {
          Eigen::Vector3d dir = (cells[i] - sample_pos).normalized();
          avg_yaw = avg_yaw  + atan2(dir[0], dir[2]);  // by zj
          avg_pitch = avg_pitch  + atan2(-dir[1], sqrt(dir[0]*dir[0]+dir[2]*dir[2]));  // by zj
        }
        avg_yaw = avg_yaw / cells.size() ;  // by zj
        avg_pitch = avg_pitch / cells.size() ;  // by zj
        if (!is_3D) avg_pitch = 0.0;

        wrapYaw(avg_yaw);
        // cout << avg_yaw << ", " << avg_pitch << endl;

        // Compute the fraction of covered and visible cells
        int visib_num = countVisibleCells(sample_pos, avg_pitch,avg_yaw, cells);
        // cout << "[frontier_fronter]: " << "visib_num = " << visib_num << endl;
        if (visib_num > min_visib_num_  || false) {
          // Viewpoint vp = { sample_pos, avg_yaw, 100 };
          Viewpoint vp = { sample_pos, avg_pitch,avg_yaw, visib_num };
          frontier.viewpoints_.push_back(vp);
          // int gain = findMaxGainYaw(sample_pos, frontier, sample_yaw);
        }
        // }
      }
    }
}

bool FrontierFinder::isFrontierCovered() {
  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max);

  auto checkChanges = [&](const list<Frontier>& frontiers) {
    for (auto ftr : frontiers) {
      if (!haveOverlap(ftr.box_min_, ftr.box_max_, update_min, update_max)) continue;
      const int change_thresh = min_view_finish_fraction_ * ftr.cells_.size();
      int change_num = 0;
      for (auto cell : ftr.cells_) {
        Eigen::Vector3i idx;
        edt_env_->sdf_map_->posToIndex(cell, idx);
        if (!(knownfree(idx) && isNeighborUnknown(idx)) && ++change_num >= change_thresh)
          return true;
      }
    }
    return false;
  };

  if (checkChanges(frontiers_) || checkChanges(dormant_frontiers_)) return true;

  return false;
}

bool FrontierFinder::isNearUnknown(const Eigen::Vector3d& pos) {
  const int vox_num = floor(min_candidate_clearance_ / resolution_);
  for (int x = -vox_num; x <= vox_num; ++x)
    for (int y = -vox_num; y <= vox_num; ++y)
      for (int z = -vox_num; z <= vox_num; ++z) {
        Eigen::Vector3d vox;
        vox << pos[0] + x * resolution_, pos[1] + y * resolution_, pos[2] + z * resolution_;
        if (edt_env_->sdf_map_->getOccupancy(vox) == SDFMap::UNKNOWN) return true;
      }
  return false;
}

int FrontierFinder::countVisibleCells(
    const Eigen::Vector3d& pos,  const double& pitch, const double& yaw, const vector<Eigen::Vector3d>& cluster) {
  percep_utils_->setPose(pos, pitch,yaw);
  int visib_num = 0;
  Eigen::Vector3i idx;
  for (auto cell : cluster) {
    // Check if frontier cell is inside FOV
    if (!percep_utils_->insideFOV(cell)) continue;

    // Check if frontier cell is visible (not occulded by obstacles)
    raycaster_->input(cell, pos);
    bool visib = true;
    while (raycaster_->nextId(idx)) {
      if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 ||
          edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        visib = false;
        break;
      }
    }
    if (visib) visib_num += 1;
  }
  return visib_num;
}

void FrontierFinder::downsample(
    const vector<Eigen::Vector3d>& cluster_in, vector<Eigen::Vector3d>& cluster_out) {
  // downsamping cluster
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudf(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto cell : cluster_in)
    cloud->points.emplace_back(cell[0], cell[1], cell[2]);

  const double leaf_size = edt_env_->sdf_map_->getResolution() * down_sample_;
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(leaf_size, leaf_size, leaf_size);
  sor.filter(*cloudf);

  cluster_out.clear();
  for (auto pt : cloudf->points)
    cluster_out.emplace_back(pt.x, pt.y, pt.z);
}

void FrontierFinder::wrapYaw(double& yaw) {
  while (yaw < -M_PI)
    yaw += 2 * M_PI;
  while (yaw > M_PI)
    yaw -= 2 * M_PI;
}

Eigen::Vector3i FrontierFinder::searchClearVoxel(const Eigen::Vector3i& pt) {
  queue<Eigen::Vector3i> init_que;
  vector<Eigen::Vector3i> nbrs;
  Eigen::Vector3i cur, start_idx;
  init_que.push(pt);
  // visited_flag_[toadr(pt)] = 1;

  while (!init_que.empty()) {
    cur = init_que.front();
    init_que.pop();
    if (knownfree(cur)) {
      start_idx = cur;
      break;
    }

    nbrs = sixNeighbors(cur);
    for (auto nbr : nbrs) {
      int adr = toadr(nbr);
      // if (visited_flag_[adr] == 0)
      // {
      //   init_que.push(nbr);
      //   visited_flag_[adr] = 1;
      // }
    }
  }
  return start_idx;
}

inline vector<Eigen::Vector3i> FrontierFinder::sixNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(6);
  Eigen::Vector3i tmp;

  tmp = voxel - Eigen::Vector3i(1, 0, 0);
  neighbors[0] = tmp;
  tmp = voxel + Eigen::Vector3i(1, 0, 0);
  neighbors[1] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 1, 0);
  neighbors[2] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 1, 0);
  neighbors[3] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 0, 1);
  neighbors[4] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 0, 1);
  neighbors[5] = tmp;

  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::tenNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(10);
  Eigen::Vector3i tmp;
  int count = 0;

  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      if (x == 0 && y == 0) continue;
      tmp = voxel + Eigen::Vector3i(x, y, 0);
      neighbors[count++] = tmp;
    }
  }
  neighbors[count++] = tmp - Eigen::Vector3i(0, 0, 1);
  neighbors[count++] = tmp + Eigen::Vector3i(0, 0, 1);
  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::allNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(26);
  Eigen::Vector3i tmp;
  int count = 0;
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) {
        if (x == 0 && y == 0 && z == 0) continue;
        tmp = voxel + Eigen::Vector3i(x, y, z);
        neighbors[count++] = tmp;
      }
  return neighbors;
}

inline bool FrontierFinder::isNeighborUnknown(const Eigen::Vector3i& voxel) {
  // At least one neighbor is unknown
  auto nbrs = sixNeighbors(voxel);
  for (auto nbr : nbrs) {
    if (edt_env_->sdf_map_->getOccupancy(nbr) == SDFMap::UNKNOWN) return true;
  }
  return false;
}

inline int FrontierFinder::toadr(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->toAddress(idx);
}

inline bool FrontierFinder::knownfree(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::FREE;
}

inline bool FrontierFinder::inmap(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->isInMap(idx);
}


void FrontierFinder::drawFirstViews() {
  // 删除 MarkerArray 中的所有内容
  mk_array.markers.clear();
  mk_array.markers.resize(0);
  views_vis_pub.publish(mk_array);
  ros::Duration(0.05).sleep();

  // 发布新的内容
  int n = 0;
  for (auto ftr : frontiers_){
    // cout << "frontier index = " << n << ",   viewpoint size = " << ftr.viewpoints_.size() << endl;
    // cout << " viewpoint_[0] vis_num = " << ftr.viewpoints_[0].visib_num_  << ", pos = " <<  ftr.viewpoints_[0].pos_.transpose()
    // << ", pitch = " <<  ftr.viewpoints_[0].pitch_ << ", yaw = " <<  ftr.viewpoints_[0].yaw_ << endl; 
    Vector3d pos = ftr.viewpoints_[0].pos_;
    double yaw = ftr.viewpoints_[0].yaw_;
    double pitch = ftr.viewpoints_[0].pitch_;
    // cout << "view = " << pos << yaw << endl;
    percep_utils_->setPose(pos, pitch, yaw);
    vector<Eigen::Vector3d> list1, list2;
    percep_utils_->getFOV(list1, list2);
    // cout <<"l1,l2 = "<<l1.size() <<l2.size() <<endl;
  

    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = n;
    mk.ns = "current_pose";
    mk.type = visualization_msgs::Marker::LINE_LIST;
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.r = 0.0;
    mk.color.g = 1.0;
    mk.color.b = 0.0;
    mk.color.a = 1.0;
    mk.scale.x = 0.04;
    mk.scale.y = 0.04;
    mk.scale.z = 0.04;



    if (list1.size() == 0) return;

    // Pub new marker
    geometry_msgs::Point pt;
    for (int i = 0; i < int(list1.size()); ++i) {
      pt.x = list1[i](0);
      pt.y = list1[i](1);
      pt.z = list1[i](2);
      mk.points.push_back(pt);

      pt.x = list2[i](0);
      pt.y = list2[i](1);
      pt.z = list2[i](2);
      mk.points.push_back(pt);
    }
    mk.action = visualization_msgs::Marker::ADD;
    // if (n==0) 
    mk_array.markers.push_back(mk);
    n++;

  }
  views_vis_pub.publish(mk_array);
}

void FrontierFinder::drawAllViews() {
  // 删除 MarkerArray 中的所有内容
  mk_array.markers.clear();
  mk_array.markers.resize(0);
  views_vis_pub.publish(mk_array);
  ros::Duration(0.005).sleep();
  // for(int i=0; i < 10; i++){
  //   views_vis_pub.publish(mk_array);
  //   ros::Duration(0.02).sleep();
  // }

  // 发布新的内容
  int n = 0,n_i = 0;
  for (auto ftr : frontiers_){
    // if (n>0) break;
    cout << n << ", viewpoint size = " << ftr.viewpoints_.size() << endl;
    cout << ftr.average_ << endl;
    int N = min(10, int(ftr.viewpoints_.size()));
    for (int i = 0; i< N; i++){
      cout << "      viewpoint vis_num = " << ftr.viewpoints_[i].visib_num_  << ", pos = " <<  ftr.viewpoints_[i].pos_.transpose()
      << ", pitch = " <<  ftr.viewpoints_[i].pitch_  << ", yaw = " <<  ftr.viewpoints_[i].yaw_ << endl; 
      Vector3d pos = ftr.viewpoints_[i].pos_;
      double yaw = ftr.viewpoints_[i].yaw_;
      double pitch = ftr.viewpoints_[i].pitch_;
      // cout << "view = " << pos << yaw << endl;
      percep_utils_->setPose(pos, pitch, yaw);
      vector<Eigen::Vector3d> list1, list2;
      percep_utils_->getFOV(list1, list2);
      // cout <<"l1,l2 = "<<l1.size() <<l2.size() <<endl;
    

      visualization_msgs::Marker mk;
      mk.header.frame_id = "world";
      mk.header.stamp = ros::Time::now();
      mk.id = n_i;
      mk.ns = "current_pose";
      mk.type = visualization_msgs::Marker::LINE_LIST;
      mk.pose.orientation.x = 0.0;
      mk.pose.orientation.y = 0.0;
      mk.pose.orientation.z = 0.0;
      mk.pose.orientation.w = 1.0;
      mk.color.r = 0.0;
      mk.color.g = 1.0;
      mk.color.b = 0.0;
      mk.color.a = 1.0;
      mk.scale.x = 0.04;
      mk.scale.y = 0.04;
      mk.scale.z = 0.04;

      if (list1.size() == 0) return;

      // Pub new marker
      geometry_msgs::Point pt;
      for (int i = 0; i < int(list1.size()); ++i) {
        pt.x = list1[i](0);
        pt.y = list1[i](1);
        pt.z = list1[i](2);
        mk.points.push_back(pt);

        pt.x = list2[i](0);
        pt.y = list2[i](1);
        pt.z = list2[i](2);
        mk.points.push_back(pt);
      }
      mk.action = visualization_msgs::Marker::ADD;
      if (n<2) mk_array.markers.push_back(mk);
      n_i++;
    }
    n++;

  }
  cout << "mk_array size = " << mk_array.markers.size() << endl;
  if(mk_array.markers.size() > 0) views_vis_pub.publish(mk_array);
}


void FrontierFinder::drawPathViews(vector<Vector3d> & path_pos, vector<double> & path_pitchs, vector<double> & path_yaws){
  // 删除 MarkerArray 中的所有内容
  path_views_mks.markers.clear();
  path_views_mks.markers.resize(0);
  path_views_vis_pub.publish(path_views_mks);
  // for(int i=0; i < 10; i++){
  //   path_views_vis_pub.publish(path_views_mks);
  //   ros::Duration(0.02).sleep();
  // }
  // 发布新的内容
  int n = 0;
  for(int i=0; i < path_pos.size(); i++){

    Vector3d pos = path_pos[i];
    double yaw = path_yaws[i];
    double pitch = path_pitchs[i];
    // cout << "view = " << pos << yaw << endl;
    percep_utils_->setPose(pos,pitch,yaw);
    vector<Eigen::Vector3d> list1, list2;
    percep_utils_->getFOV(list1, list2);
    // cout <<"l1,l2 = "<<l1.size() <<l2.size() <<endl;
  

    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = n;
    mk.ns = "current_pose";
    mk.type = visualization_msgs::Marker::LINE_LIST;
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.r = 1.0;
    mk.color.g = 0.84;
    mk.color.b = 0.0;
    mk.color.a = 1.0;
    mk.scale.x = 0.08;
    mk.scale.y = 0.08;
    mk.scale.z = 0.08;

    if (list1.size() == 0) return;

    // Pub new marker
    geometry_msgs::Point pt;
    for (int i = 0; i < int(list1.size()); ++i) {
      pt.x = list1[i](0);
      pt.y = list1[i](1);
      pt.z = list1[i](2);
      mk.points.push_back(pt);

      pt.x = list2[i](0);
      pt.y = list2[i](1);
      pt.z = list2[i](2);
      mk.points.push_back(pt);
    }
    mk.action = visualization_msgs::Marker::ADD;
    path_views_mks.markers.push_back(mk);
    n++;

  }

  path_views_vis_pub.publish(path_views_mks);



}




void FrontierFinder::drawTraj(const vector<Vector3d>& plan_path){
  // 创建路径消息的可视化对象
  visualization_msgs::Marker traj_marker;
  ros::Time current_time = ros::Time::now();
  traj_marker.header.frame_id = "world";
  traj_marker.header.stamp = current_time;
  traj_marker.ns = "trajectory";
  traj_marker.id = 0;
  traj_marker.type = visualization_msgs::Marker::LINE_STRIP;
  traj_marker.action = visualization_msgs::Marker::ADD;
  traj_marker.pose.orientation.w = 1.0;
  traj_marker.scale.x = 0.02; // 设置线条的宽度

  // 设置路径线条的颜色为绿色
  traj_marker.color.r = 0.0;
  traj_marker.color.g = 1.0;
  traj_marker.color.b = 0.0;
  traj_marker.color.a = 1.0; // 不透明度

  // 设置路径线条的顶点
  for (int i = 0; i < plan_path.size(); i++) {
    geometry_msgs::Point point;
    point.x = plan_path[i][0];
    point.y = plan_path[i][1];
    point.z = plan_path[i][2];
    traj_marker.points.push_back(point);
  }

  // 发布可视化消息
  traj_pub.publish(traj_marker);

  // // 循环输出plan_path，plan_path是全局路径
  // cout << "explptation plan_path = " << endl;
  // for (int i = 0; i < plan_path.size(); i++) {
  //     cout  << plan_path[i].transpose() << "->";
  // }
  // cout << endl;

}






}  // namespace fast_planner