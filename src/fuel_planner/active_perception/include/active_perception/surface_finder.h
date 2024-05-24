#ifndef _SURFACE_FINDER_H_
#define _SURFACE_FINDER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <list>
#include <utility>
#include <string>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <active_perception/frontier_finder.h>
#include <Eigen/Eigen>
#include <nav_msgs/Path.h>

using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;
using std::list;
using std::pair;
using std::string;
using namespace std;

class RayCaster;

namespace fast_planner {
class EDTEnvironment;
class PerceptionUtils;
class FrontierFinder;
struct Viewpoint;


struct PointWithUncertainty {
    pcl::PointXYZRGB point;
    Eigen::Vector3d normal;
    double uncertainty;
};


// A surface cluster, the viewpoints to cover it
struct SurfaceCluster {
    // vector<pcl::PointXYZRGB> points;
    vector<Vector3d> points_;
    vector<Eigen::Vector3d> normals_;
    vector<double> uncers_;

    // Average position of all voxels
    Vector3d average_;
    // Idx of cluster
    int id_;
    // Viewpoints that can cover the cluster
    vector<Viewpoint> viewpoints_;
    // Path and cost from this cluster to other clusters
    list<vector<Vector3d>> paths_;
    list<double> costs_;



    // 析构函数定义
    clearAll() {
        // 清空容器中的元素
        points_.clear();
        normals_.clear();
        uncers_.clear();
        viewpoints_.clear();

        // 释放可能需要手动释放的资源
        // ...

        // 将结构体的成员变量重置为默认值（可选）
        average_ = Vector3d();
        id_ = 0;
    }

};




class SurfaceFinder {
public:
  SurfaceFinder(const shared_ptr<EDTEnvironment>& edt, ros::NodeHandle& nh);
  ~SurfaceFinder();

  void pubmeshCallback();
  void get_surface_points(const Eigen::Vector3d& cur_pos);
  Eigen::MatrixXf readTextFile(const std::string& filename);
  void clusterPoints(vector<PointWithUncertainty>& points, vector<vector<PointWithUncertainty>>& clusters);
  void applyColorMapToClusters();
  void test(const ros::TimerEvent& e);
  void computeSurfaceclustersToVisit();
  void sampleViewpoints(SurfaceCluster& surf_ctr);
  void wrapYaw(double& yaw);
  void computerViewInfo(const Eigen::Vector3d& pos,  const double& pitch, const double& yaw, const SurfaceCluster& surf_ctr, 
                     int& visib_num, double& view_gain);
  void getFullCostMatrix(const Vector3d& cur_pos, const double& cur_yaw, const double& cur_pitch, Eigen::MatrixXd& mat);
  void findGlobalTour(const Vector3d& cur_pos, const double& cur_yaw, const double& cur_pitch, vector<int>& indices);
  void getPathForTour(const Vector3d& pos, const vector<int>& surf_ctr_ids, vector<Vector3d>& path, 
            vector<Vector3d>& points, vector<double>& yaws, vector<double>& pitchs);
  void drawFirstViews();
  void drawAllViews();
  void drawTraj(const vector<Vector3d>& plan_path);
  void visualize(vector<Vector3d>& surface_path);
  void drawSampleViews(vector<Vector3d> & sample_pos, vector<double> & sample_pitchs, vector<double> & sample_yaws, vector<double> & uncers);


  ros::Timer mesh_pub_timer_, test_timer_;
  string mesh_base_dir,mesh_data_base_dir ;  // mesh path
  ros::Publisher mesh_pub, surface_clusters_pub;
  sensor_msgs::PointCloud2 mesh_msg, surface_clusters_msg;
  ros::Publisher views_vis_pub;
  ros::Publisher traj_pub;
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr surface_cloud, surface_clusters_cloud;
  vector<PointWithUncertainty> points;
  vector<vector<PointWithUncertainty>> points_clusters;
  vector<SurfaceCluster> surface_clusters;

  shared_ptr<PerceptionUtils> percep_utils_;
  shared_ptr<EDTEnvironment> edt_env_;

  // Utils
  unique_ptr<RayCaster> raycaster_;


  visualization_msgs::MarkerArray sample_views_mks;

  // path length
  double points_radius_;
  double cluster_radius_;
  int cluster_num_;
  double max_path_length_;






private:

  bool isNearUnknown(const Vector3d& pos);

  // Params
  int cluster_min_;
  double cluster_size_xy_, cluster_size_z_;
  double candidate_rmax_, candidate_rmin_, candidate_dphi_, min_candidate_dist_, min_candidate_clearance_;
  int down_sample_;
  double min_view_finish_fraction_, resolution_;
  int min_visib_num_, candidate_rnum_;
  
  string tsp_dir_;

  vector<Vector3d> surf_points_;
  vector<double> surf_pitchs_;
  vector<double> surf_yaws_;

  






};

}  // namespace fast_planner
#endif