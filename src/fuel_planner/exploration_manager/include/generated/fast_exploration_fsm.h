#ifndef _FAST_EXPLORATION_FSM_H_
#define _FAST_EXPLORATION_FSM_H_

#include <Eigen/Eigen>

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Empty.h>
#include <nav_msgs/Odometry.h>
#include <opencv2/opencv.hpp>


#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <thread>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

using Eigen::Vector3d;
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::string;

#define METHOD_INDEX 6

namespace fast_planner {
class FastPlannerManager;
class FastExplorationManager;
class PlanningVisualization;
struct FSMParam;
struct FSMData;

enum EXPL_STATE { INIT, WAIT_TRIGGER, PLAN_TRAJ, PUB_TRAJ, EXEC_TRAJ, FINISH, TEST, RUN};



class FastExplorationFSM {
private:

  shared_ptr<FSMParam> fp_;
  shared_ptr<FSMData> fd_;
  EXPL_STATE state_;

  bool classic_;

  /* ROS utils */
  ros::NodeHandle node_;
  ros::Timer exec_timer_, safety_timer_, vis_timer_, frontier_timer_, getdata_timer_, http_timer_,send_data_timer_;
  ros::Subscriber trigger_sub_, odom_sub_;
  ros::Publisher odom_pub_,replan_pub_, new_pub_, bspline_pub_;
  ros::Publisher depth_pub, pose_pub;
  
  /* helper functions */
  int callExplorationPlanner();
  void transitState(EXPL_STATE new_state, string pos_call);

  /* ROS functions */
  void getdataCallback(const ros::TimerEvent& e);
  void httpCallback(const ros::TimerEvent& e);
  void FSMCallback(const ros::TimerEvent& e);
  void safetyCallback(const ros::TimerEvent& e);
  void frontierCallback(const ros::TimerEvent& e);
  void triggerCallback(const nav_msgs::PathConstPtr& msg);
  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);
  void visualize();
  void clearVisMarker();
  void get_picture();
  void is_finish();
  void publishImage();
  void read_data();
  void send_data_periodicCallback(const ros::TimerEvent& event);
  void addDepthNoise(cv::Mat& depth_mat, double far);

  // Metric
  bool createFolder(const std::string& folderPath);
  void removeIfExists(const std::string& filename);
  void createEmptyFileIfNotExists(const std::string& filename);
  void save_metric();
  ros::Timer sample_test_views_timer_;
  void sample_test_views_periodicCallback(const ros::TimerEvent& event);
  void sampleTestViews(vector<Vector3d> & sample_pos, vector<double> & sample_pitchs, 
                           vector<double> & sample_yaws, vector<double> & uncers, int N);
  std::vector<Eigen::MatrixXf> readMetric(const std::string& filename, int col_num);
  void demo_traj();




public:
  FastExplorationFSM(/* args */) {
  }
  ~FastExplorationFSM() {
  }

  void init(ros::NodeHandle& nh);


  /* planning utils */
  shared_ptr<FastExplorationManager> expl_manager_;
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<PlanningVisualization> visualization_;
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace fast_planner

#endif
