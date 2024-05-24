#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <vector>
#include <bspline/Bspline.h>
#include <nav_msgs/Path.h>

using std::vector;
using Eigen::Vector3d;

namespace fast_planner {
struct FSMData {
  // FSM data
  bool trigger_, have_odom_, static_state_;
  vector<string> state_str_;

  Eigen::Vector3d odom_pos_, odom_vel_;  // odometry state
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;

  Eigen::Vector3d start_pt_, start_vel_, start_acc_, start_dir_;  // start state

  vector<Eigen::Vector3d> start_poss;
  bspline::Bspline newest_traj_;
  

};

struct FSMParam {
  double replan_thresh1_;
  double replan_thresh2_;
  double replan_thresh3_;
  double replan_time_;  // second
};

struct plannerResult {
  int res_state_;
  vector<Eigen::Vector3d> res_pts_; 
  vector<double> res_yaws_;
  vector<double> res_pitchs_;  
};

struct ExplorationData {
  

  vector<vector<Vector3d>> frontiers_;
  vector<vector<Vector3d>> dead_frontiers_;
  vector<pair<Vector3d, Vector3d>> frontier_boxes_;
  vector<Vector3d> points_;
  vector<double> pitchs_;
  vector<double> yaws_;
  vector<Vector3d> averages_;
  vector<Vector3d> views_;
  vector<Vector3d> global_tour_;

  vector<int> refined_ids_;
  vector<vector<Vector3d>> n_points_;
  vector<vector<double>> n_pitchs_, n_yaws_;


  vector<Vector3d> unrefined_points_;
  vector<Vector3d> refined_points_;
  vector<double> refined_pitchs_, refined_yaws_;

  vector<Vector3d> refined_views_;  // points + dir(yaw)
  vector<Vector3d> refined_views1_, refined_views2_;
  vector<Vector3d> refined_tour_;

  vector<Vector3d> path_next_goal_;

  // viewpoint planning
  // vector<Vector4d> views_;
  vector<Vector3d> views_vis1_, views_vis2_;
  vector<Vector3d> centers_, scales_;

  // local ip
  string local_ip_;
   
  // model num
  int mode_num;
  int exp_mode_num;
  int rec_mode_num;
  bool is_only_exp;

  // exploration trajectpry
  vector<Vector3d> exp_path;
  int failed_cnt;
  bool is_exp;

  // surface trajectpry
  vector<Vector3d> surf_path;
  vector<Vector3d> surf_path1_, surf_path2_;
  vector<Vector3d> surf_points_;
  vector<double> surf_pitchs_;
  vector<double> surf_yaws_;
  vector<Vector3d> surf_path_next_goal_;

  // all trajectory
  vector<Eigen::Vector4d> all_path;
  ros::Publisher all_traj_pub;

  // exploration + reconstruction
  vector<Vector3d> all_points_;
  vector<double> all_pitchs_;
  vector<double> all_yaws_;


  // current pose
  Vector3d cur_pos;
  double cur_pitch, cur_yaw;

  //is 3D for Fuel / Fuel* with pitch
  bool is_3D;

  // metric
  int planning_num;
  string metric_path;
  double T_task, T_atsp, T_sp, T_switch;
  double path_length;
  vector<Eigen::Vector4d> single_path;
  vector<vector<double>> single_views;



};


struct ExplorationParam {
  // params
  bool refine_local_;
  int refined_num_;
  double refined_radius_;
  int top_view_num_;
  double max_decay_;
  string tsp_dir_;  // resource dir of tsp solver
  double relax_time_;
};

}  // namespace fast_planner

#endif